use super::intervention::*;
use super::tool_execution::*;
use super::*;

impl ContinuationHandler for AgentHandler {
    fn validate_payload(&self, payload: &Value) -> Result<(), GraphError> {
        let payload = decode_payload(payload).map_err(|error| {
            GraphError::GraphValidation(format!("invalid registered agent payload: {error}"))
        })?;
        match (payload.control_handler_key, self.controller.is_some()) {
            (Some(key), false) => Err(GraphError::MissingHandler(key)),
            (None, true) => Err(GraphError::GraphValidation(
                "registered agent controller is absent from its payload".into(),
            )),
            (Some(_), true) | (None, false) => Ok(()),
        }
    }

    fn start<'a>(
        &'a self,
        payload: &'a Value,
        state: Option<Value>,
        inputs: Vec<Value>,
        ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        async move { self.start_agent(payload, state, inputs, ctx).await }.boxed()
    }

    fn advance<'a>(
        &'a self,
        payload: &'a Value,
        checkpoint: Value,
        event: ContinuationEvent,
        ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        async move { self.advance_agent(payload, checkpoint, event, ctx).await }.boxed()
    }
}

impl AgentHandler {
    /// Configures one invocation and stores its initial controller boundary.
    async fn start_agent(
        &self,
        payload: &Value,
        state: Option<Value>,
        inputs: Vec<Value>,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let payload = decode_payload(payload)?;
        let input = single_input(inputs, "agent")?;
        let config = self
            .configure
            .configure(input.clone(), ctx.context().clone())
            .await?;
        let (resolved, message, budget) =
            resolve_agent_config(&payload, config, ctx.context()).await?;
        let session_id = self.session_id(&resolved, state)?;
        ctx.push_history(&session_id, &payload.agent_id, message)
            .await?;
        let selected_tools = resolved.tools.clone();
        let mut checkpoint = EdgeAgentCheckpoint {
            version: CHECKPOINT_VERSION,
            phase: EdgeAgentPhase::BeforeModel,
            session_id,
            input,
            resolved,
            selected_tools,
            budget,
            guidance: None,
            metrics: AgentLoopMetrics::default(),
            control_state: None,
        };
        if self.controller.is_none() {
            checkpoint.phase = normal_dispatch_phase(&checkpoint);
        }
        persist_checkpoint(checkpoint)
    }

    fn session_id(
        &self,
        resolved: &ResolvedAgentConfig,
        state: Option<Value>,
    ) -> Result<String, GraphError> {
        if resolved.keep_alive {
            Ok(restore_agent_state(state)?.unwrap_or_else(|| Uuid::now_v7().to_string()))
        } else {
            Ok(Uuid::now_v7().to_string())
        }
    }

    /// Validates and advances one serialized agent checkpoint event.
    async fn advance_agent(
        &self,
        payload: &Value,
        checkpoint: Value,
        event: ContinuationEvent,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let payload = decode_payload(payload)?;
        let mut checkpoint: EdgeAgentCheckpoint = from_value(checkpoint).map_err(|err| {
            GraphError::SnapshotValidation(format!("failed to decode agent checkpoint: {err}"))
        })?;
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(GraphError::UnsupportedVersion {
                format: "agent checkpoint",
                got: checkpoint.version,
                expected: CHECKPOINT_VERSION,
            });
        }
        validate_checkpoint(&payload, &checkpoint)?;
        match event {
            ContinuationEvent::Poll => self.poll(&payload, &mut checkpoint, ctx).await,
            ContinuationEvent::ChildResult { call_id, output } => {
                self.child_result(&payload, &mut checkpoint, call_id, output, ctx)
                    .await
            }
            ContinuationEvent::Resume { input } => {
                self.resume_agent(&payload, &mut checkpoint, input)
            }
        }
    }

    /// Advances the explicit phase currently stored in the checkpoint.
    async fn poll(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        match checkpoint.phase.clone() {
            EdgeAgentPhase::BeforeModel => {
                self.control_or_continue(
                    payload,
                    checkpoint,
                    AgentInterventionPoint::BeforeModel,
                    ctx,
                )
                .await
            }
            EdgeAgentPhase::Dispatch { conclusion } => {
                self.dispatch(payload, checkpoint, conclusion, ctx).await
            }
            EdgeAgentPhase::BeforeTools { .. } => {
                self.control_or_continue(
                    payload,
                    checkpoint,
                    AgentInterventionPoint::BeforeTools,
                    ctx,
                )
                .await
            }
            EdgeAgentPhase::AcceptedTools { .. } => {
                self.accept_staged_proposal(payload, checkpoint, ctx).await
            }
            EdgeAgentPhase::PendingTool { .. } => persist_checkpoint(checkpoint.clone()),
            EdgeAgentPhase::AfterTools { .. } => {
                self.control_or_continue(
                    payload,
                    checkpoint,
                    AgentInterventionPoint::AfterTools,
                    ctx,
                )
                .await
            }
        }
    }

    /// Evaluates the optional controller and applies its decision atomically.
    async fn control_or_continue(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        point: AgentInterventionPoint,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let decision = if let Some(controller) = &self.controller {
            let data = self.loop_data(payload, checkpoint, point, &ctx).await?;
            controller.control(data, ctx.context().clone()).await?
        } else {
            AgentDecision::continue_()
        };
        self.apply_decision(payload, checkpoint, point, decision)
    }

    /// Builds the owned read-only observation passed to one controller call.
    async fn loop_data(
        &self,
        payload: &AgentPayload,
        checkpoint: &EdgeAgentCheckpoint,
        point: AgentInterventionPoint,
        ctx: &ContinuationContext,
    ) -> Result<AgentLoopData, GraphError> {
        let history = ctx.history_for_session(&checkpoint.session_id).await;
        let proposal = match &checkpoint.phase {
            EdgeAgentPhase::BeforeTools { calls, .. } => {
                calls.iter().map(|call| call.proposal.clone()).collect()
            }
            _ => Vec::new(),
        };
        let results = match &checkpoint.phase {
            EdgeAgentPhase::AfterTools { results } => results.clone(),
            _ => Vec::new(),
        };
        let active_tools = effective_tools(
            &checkpoint.resolved.tools,
            &checkpoint.selected_tools,
            checkpoint.budget.as_ref(),
        );
        Ok(AgentLoopData {
            input: checkpoint.input.clone(),
            point,
            agent_id: payload.agent_id.clone(),
            session_id: checkpoint.session_id.clone(),
            configured_tools: tool_infos(payload, &checkpoint.resolved.tools),
            active_tools: tool_infos(payload, &active_tools),
            history,
            proposal,
            results,
            metrics: checkpoint.metrics.clone(),
            budget: checkpoint.budget.clone(),
            control_state: checkpoint.control_state.clone(),
        })
    }

    /// Validates one decision and returns its next checkpoint or suspension.
    fn apply_decision(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        point: AgentInterventionPoint,
        decision: AgentDecision,
    ) -> Result<ContinuationTransition, GraphError> {
        validate_decision(&decision)?;
        if let AgentDecisionKind::Abort(reason) = &decision.kind {
            return Err(GraphError::AgentPolicyAbort {
                agent: payload.agent_id.clone(),
                reason: reason.clone(),
            });
        }
        apply_control_state(checkpoint, decision.state);
        match decision.kind {
            AgentDecisionKind::Continue => self.continue_at(checkpoint, point),
            AgentDecisionKind::Redirect(directive) => {
                apply_directive(payload, checkpoint, directive)?;
                checkpoint.phase = redirect_phase(point, checkpoint);
                persist_checkpoint(checkpoint.clone())
            }
            AgentDecisionKind::Conclude(guidance) => {
                checkpoint.guidance = Some(guidance);
                checkpoint.phase = EdgeAgentPhase::Dispatch {
                    conclusion: Some(ConclusionCause::Explicit),
                };
                persist_checkpoint(checkpoint.clone())
            }
            AgentDecisionKind::Suspend(value) => suspend_agent(payload, checkpoint, point, value),
            AgentDecisionKind::Abort(_) => Err(GraphError::Invalid(
                "agent abort decision escaped validation".into(),
            )),
        }
    }

    /// Commits the normal next phase for one accepted intervention boundary.
    fn continue_at(
        &self,
        checkpoint: &mut EdgeAgentCheckpoint,
        point: AgentInterventionPoint,
    ) -> Result<ContinuationTransition, GraphError> {
        match point {
            AgentInterventionPoint::BeforeTools => accept_staged_boundary(checkpoint),
            AgentInterventionPoint::BeforeModel | AgentInterventionPoint::AfterTools => {
                checkpoint.phase = normal_dispatch_phase(checkpoint);
                persist_checkpoint(checkpoint.clone())
            }
        }
    }

    /// Decodes an external agent resume value and applies it at the saved point.
    fn resume_agent(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        input: Value,
    ) -> Result<ContinuationTransition, GraphError> {
        let resume: AgentResume = from_value(input).map_err(|err| {
            GraphError::AgentResumeValidation(format!("failed to decode AgentResume: {err}"))
        })?;
        let point = checkpoint_point(&checkpoint.phase).ok_or_else(|| {
            GraphError::AgentResumeValidation(
                "agent checkpoint is not at an intervention boundary".into(),
            )
        })?;
        let decision = match resume {
            AgentResume::Continue => AgentDecision::continue_(),
            AgentResume::Redirect { guidance, tools } => {
                AgentDecision::redirect_names(guidance, tools)
            }
            AgentResume::Conclude { guidance } => AgentDecision::conclude(guidance),
            AgentResume::Abort { reason } => AgentDecision::abort(reason),
        };
        self.apply_decision(payload, checkpoint, point, decision)
    }

    /// Performs one model request with the active tool surface and guidance.
    async fn dispatch(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        conclusion: Option<ConclusionCause>,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let concluding = conclusion.is_some();
        let request_ctx = ctx.context();
        let tools = if concluding {
            Vec::new()
        } else {
            let active = effective_tools(
                &checkpoint.resolved.tools,
                &checkpoint.selected_tools,
                checkpoint.budget.as_ref(),
            );
            tool_definitions(payload, &active)
        };
        let options = client_options(payload, checkpoint, tools);
        let client = request_ctx
            .client_factory()
            .create(&checkpoint.resolved.model, options)
            .map_err(|err| GraphError::AgentClient(format!("client creation failed: {err}")))?;
        ctx.validate_history_for_session(&checkpoint.session_id)
            .await?;
        let history = ctx.history_for_session(&checkpoint.session_id).await;
        let mut messages = materialize_messages(&history, request_ctx)
            .await
            .map_err(|err| GraphError::Invalid(format!("message materialization failed: {err}")))?;
        append_guidance(&mut messages, checkpoint.guidance.as_deref());
        if matches!(conclusion, Some(ConclusionCause::TurnBudget)) {
            append_budget_reminder(&mut messages, client.as_ref(), payload);
        }
        let response = client
            .execute(&messages)
            .await
            .map_err(|err| GraphError::AgentClient(format!("execution failed: {err}")))?;
        match response.output {
            ClientOutput::Output(output) => {
                self.complete_output(payload, checkpoint, output, response.usage, concluding, ctx)
                    .await
            }
            ClientOutput::ToolCalls { thought: _, calls } if concluding => {
                Err(GraphError::AgentConclusion {
                    agent: payload.agent_id.clone(),
                    reason: format!(
                        "tool-disabled final turn proposed {} tool call(s)",
                        calls.len()
                    ),
                })
            }
            ClientOutput::ToolCalls { thought, calls } => {
                self.stage_proposal(checkpoint, thought, calls, response.usage)
            }
        }
    }

    /// Validates and commits one final structured model response.
    async fn complete_output(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        output: JsonValue,
        usage: Option<crate::clients::TokenUsage>,
        concluding: bool,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        validate_agent_output(payload, &output, concluding)?;
        let content = serde_json::to_string(&output).map_err(|err| {
            GraphError::Invalid(format!("failed to serialize agent output: {err}"))
        })?;
        let message = match usage {
            Some(usage) => Message::assistant(content).with_usage(usage),
            None => Message::assistant(content),
        };
        let output = to_value(output).map_err(|err| GraphError::ValueConversion {
            target: "agent output".into(),
            reason: err.to_string(),
        })?;
        checkpoint.metrics.record_output(usage)?;
        ctx.push_history(&checkpoint.session_id, &payload.agent_id, message)
            .await?;
        ctx.compact_history(&checkpoint.session_id).await?;
        Ok(ContinuationTransition {
            checkpoint: None,
            state: completed_agent_state(checkpoint)?,
            outputs: vec![output],
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: None,
        })
    }

    /// Stores a complete model tool proposal before any history mutation.
    fn stage_proposal(
        &self,
        checkpoint: &mut EdgeAgentCheckpoint,
        thought: Option<String>,
        calls: Vec<ToolCall>,
        usage: Option<crate::clients::TokenUsage>,
    ) -> Result<ContinuationTransition, GraphError> {
        if calls.is_empty() {
            return Err(GraphError::AgentClient(
                "model returned an empty tool-call batch".into(),
            ));
        }
        let staged = stage_tool_calls(calls)?;
        let proposals = staged
            .iter()
            .map(|call| call.proposal.clone())
            .collect::<Vec<_>>();
        checkpoint.metrics.record_proposal(&proposals, usage)?;
        checkpoint.guidance = None;
        checkpoint.phase = EdgeAgentPhase::BeforeTools {
            thought,
            calls: staged,
            usage,
        };
        persist_checkpoint(checkpoint.clone())
    }

    /// Commits an accepted assistant proposal and prepares its tool child calls.
    async fn accept_staged_proposal(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let EdgeAgentPhase::AcceptedTools {
            thought,
            calls,
            usage,
        } = checkpoint.phase.clone()
        else {
            return Err(GraphError::AgentControlValidation(
                "BeforeTools decision has no staged proposal".into(),
            ));
        };
        let mut prepared = self.prepare_tool_calls(payload, checkpoint, &calls)?;
        let assistant = assistant_tool_call_message(thought, &calls, usage)?;
        let mut messages = Vec::with_capacity(1 + prepared.recoverable_messages.len());
        messages.push(assistant);
        messages.extend(std::mem::take(&mut prepared.recoverable_messages));
        let transition = self.begin_tool_execution(checkpoint, prepared)?;
        ctx.push_history_batch(&checkpoint.session_id, &payload.agent_id, messages)
            .await?;
        ctx.compact_history(&checkpoint.session_id).await?;
        Ok(transition)
    }

    /// Resolves an accepted proposal into executable and recoverable calls.
    fn prepare_tool_calls(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        calls: &[EdgeProposedToolCall],
    ) -> Result<PreparedToolCalls, GraphError> {
        let mut prepared = PreparedToolCalls {
            child_calls: Vec::new(),
            recoverable_messages: Vec::new(),
            running_tools: BTreeSet::new(),
            active: Vec::new(),
            waiting: Vec::new(),
            results: Vec::new(),
        };
        if checkpoint.budget.is_none() {
            for (position, call) in calls.iter().enumerate() {
                self.prepare_tool_call(
                    payload,
                    None,
                    &checkpoint.selected_tools,
                    call,
                    position,
                    &mut prepared,
                )?;
            }
            return Ok(prepared);
        }
        let exposed = effective_tools(
            &checkpoint.resolved.tools,
            &checkpoint.selected_tools,
            checkpoint.budget.as_ref(),
        )
        .into_owned();
        for (position, call) in calls.iter().enumerate() {
            self.prepare_tool_call(
                payload,
                checkpoint.budget.as_mut(),
                &exposed,
                call,
                position,
                &mut prepared,
            )?;
        }
        Ok(prepared)
    }

    /// Validates one proposed call against the active prepared tool surface.
    fn prepare_tool_call(
        &self,
        payload: &AgentPayload,
        budget: Option<&mut AgentBudgetState>,
        exposed: &[String],
        call: &EdgeProposedToolCall,
        position: usize,
        prepared: &mut PreparedToolCalls,
    ) -> Result<(), GraphError> {
        let proposal = &call.proposal;
        let Some(tool) = payload
            .tools
            .iter()
            .find(|tool| tool.name == proposal.tool_name() && exposed.contains(&tool.name))
        else {
            add_unavailable_result(proposal, position, prepared)?;
            return Ok(());
        };
        if !budget.is_none_or(|budget| budget.admit(&tool.name)) {
            add_unavailable_result(proposal, position, prepared)?;
            return Ok(());
        }
        let runtime = self.tool_runtime(tool.child_index)?;
        let json_args = serde_json::to_value(proposal.arguments()).map_err(|err| {
            GraphError::ValueConversion {
                target: format!("tool '{}' arguments", tool.name),
                reason: err.to_string(),
            }
        })?;
        let input = match (runtime.decode_args)(json_args) {
            Ok(input) => input,
            Err(err) if !err.is_fatal() => {
                add_decode_error_result(proposal, position, err, prepared)?;
                return Ok(());
            }
            Err(err) => return Err(GraphError::Invalid(err.to_string())),
        };
        add_executable_call(tool, proposal, position, input, prepared);
        Ok(())
    }

    /// Starts ready tool children or completes a fully synthetic result batch.
    fn begin_tool_execution(
        &self,
        checkpoint: &mut EdgeAgentCheckpoint,
        mut prepared: PreparedToolCalls,
    ) -> Result<ContinuationTransition, GraphError> {
        if prepared.child_calls.is_empty() {
            let results = finish_results(&mut prepared.results);
            checkpoint.metrics.record_results(&results)?;
            checkpoint.phase = if self.controller.is_some() {
                EdgeAgentPhase::AfterTools { results }
            } else {
                normal_dispatch_phase(checkpoint)
            };
            return persist_checkpoint(checkpoint.clone());
        }
        checkpoint.phase = EdgeAgentPhase::PendingTool {
            active: prepared.active,
            waiting: prepared.waiting,
            results: prepared.results,
        };
        transition_with_children(checkpoint.clone(), prepared.child_calls)
    }

    /// Persists and commits one returned tool child result.
    async fn child_result(
        &self,
        payload: &AgentPayload,
        checkpoint: &mut EdgeAgentCheckpoint,
        call_id: String,
        output: Value,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let active_call = take_active_call(payload, checkpoint, &call_id)?;
        let rendered = self.render_tool_result(payload, &active_call, output)?;
        let EdgeRenderedToolResult {
            message,
            value,
            error,
        } = rendered;
        let message = message.with_call_id(call_id);
        ctx.push_history(&checkpoint.session_id, &payload.agent_id, message)
            .await?;
        ctx.compact_history(&checkpoint.session_id).await?;
        self.commit_tool_result(checkpoint, active_call, value, error)
    }

    /// Converts one child output into history and controller-visible values.
    fn render_tool_result(
        &self,
        payload: &AgentPayload,
        active: &EdgeActiveToolCall,
        output: Value,
    ) -> Result<EdgeRenderedToolResult, GraphError> {
        let tool = payload
            .tools
            .get(active.child_index)
            .ok_or_else(|| GraphError::Invalid("tool child index is invalid".into()))?;
        let runtime = self.tool_runtime(active.child_index)?;
        match (runtime.render_result)(output) {
            Ok(result) => Ok(result),
            Err(EdgeToolMessageError::Recoverable(err)) if !err.is_fatal() => {
                recoverable_rendered_result(err, &tool.name)
            }
            Err(EdgeToolMessageError::Recoverable(err)) => {
                Err(GraphError::Invalid(err.to_string()))
            }
            Err(EdgeToolMessageError::Fatal {
                expected,
                reason,
                raw,
            }) => Err(GraphError::Invalid(format!(
                "tool '{}' output decode failed; expected {expected}: {reason}; raw: {raw}",
                tool.name
            ))),
        }
    }

    /// Records a tool result and schedules the next queued call for that tool.
    fn commit_tool_result(
        &self,
        checkpoint: &mut EdgeAgentCheckpoint,
        active_call: EdgeActiveToolCall,
        value: Value,
        error: bool,
    ) -> Result<ContinuationTransition, GraphError> {
        let result = AgentToolResult::new(
            active_call.call_id.clone(),
            active_call.tool_name.clone(),
            active_call.args.clone(),
            value,
            error,
        );
        let next = complete_active_call(checkpoint, active_call, result)?;
        if let Some((active, child)) = next {
            add_active_call(checkpoint, active)?;
            return transition_with_children(checkpoint.clone(), vec![child]);
        }
        self.finish_tool_round(checkpoint)
    }

    /// Enters `AfterTools` after every accepted call has completed.
    fn finish_tool_round(
        &self,
        checkpoint: &mut EdgeAgentCheckpoint,
    ) -> Result<ContinuationTransition, GraphError> {
        let EdgeAgentPhase::PendingTool {
            active,
            waiting,
            results,
        } = &mut checkpoint.phase
        else {
            return Err(GraphError::Invalid("agent tool phase disappeared".into()));
        };
        if !active.is_empty() || !waiting.is_empty() {
            return persist_checkpoint(checkpoint.clone());
        }
        let results = finish_results(results);
        checkpoint.metrics.record_results(&results)?;
        checkpoint.phase = if self.controller.is_some() {
            EdgeAgentPhase::AfterTools { results }
        } else {
            normal_dispatch_phase(checkpoint)
        };
        persist_checkpoint(checkpoint.clone())
    }

    fn tool_runtime(&self, index: usize) -> Result<&EdgeAgentToolRuntime, GraphError> {
        self.tools
            .get(index)
            .map(Arc::as_ref)
            .ok_or_else(|| GraphError::Invalid(format!("tool runtime {index} is missing")))
    }
}

/// Selects the one shared dispatch phase for normal and budget conclusion paths.
pub(super) fn normal_dispatch_phase(checkpoint: &EdgeAgentCheckpoint) -> EdgeAgentPhase {
    let conclusion = if can_dispatch_normally(checkpoint.budget.as_ref(), &checkpoint.metrics) {
        None
    } else {
        Some(ConclusionCause::TurnBudget)
    };
    EdgeAgentPhase::Dispatch { conclusion }
}

/// Checkpoints acceptance separately from history commit and tool scheduling.
fn accept_staged_boundary(
    checkpoint: &mut EdgeAgentCheckpoint,
) -> Result<ContinuationTransition, GraphError> {
    let EdgeAgentPhase::BeforeTools {
        thought,
        calls,
        usage,
    } = checkpoint.phase.clone()
    else {
        return Err(GraphError::AgentControlValidation(
            "BeforeTools decision has no staged proposal".into(),
        ));
    };
    checkpoint.phase = EdgeAgentPhase::AcceptedTools {
        thought,
        calls,
        usage,
    };
    persist_checkpoint(checkpoint.clone())
}

/// Performs full JSON Schema validation before final history mutation.
fn validate_agent_output(
    payload: &AgentPayload,
    output: &JsonValue,
    concluding: bool,
) -> Result<(), GraphError> {
    let result = jsonschema::validator_for(&payload.output_schema)
        .map_err(|error| {
            GraphError::GraphValidation(format!(
                "agent output schema '{}' cannot be compiled: {error}",
                payload.output_type_name
            ))
        })?
        .validate(output)
        .map_err(|error| GraphError::Schema {
            label: "agent structured output".into(),
            expected: payload.output_type_name.clone(),
            value: error.to_string(),
        });
    if concluding {
        result.map_err(|error| GraphError::AgentConclusion {
            agent: payload.agent_id.clone(),
            reason: format!("invalid structured output: {error}"),
        })
    } else {
        result
    }
}

fn tool_infos(payload: &AgentPayload, selected: &[String]) -> Vec<ToolInfo> {
    payload
        .tools
        .iter()
        .filter(|tool| selected.contains(&tool.name))
        .map(ToolInfo::from_payload)
        .collect()
}

fn tool_definitions(payload: &AgentPayload, selected: &[String]) -> Vec<ToolDefinition> {
    payload
        .tools
        .iter()
        .filter(|tool| selected.contains(&tool.name))
        .map(|tool| ToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        })
        .collect()
}

/// Builds provider options from the immutable payload and active tool subset.
fn client_options(
    payload: &AgentPayload,
    checkpoint: &EdgeAgentCheckpoint,
    tools: Vec<ToolDefinition>,
) -> ClientOptions {
    let tool_choice = if tools.is_empty() {
        ToolChoice::Disabled
    } else {
        ToolChoice::Auto
    };
    ClientOptions {
        output_type_name: payload.output_type_name.clone(),
        provider_config: checkpoint.resolved.provider_config.clone(),
        ..ClientOptions::default()
    }
    .with_input_schema(payload.input_schema.clone())
    .with_tools(tools)
    .with_tool_choice(tool_choice)
    .with_name(payload.agent_id.clone())
    .with_output_schema(payload.output_schema.clone())
    .with_preamble(effective_preamble(payload, &checkpoint.resolved))
}

fn append_guidance(messages: &mut Vec<Message>, guidance: Option<&str>) {
    let Some(guidance) = guidance else {
        return;
    };
    messages.push(Message {
        role: Role::System,
        content: format!("<pravah_agent_intervention>\n{guidance}\n</pravah_agent_intervention>"),
        attachments: Vec::new(),
        usage: None,
    });
}

/// Appends Rath's provider-aware reminder for an automatic conclusion turn.
fn append_budget_reminder(
    messages: &mut Vec<Message>,
    client: &dyn crate::clients::Client,
    payload: &AgentPayload,
) {
    let exit_tool = (client.uses_exit_tool() && !payload.output_type_name.is_empty())
        .then_some(payload.output_type_name.as_str());
    messages.push(Message::user(client.default_turn_budget_message(exit_tool)));
}
