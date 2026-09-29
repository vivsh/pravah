use super::effects::*;
use super::intervention::*;
use super::tool_execution::*;
use super::*;

impl ContinuationHandler for AgentHandler {
    fn validate_payload(&self, payload: &Value) -> Result<(), GraphError> {
        self.validated_payload(payload).map(|_| ())
    }

    fn start<'a>(
        &'a self,
        payload: &'a Value,
        state: Option<Value>,
        inputs: Vec<Value>,
        _ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let metadata = AgentPayloadView::read(payload)?;
        let input = single_input(inputs, "agent")?;
        let hook = AgentHook {
            version: 1,
            handler: metadata.agent_id.into(),
            payload: payload.clone(),
            operation: AgentHookOperation::Configure {
                input: input.clone(),
            },
        };
        effect(
            AgentEffectCheckpoint::Configure {
                version: CHECKPOINT_VERSION,
                input,
                state,
            },
            hook.into_request()?,
        )
    }

    fn advance<'a>(
        &'a self,
        payload: &'a Value,
        checkpoint: Value,
        event: ContinuationEvent,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        if checkpoint.get("effect").is_some() {
            let checkpoint = AgentEffectCheckpoint::from_value(&checkpoint)?;
            return self.advance_effect(payload, checkpoint, event, ctx);
        }
        self.advance_agent(payload, checkpoint, event, ctx)
    }
}

impl AgentHandler {
    /// Validates and advances one serialized agent checkpoint event.
    fn advance_agent(
        &self,
        payload: &Value,
        checkpoint: Value,
        event: ContinuationEvent,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let payload = AgentPayloadView::read(payload)?;
        let checkpoint = EdgeAgentCheckpoint::from_value(&checkpoint)?;
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(GraphError::UnsupportedVersion {
                format: "agent checkpoint",
                got: checkpoint.version,
                expected: CHECKPOINT_VERSION,
            });
        }
        validate_checkpoint_progress(&payload.tools, &checkpoint)?;
        match event {
            ContinuationEvent::Fetch { .. } => {
                Err(GraphError::Invalid("unexpected agent Fetch outcome".into()))
            }
            ContinuationEvent::Poll => self.poll(&payload, checkpoint, ctx),
            ContinuationEvent::ChildResult { call_id, output } => {
                self.child_result(&payload, checkpoint, call_id, output, ctx)
            }
            ContinuationEvent::Resume { input } => self.resume_agent(&payload, checkpoint, input),
        }
    }

    /// Advances the explicit phase currently stored in the checkpoint.
    fn poll(
        &self,
        payload: &AgentPayloadView<'_>,
        checkpoint: EdgeAgentCheckpoint,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        match &checkpoint.phase {
            EdgeAgentPhase::BeforeModel => self.control_or_continue(
                payload,
                checkpoint,
                AgentInterventionPoint::BeforeModel,
                ctx,
            ),
            EdgeAgentPhase::Dispatch { conclusion } => {
                let conclusion = *conclusion;
                self.dispatch(payload, checkpoint, conclusion, ctx)
            }
            EdgeAgentPhase::BeforeTools { .. } => self.control_or_continue(
                payload,
                checkpoint,
                AgentInterventionPoint::BeforeTools,
                ctx,
            ),
            EdgeAgentPhase::AcceptedTools { .. } => {
                self.accept_staged_proposal(payload, checkpoint, ctx)
            }
            EdgeAgentPhase::PendingTool { .. } => persist_checkpoint(checkpoint),
            EdgeAgentPhase::AfterTools { .. } => self.control_or_continue(
                payload,
                checkpoint,
                AgentInterventionPoint::AfterTools,
                ctx,
            ),
        }
    }

    /// Evaluates the optional controller and applies its decision atomically.
    fn control_or_continue(
        &self,
        payload: &AgentPayloadView<'_>,
        checkpoint: EdgeAgentCheckpoint,
        point: AgentInterventionPoint,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        if self.controller.is_none() {
            return self.apply_decision(payload, checkpoint, point, AgentDecision::continue_());
        }
        let observation = self.loop_data(payload, &checkpoint, point, &ctx)?;
        let hook = AgentHook {
            version: 1,
            handler: payload.agent_id.into(),
            payload: payload.raw.clone(),
            operation: AgentHookOperation::Control { observation },
        };
        effect(
            AgentEffectCheckpoint::Control {
                version: CHECKPOINT_VERSION,
                checkpoint: checkpoint.into_value()?,
            },
            hook.into_request()?,
        )
    }

    /// Builds the owned read-only observation passed to one controller call.
    fn loop_data(
        &self,
        payload: &AgentPayloadView<'_>,
        checkpoint: &EdgeAgentCheckpoint,
        point: AgentInterventionPoint,
        ctx: &ContinuationContext<'_>,
    ) -> Result<AgentLoopData, GraphError> {
        let history = ctx.history().for_session(&checkpoint.session_id);
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
        let configured_tools = checkpoint.configured_tools()?;
        let active_tools = effective_tools(
            &configured_tools,
            &checkpoint.selected_tools,
            checkpoint.budget.as_ref(),
        );
        Ok(AgentLoopData {
            input: checkpoint.input.clone(),
            point,
            agent_id: payload.agent_id.to_owned(),
            session_id: checkpoint.session_id.clone(),
            configured_tools: tool_infos(payload, &configured_tools),
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
    pub(super) fn apply_decision(
        &self,
        payload: &AgentPayloadView<'_>,
        mut checkpoint: EdgeAgentCheckpoint,
        point: AgentInterventionPoint,
        decision: AgentDecision,
    ) -> Result<ContinuationTransition, GraphError> {
        validate_decision(&decision)?;
        if let AgentDecisionKind::Abort(reason) = &decision.kind {
            return Err(GraphError::AgentPolicyAbort {
                agent: payload.agent_id.to_owned(),
                reason: reason.clone(),
            });
        }
        apply_control_state(&mut checkpoint, decision.state);
        match decision.kind {
            AgentDecisionKind::Continue => self.continue_at(checkpoint, point),
            AgentDecisionKind::Redirect(directive) => {
                apply_directive(payload, &mut checkpoint, directive)?;
                checkpoint.phase = redirect_phase(point, &checkpoint);
                persist_checkpoint(checkpoint)
            }
            AgentDecisionKind::Conclude(guidance) => {
                checkpoint.guidance = Some(guidance);
                checkpoint.phase = EdgeAgentPhase::Dispatch {
                    conclusion: Some(ConclusionCause::Explicit),
                };
                persist_checkpoint(checkpoint)
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
        mut checkpoint: EdgeAgentCheckpoint,
        point: AgentInterventionPoint,
    ) -> Result<ContinuationTransition, GraphError> {
        match point {
            AgentInterventionPoint::BeforeTools => accept_staged_boundary(checkpoint),
            AgentInterventionPoint::BeforeModel | AgentInterventionPoint::AfterTools => {
                checkpoint.phase = normal_dispatch_phase(&checkpoint);
                persist_checkpoint(checkpoint)
            }
        }
    }

    /// Decodes an external agent resume value and applies it at the saved point.
    fn resume_agent(
        &self,
        payload: &AgentPayloadView<'_>,
        checkpoint: EdgeAgentCheckpoint,
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

    /// Freezes one preparation request without calling a client or a policy.
    fn dispatch(
        &self,
        payload: &AgentPayloadView<'_>,
        checkpoint: EdgeAgentCheckpoint,
        conclusion: Option<ConclusionCause>,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let request = super::request::generation(payload, &checkpoint, conclusion.is_some())?;
        let mut guidance = Vec::new();
        append_guidance(&mut guidance, checkpoint.guidance.as_deref());
        let preparation = super::effect_values::preparation_request(
            &checkpoint,
            request,
            guidance,
            matches!(conclusion, Some(ConclusionCause::TurnBudget)),
            ctx,
        )?;
        effect(
            AgentEffectCheckpoint::Prepare {
                version: CHECKPOINT_VERSION,
                checkpoint: checkpoint.into_value()?,
            },
            preparation,
        )
    }

    /// Interprets an accepted model outcome; errors cannot repeat generation.
    pub(super) fn accept_generation(
        &self,
        payload: &AgentPayloadView<'_>,
        checkpoint: EdgeAgentCheckpoint,
        response: crate::clients::ClientResponse,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let EdgeAgentPhase::Dispatch { conclusion } = checkpoint.phase else {
            return Err(GraphError::SnapshotValidation(
                "invalid generation phase".into(),
            ));
        };
        let concluding = conclusion.is_some();
        match response.output {
            ClientOutput::Output(output) => {
                self.complete_output(payload, checkpoint, output, response.usage, concluding, ctx)
            }
            ClientOutput::ToolCalls { text: _, calls } if concluding => {
                Err(GraphError::AgentConclusion {
                    agent: payload.agent_id.to_owned(),
                    reason: format!(
                        "tool-disabled final turn proposed {} tool call(s)",
                        calls.len()
                    ),
                })
            }
            ClientOutput::ToolCalls {
                text: thought,
                calls,
            } => self.stage_proposal(checkpoint, thought, calls, response.usage),
        }
    }

    /// Validates and commits one final structured model response.
    fn complete_output(
        &self,
        payload: &AgentPayloadView<'_>,
        mut checkpoint: EdgeAgentCheckpoint,
        output: JsonValue,
        usage: Option<crate::clients::TokenUsage>,
        concluding: bool,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        validate_agent_output(payload, &output, concluding, &ctx)?;
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
        let transition = ContinuationTransition {
            checkpoint: None,
            state: completed_agent_state(&checkpoint)?,
            outputs: vec![output],
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: None,
            ..Default::default()
        };
        record(
            ctx,
            &checkpoint.session_id,
            payload.agent_id,
            vec![message],
            transition,
        )
    }

    /// Stores a complete model tool proposal before any history mutation.
    fn stage_proposal(
        &self,
        mut checkpoint: EdgeAgentCheckpoint,
        thought: Option<String>,
        calls: Vec<ToolCall>,
        usage: Option<crate::clients::TokenUsage>,
    ) -> Result<ContinuationTransition, GraphError> {
        if calls.is_empty() {
            return Err(GraphError::AgentResponseValidation(
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
        persist_checkpoint(checkpoint)
    }

    /// Commits an accepted assistant proposal and prepares its tool child calls.
    fn accept_staged_proposal(
        &self,
        payload: &AgentPayloadView<'_>,
        mut checkpoint: EdgeAgentCheckpoint,
        ctx: ContinuationContext<'_>,
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
        let mut prepared = self.prepare_tool_calls(payload, &mut checkpoint, &calls)?;
        let assistant = assistant_tool_call_message(thought, &calls, usage)?;
        let mut messages = Vec::with_capacity(1 + prepared.recoverable_messages.len());
        messages.push(assistant);
        messages.extend(std::mem::take(&mut prepared.recoverable_messages));
        let session = checkpoint.session_id.clone();
        let transition = self.begin_tool_execution(checkpoint, prepared)?;
        record(ctx, &session, payload.agent_id, messages, transition)
    }

    /// Resolves an accepted proposal into executable and recoverable calls.
    fn prepare_tool_calls(
        &self,
        payload: &AgentPayloadView<'_>,
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
        let configured_tools = checkpoint.configured_tools()?;
        let exposed = effective_tools(
            &configured_tools,
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
        payload: &AgentPayloadView<'_>,
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
        mut checkpoint: EdgeAgentCheckpoint,
        mut prepared: PreparedToolCalls,
    ) -> Result<ContinuationTransition, GraphError> {
        if prepared.child_calls.is_empty() {
            let results = finish_results(&mut prepared.results);
            checkpoint.metrics.record_results(&results)?;
            checkpoint.phase = if self.controller.is_some() {
                EdgeAgentPhase::AfterTools { results }
            } else {
                normal_dispatch_phase(&checkpoint)
            };
            return persist_checkpoint(checkpoint);
        }
        checkpoint.phase = EdgeAgentPhase::PendingTool {
            active: prepared.active,
            waiting: prepared.waiting,
            results: prepared.results,
        };
        transition_with_children(checkpoint, prepared.child_calls)
    }

    /// Persists and commits one returned tool child result.
    fn child_result(
        &self,
        payload: &AgentPayloadView<'_>,
        mut checkpoint: EdgeAgentCheckpoint,
        call_id: String,
        output: Value,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let active_call = take_active_call(payload, &mut checkpoint, &call_id)?;
        let rendered = self.render_tool_result(payload, &active_call, output)?;
        let EdgeRenderedToolResult {
            message,
            value,
            error,
        } = rendered;
        let message = message.with_call_id(call_id);
        let session = checkpoint.session_id.clone();
        let transition = self.commit_tool_result(checkpoint, active_call, value, error)?;
        record(ctx, &session, payload.agent_id, vec![message], transition)
    }

    /// Converts one child output into history and controller-visible values.
    fn render_tool_result(
        &self,
        payload: &AgentPayloadView<'_>,
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
        mut checkpoint: EdgeAgentCheckpoint,
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
        let next = complete_active_call(&mut checkpoint, active_call, result)?;
        if let Some((active, child)) = next {
            add_active_call(&mut checkpoint, active)?;
            return transition_with_children(checkpoint, vec![child]);
        }
        self.finish_tool_round(checkpoint)
    }

    /// Enters `AfterTools` after every accepted call has completed.
    fn finish_tool_round(
        &self,
        mut checkpoint: EdgeAgentCheckpoint,
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
            return persist_checkpoint(checkpoint);
        }
        let results = finish_results(results);
        checkpoint.metrics.record_results(&results)?;
        checkpoint.phase = if self.controller.is_some() {
            EdgeAgentPhase::AfterTools { results }
        } else {
            normal_dispatch_phase(&checkpoint)
        };
        persist_checkpoint(checkpoint)
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
    mut checkpoint: EdgeAgentCheckpoint,
) -> Result<ContinuationTransition, GraphError> {
    let EdgeAgentPhase::BeforeTools {
        thought,
        calls,
        usage,
    } = checkpoint.phase
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
    persist_checkpoint(checkpoint)
}

/// Performs full JSON Schema validation before final history mutation.
fn validate_agent_output(
    payload: &AgentPayloadView<'_>,
    output: &JsonValue,
    concluding: bool,
    ctx: &ContinuationContext<'_>,
) -> Result<(), GraphError> {
    let validator = ctx
        .output_validator()
        .ok_or_else(|| GraphError::Invalid("prepared agent output validator is missing".into()))?;
    let result = validator
        .validate(output)
        .map_err(|error| GraphError::Schema {
            label: "agent structured output".into(),
            expected: payload.output_type_name.to_owned(),
            value: error.to_string(),
        });
    if concluding {
        result.map_err(|error| GraphError::AgentConclusion {
            agent: payload.agent_id.to_owned(),
            reason: format!("invalid structured output: {error}"),
        })
    } else {
        result
    }
}

fn tool_infos(payload: &AgentPayloadView<'_>, selected: &[String]) -> Vec<ToolInfo> {
    payload
        .tools
        .iter()
        .filter(|tool| selected.contains(&tool.name))
        .map(ToolInfo::from_payload)
        .collect()
}

fn append_guidance(messages: &mut Vec<Message>, guidance: Option<&str>) {
    let Some(guidance) = guidance else {
        return;
    };
    messages.push(Message::new(
        Role::System,
        format!("<pravah_agent_intervention>\n{guidance}\n</pravah_agent_intervention>"),
    ));
}
