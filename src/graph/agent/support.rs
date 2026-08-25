use super::*;

pub(super) fn maybe_inject_edge_turn_budget_message(
    client: &dyn Client,
    agent_name: &str,
    session_msgs: &mut Vec<Message>,
    turn_offset: usize,
    completed: usize,
) {
    let options = client.options();
    if options.tools.is_empty() && !client.uses_exit_tool() {
        return;
    }
    let Some(budget) = options.turn_budget else {
        return;
    };
    let turns_this_invocation = completed.saturating_sub(turn_offset);
    if turns_this_invocation.saturating_add(1) < budget as usize {
        return;
    }
    let exit_tool_name = (client.uses_exit_tool() && !options.output_type_name.is_empty())
        .then_some(options.output_type_name.as_str());
    let text = options
        .turn_budget_message
        .as_deref()
        .map(|msg| client.wrap_system_reminder(msg))
        .unwrap_or_else(|| client.default_turn_budget_message(exit_tool_name));
    tracing::warn!(
        agent = %agent_name,
        completed_turns = completed,
        turns_this_invocation,
        budget = budget,
        "turn budget reached; injecting last-turn reminder"
    );
    session_msgs.push(Message::user(text));
}

/// Validates and freezes one activation-time agent configuration.
pub(super) async fn resolve_agent_config(
    payload: &AgentPayload,
    config: AgentConfig,
    ctx: &Context,
) -> Result<(ResolvedAgentConfig, Message), GraphError> {
    validate_agent_config(&config)?;
    let mut refs = BTreeSet::new();
    for resource in &config.resources {
        if !refs.insert(resource.clone()) {
            return Err(GraphError::AgentConfigValidation(format!(
                "duplicate MCP resource '{}:{}'",
                resource.server(),
                resource.uri()
            )));
        }
    }
    let tools = payload
        .tools
        .iter()
        .filter(|tool| config.tool_filter.allows(tool))
        .map(|tool| tool.name.clone())
        .collect();
    let resources = resolve_resources(ctx, &config.resources).await?;
    let resolved = ResolvedAgentConfig {
        model: config.model,
        instructions: config.instructions,
        memory: config.memory,
        provider_config: config.provider_config,
        keep_alive: config.keep_alive,
        turn_budget: config.turn_budget,
        turn_budget_message: config.turn_budget_message,
        tools,
        resources,
    };
    Ok((resolved, config.message))
}

fn validate_agent_config(config: &AgentConfig) -> Result<(), GraphError> {
    if config.model.trim().is_empty() {
        return Err(GraphError::AgentConfigValidation(
            "model must not be empty".into(),
        ));
    }
    if !matches!(config.message.role, Role::User) {
        return Err(GraphError::AgentConfigValidation(
            "initial agent message must have the user role".into(),
        ));
    }
    if config.turn_budget == Some(0) {
        return Err(GraphError::AgentConfigValidation(
            "turn budget must be greater than zero".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "mcp")]
async fn resolve_resources(
    ctx: &Context,
    resources: &[McpResourceRef],
) -> Result<Vec<ResolvedResource>, GraphError> {
    crate::graph::mcp::resolve_resources(ctx, resources).await
}

#[cfg(not(feature = "mcp"))]
async fn resolve_resources(
    _ctx: &Context,
    resources: &[McpResourceRef],
) -> Result<Vec<ResolvedResource>, GraphError> {
    if resources.is_empty() {
        Ok(Vec::new())
    } else {
        Err(GraphError::McpResource(
            "MCP resources require the 'mcp' crate feature".into(),
        ))
    }
}

pub(super) trait MessageCallIdExt {
    fn with_call_id(self, call_id: String) -> Self;
}

impl MessageCallIdExt for Message {
    fn with_call_id(mut self, call_id: String) -> Self {
        self.role = Role::Tool { call_id };
        self
    }
}

pub(super) fn error_tool_spec(err: String) -> AgentToolSpec {
    let payload = AgentToolPayload {
        name: "__invalid_tool__".into(),
        child_index: 0,
        description: err,
        parameters: JsonValue::Object(Default::default()),
    };
    AgentToolSpec {
        payload,
        graph: empty_error_graph(),
        registry: HandlerRegistry::new(),
        runtime: Arc::new(EdgeAgentToolRuntime {
            decode_args: Arc::new(|_| Err(ToolError::Fatal("invalid tool".into()))),
            to_message: Arc::new(|_| Err(ToolError::Fatal("invalid tool".into()).into())),
        }),
    }
}

pub(super) fn build_tool_handler_flow<I, O, Fut, H>(
    func: H,
) -> Result<CompiledFlow<I, JsonValue>, String>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
    H: Fn(I, Context) -> Fut + Send + Sync + 'static,
{
    let builder = crate::graph::TypedGraphBuilder::<I>::new();
    let root = builder.root();
    let output = builder.work(root, move |input, ctx| {
        let fut = func(input, ctx);
        async move {
            match fut.await {
                Ok(output) => serde_json::to_value(output)
                    .map(|value| EdgeToolResult::Success { value })
                    .and_then(serde_json::to_value)
                    .map_err(|err| GraphError::Invalid(format!("tool serialize failed: {err}"))),
                Err(err) if !err.is_fatal() => serde_json::to_value(EdgeToolResult::Error {
                    value: err.to_json(""),
                })
                .map_err(|encode| {
                    GraphError::Invalid(format!("tool error serialize failed: {encode}"))
                }),
                Err(err) => Err(GraphError::Invalid(err.to_string())),
            }
        }
    });
    builder.finish(output).map_err(|err| err.to_string())
}

pub(super) fn decode_tool_result(value: Value) -> Result<EdgeToolResult, EdgeToolMessageError> {
    from_value(value.clone()).map_err(|err| EdgeToolMessageError::Fatal {
        expected: "Pravah tool result envelope".into(),
        reason: err.to_string(),
        raw: preview_value(&value),
    })
}

pub(super) fn decode_handler_tool_message<O: ToolOutput>(
    value: Value,
) -> Result<Message, EdgeToolMessageError> {
    match decode_tool_result(value)? {
        EdgeToolResult::Success { value } => decode_json_tool_message::<O>(value),
        EdgeToolResult::Error { value } => {
            Ok(Message::tool_output(String::new(), value.to_string()))
        }
    }
}

pub(super) fn decode_json_tool_message<O: ToolOutput>(
    value: JsonValue,
) -> Result<Message, EdgeToolMessageError> {
    match serde_json::from_value::<O>(value.clone()) {
        Ok(output) => output.to_message().map_err(Into::into),
        Err(first) => {
            if let JsonValue::String(text) = &value {
                match serde_json::from_str::<O>(text) {
                    Ok(output) => return output.to_message().map_err(Into::into),
                    Err(second) => {
                        return Err(EdgeToolMessageError::Fatal {
                            expected: O::schema_name(),
                            reason: second.to_string(),
                            raw: preview_display(&value),
                        });
                    }
                }
            }
            Err(EdgeToolMessageError::Fatal {
                expected: O::schema_name(),
                reason: first.to_string(),
                raw: preview_display(&value),
            })
        }
    }
}

pub(super) fn decode_runtime_tool_message<O: ToolOutput>(
    value: Value,
) -> Result<Message, EdgeToolMessageError> {
    match from_value::<O>(value.clone()) {
        Ok(output) => output.to_message().map_err(Into::into),
        Err(first) => {
            if let Some(text) = value.as_str()
                && let Ok(output) = serde_json::from_str::<O>(text)
            {
                return output.to_message().map_err(Into::into);
            }
            Err(EdgeToolMessageError::Fatal {
                expected: O::schema_name(),
                reason: first.to_string(),
                raw: preview_value(&value),
            })
        }
    }
}

pub(super) fn decode_payload(payload: &Value) -> Result<AgentPayload, GraphError> {
    let payload: AgentPayload = from_value(payload.clone())
        .map_err(|err| GraphError::Invalid(format!("failed to decode agent payload: {err}")))?;
    if payload.version != PAYLOAD_VERSION {
        return Err(GraphError::UnsupportedVersion {
            format: "agent payload",
            got: payload.version,
            expected: PAYLOAD_VERSION,
        });
    }
    if payload.configure_handler_key.is_empty() || payload.configure_handler_key != payload.agent_id
    {
        return Err(GraphError::AgentConfigValidation(
            "agent configure handler identity is missing or inconsistent".into(),
        ));
    }
    Ok(payload)
}

pub(crate) fn dispatch_injection_target(
    payload: &Value,
    checkpoint: &Value,
) -> Result<Option<(String, String)>, GraphError> {
    let payload: AgentPayload = match from_value(payload.clone()) {
        Ok(payload) => payload,
        Err(_) => return Ok(None),
    };
    if payload.version != PAYLOAD_VERSION {
        return Ok(None);
    }
    let checkpoint: EdgeAgentCheckpoint = from_value(checkpoint.clone())
        .map_err(|err| GraphError::Invalid(format!("failed to decode agent checkpoint: {err}")))?;
    validate_checkpoint(&payload, &checkpoint)?;
    if !matches!(checkpoint.phase, EdgeAgentPhase::Dispatch) {
        return Ok(None);
    }
    Ok(Some((checkpoint.session_id, payload.agent_id)))
}

pub(super) fn single_input(mut inputs: Vec<Value>, label: &str) -> Result<Value, GraphError> {
    if inputs.len() != 1 {
        return Err(GraphError::Invalid(format!(
            "{label} expected one input, got {}",
            inputs.len()
        )));
    }
    inputs
        .pop()
        .ok_or_else(|| GraphError::Invalid(format!("{label} input disappeared")))
}

pub(super) fn persist_checkpoint(
    checkpoint: EdgeAgentCheckpoint,
) -> Result<ContinuationTransition, GraphError> {
    Ok(ContinuationTransition {
        checkpoint: Some(to_value(checkpoint).map_err(|err| {
            GraphError::Invalid(format!("failed to encode agent  checkpoint: {err}"))
        })?),
        state: None,
        outputs: Vec::new(),
        writes: Vec::new(),
        child_calls: Vec::new(),
    })
}

pub(super) fn restore_agent_state(state: Option<Value>) -> Result<Option<String>, GraphError> {
    let Some(state) = state else {
        return Ok(None);
    };
    let state: EdgeAgentSavedState = from_value(state)
        .map_err(|err| GraphError::Invalid(format!("failed to decode agent saved state: {err}")))?;
    if state.version != CHECKPOINT_VERSION {
        return Err(GraphError::UnsupportedVersion {
            format: "agent saved state",
            got: state.version,
            expected: CHECKPOINT_VERSION,
        });
    }
    if state.session_id.is_empty() {
        return Err(GraphError::SnapshotValidation(
            "agent saved state session id is empty".into(),
        ));
    }
    Ok(Some(state.session_id))
}

pub(crate) fn validate_agent_snapshot_state(
    payload: &Value,
    checkpoint: Option<&Value>,
    state: Option<&Value>,
) -> Result<bool, GraphError> {
    let is_agent = payload.get("agent_id").is_some() && payload.get("output_schema").is_some();
    if !is_agent {
        return Ok(false);
    }
    let payload = decode_payload(payload)?;
    if let Some(checkpoint) = checkpoint {
        let checkpoint: EdgeAgentCheckpoint = from_value(checkpoint.clone()).map_err(|err| {
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
    }
    if let Some(state) = state {
        restore_agent_state(Some(state.clone()))?;
    }
    Ok(true)
}

pub(super) fn validate_checkpoint(
    payload: &AgentPayload,
    checkpoint: &EdgeAgentCheckpoint,
) -> Result<(), GraphError> {
    if checkpoint.session_id.is_empty() {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint session id is empty".into(),
        ));
    }
    validate_resolved_tools(payload, &checkpoint.resolved.tools)?;
    validate_resolved_resources(&checkpoint.resolved.resources)?;
    if checkpoint.resolved.model.trim().is_empty() {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint model is empty".into(),
        ));
    }
    if checkpoint.resolved.turn_budget == Some(0) {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint turn budget is zero".into(),
        ));
    }
    validate_pending_calls(payload, checkpoint)
}

fn validate_resolved_tools(payload: &AgentPayload, selected: &[String]) -> Result<(), GraphError> {
    let expected = payload
        .tools
        .iter()
        .filter(|tool| selected.contains(&tool.name))
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    let actual = selected.iter().map(String::as_str).collect::<Vec<_>>();
    if expected != actual {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint tools are unknown, duplicated, or unordered".into(),
        ));
    }
    Ok(())
}

fn validate_resolved_resources(resources: &[ResolvedResource]) -> Result<(), GraphError> {
    let mut seen = BTreeSet::new();
    for resource in resources {
        if resource.server.is_empty() || resource.uri.is_empty() {
            return Err(GraphError::SnapshotValidation(
                "agent checkpoint resource identity is empty".into(),
            ));
        }
        if !seen.insert((resource.server.as_str(), resource.uri.as_str())) {
            return Err(GraphError::SnapshotValidation(
                "agent checkpoint contains duplicate resources".into(),
            ));
        }
    }
    Ok(())
}

fn validate_pending_calls(
    payload: &AgentPayload,
    checkpoint: &EdgeAgentCheckpoint,
) -> Result<(), GraphError> {
    let EdgeAgentPhase::PendingTool { active, waiting } = &checkpoint.phase else {
        return Ok(());
    };
    let mut ids = BTreeSet::new();
    for call in active
        .iter()
        .map(|call| (&call.call_id, &call.tool_name, call.child_index))
        .chain(
            waiting
                .iter()
                .map(|call| (&call.call_id, &call.tool_name, call.child_index)),
        )
    {
        let tool = payload.tools.get(call.2).ok_or_else(|| {
            GraphError::SnapshotValidation("agent tool call child index is invalid".into())
        })?;
        if tool.name.as_str() != call.1.as_str() || !checkpoint.resolved.tools.contains(&tool.name)
        {
            return Err(GraphError::SnapshotValidation(
                "agent tool call does not match a selected tool".into(),
            ));
        }
        if !ids.insert(call.0) {
            return Err(GraphError::SnapshotValidation(
                "agent checkpoint contains duplicate tool call ids".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn completed_agent_state(
    checkpoint: &EdgeAgentCheckpoint,
) -> Result<Option<Value>, GraphError> {
    if !checkpoint.resolved.keep_alive {
        return Ok(None);
    }
    to_value(EdgeAgentSavedState {
        version: CHECKPOINT_VERSION,
        session_id: checkpoint.session_id.clone(),
    })
    .map(Some)
    .map_err(|err| GraphError::Invalid(format!("failed to encode agent saved state: {err}")))
}

pub(super) fn transition_with_children(
    checkpoint: EdgeAgentCheckpoint,
    child_calls: Vec<ContinuationChildCall>,
) -> Result<ContinuationTransition, GraphError> {
    Ok(ContinuationTransition {
        checkpoint: Some(to_value(checkpoint).map_err(|err| {
            GraphError::Invalid(format!("failed to encode agent  checkpoint: {err}"))
        })?),
        state: None,
        outputs: Vec::new(),
        writes: Vec::new(),
        child_calls,
    })
}

pub(super) fn pending_waiting(checkpoint: &EdgeAgentCheckpoint) -> Vec<EdgeWaitingToolCall> {
    match &checkpoint.phase {
        EdgeAgentPhase::PendingTool { waiting, .. } => waiting.clone(),
        EdgeAgentPhase::Dispatch => Vec::new(),
    }
}

pub(super) fn effective_preamble(payload: &AgentPayload, resolved: &ResolvedAgentConfig) -> String {
    let hint = format!(
        "The user message is JSON. Interpret it using this JSON Schema: {}",
        payload.input_schema
    );
    let mut sections = Vec::new();
    if !resolved.instructions.is_empty() {
        sections.push(resolved.instructions.clone());
    }
    if let Some(memory) = &resolved.memory {
        sections.push(format!("<memory>\n{memory}\n</memory>"));
    }
    for resource in &resolved.resources {
        sections.push(format!(
            "<resource server=\"{}\" uri=\"{}\">\n{}\n</resource>",
            resource.server, resource.uri, resource.text
        ));
    }
    sections.push(hint);
    sections.join("\n\n")
}

pub(super) fn schema_for<T: JsonSchema>() -> JsonValue {
    serde_json::to_value(schemars::r#gen::SchemaGenerator::default().root_schema_for::<T>())
        .unwrap_or_else(|_| serde_json::json!({"type": "object"}))
}

pub(super) fn preview_value(value: &Value) -> String {
    preview_display(value)
}

fn preview_display(value: &impl std::fmt::Display) -> String {
    let raw = value.to_string();
    let mut chars = raw.chars();
    let preview = chars.by_ref().take(512).collect::<String>();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

pub(super) fn empty_error_graph() -> UntypedGraph {
    let flow = crate::graph::TypedGraphBuilder::<JsonValue>::new();
    let root = flow.root();
    flow.finish(root)
        .map(|flow| flow.into_parts().0)
        .unwrap_or_else(|_| UntypedGraph {
            schema_version: crate::graph::UNTYPED_GRAPH_SCHEMA_VERSION,
            name: "invalid_tool".into(),
            edges: Vec::new(),
            variables: Vec::new(),
            marks: Vec::new(),
            nodes: Vec::new(),
            entry: crate::graph::EdgeId(0),
            exit: crate::graph::EdgeId(0),
        })
}
