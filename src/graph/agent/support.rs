use super::*;
use crate::graph::model::TypeSpec;

/// Validates and freezes one activation-time agent configuration.
pub(super) async fn resolve_agent_config(
    payload: &AgentPayload,
    config: AgentConfig,
    ctx: &Context,
) -> Result<(ResolvedAgentConfig, Message, Option<AgentBudgetState>), GraphError> {
    validate_agent_config(payload, &config)?;
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
        .collect::<Vec<_>>();
    let budget = AgentBudgetState::resolve(&config, &tools);
    let resources = resolve_resources(ctx, &config.resources).await?;
    let resolved = ResolvedAgentConfig {
        model: config.model,
        instructions: config.instructions,
        memory: config.memory,
        provider_config: config.provider_config,
        keep_alive: config.keep_alive,
        tools,
        resources,
    };
    Ok((resolved, config.message, budget))
}

/// Checks activation settings that must hold before history can change.
fn validate_agent_config(payload: &AgentPayload, config: &AgentConfig) -> Result<(), GraphError> {
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
    validate_budget_config(payload, config)
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

/// Produces a non-runnable placeholder that preserves a tool build error.
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
            render_result: Arc::new(|_| Err(ToolError::Fatal("invalid tool".into()).into())),
        }),
    }
}

/// Lowers a small asynchronous tool handler into the canonical child graph.
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

/// Decodes one handler-backed tool envelope into history and runtime values.
pub(super) fn decode_handler_tool_result<O: ToolOutput>(
    value: Value,
) -> Result<EdgeRenderedToolResult, EdgeToolMessageError> {
    match decode_tool_result(value)? {
        EdgeToolResult::Success { value } => {
            let message = decode_json_tool_message::<O>(value.clone())?;
            let value = to_value(value).map_err(|err| EdgeToolMessageError::Fatal {
                expected: "tool output value".into(),
                reason: err.to_string(),
                raw: "<tool output>".into(),
            })?;
            Ok(EdgeRenderedToolResult {
                message,
                value,
                error: false,
            })
        }
        EdgeToolResult::Error { value } => {
            let runtime_value =
                to_value(value.clone()).map_err(|err| EdgeToolMessageError::Fatal {
                    expected: "tool error value".into(),
                    reason: err.to_string(),
                    raw: preview_display(&value),
                })?;
            Ok(EdgeRenderedToolResult {
                message: Message::tool_output(String::new(), value.to_string()),
                value: runtime_value,
                error: true,
            })
        }
    }
}

/// Renders a JSON tool value through its declared `ToolOutput` contract.
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

/// Renders a typed runtime tool value without losing recoverable error tags.
pub(super) fn decode_runtime_tool_result<O: ToolOutput>(
    value: Value,
) -> Result<EdgeRenderedToolResult, EdgeToolMessageError> {
    let message = match from_value::<O>(value.clone()) {
        Ok(output) => output.to_message().map_err(EdgeToolMessageError::from)?,
        Err(first) => {
            if let Some(text) = value.as_str()
                && let Ok(output) = serde_json::from_str::<O>(text)
            {
                output.to_message().map_err(EdgeToolMessageError::from)?
            } else {
                return Err(EdgeToolMessageError::Fatal {
                    expected: O::schema_name(),
                    reason: first.to_string(),
                    raw: preview_value(&value),
                });
            }
        }
    };
    Ok(EdgeRenderedToolResult {
        message,
        value,
        error: false,
    })
}

/// Decodes and validates the current serialized agent payload version.
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
    let expected_control = format!("{}::control", payload.agent_id);
    if payload
        .control_handler_key
        .as_deref()
        .is_some_and(|key| key != expected_control)
    {
        return Err(GraphError::AgentConfigValidation(
            "agent control handler identity is inconsistent".into(),
        ));
    }
    Ok(payload)
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
        suspension: None,
    })
}

/// Restores the keep-alive session identity from opaque continuation state.
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

/// Validates agent-specific checkpoint and saved state during graph restore.
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

/// Validates all stable agent checkpoint identities and phase relationships.
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
    validate_selected_tools(payload, checkpoint)?;
    validate_budget_state(&checkpoint.resolved.tools, checkpoint.budget.as_ref())?;
    validate_resolved_resources(&checkpoint.resolved.resources)?;
    if checkpoint.resolved.model.trim().is_empty() {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint model is empty".into(),
        ));
    }
    validate_checkpoint_phase(payload, checkpoint)
}

/// Requires configured tool identities to be unique and in prepared order.
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

/// Requires controller-selected tools to be an ordered configured subset.
fn validate_selected_tools(
    payload: &AgentPayload,
    checkpoint: &EdgeAgentCheckpoint,
) -> Result<(), GraphError> {
    validate_resolved_tools(payload, &checkpoint.selected_tools)?;
    if checkpoint
        .selected_tools
        .iter()
        .any(|tool| !checkpoint.resolved.tools.contains(tool))
    {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint selects a tool outside its configured set".into(),
        ));
    }
    Ok(())
}

/// Rejects empty or duplicated checkpointed MCP resource identities.
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

/// Dispatches validation for the checkpoint's explicit agent-loop phase.
fn validate_checkpoint_phase(
    payload: &AgentPayload,
    checkpoint: &EdgeAgentCheckpoint,
) -> Result<(), GraphError> {
    match &checkpoint.phase {
        EdgeAgentPhase::BeforeTools { calls, .. } | EdgeAgentPhase::AcceptedTools { calls, .. } => {
            validate_staged_calls(calls)
        }
        EdgeAgentPhase::PendingTool {
            active,
            waiting,
            results,
        } => validate_pending_calls(payload, checkpoint, active, waiting, results),
        EdgeAgentPhase::AfterTools { results } => validate_completed_results(results),
        EdgeAgentPhase::BeforeModel | EdgeAgentPhase::Dispatch { .. } => Ok(()),
    }
}

/// Validates staged call identities before acceptance or restoration.
fn validate_staged_calls(calls: &[EdgeProposedToolCall]) -> Result<(), GraphError> {
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    if calls.is_empty()
        || calls.iter().any(|call| {
            call.proposal.call_id().is_empty()
                || call.proposal.tool_name().is_empty()
                || !ids.insert(call.proposal.call_id())
        })
    {
        return Err(GraphError::SnapshotValidation(
            "agent staged tool calls are empty, duplicated, or invalid".into(),
        ));
    }
    Ok(())
}

/// Validates pending child calls and completed results as one unique batch.
fn validate_pending_calls(
    payload: &AgentPayload,
    checkpoint: &EdgeAgentCheckpoint,
    active: &[EdgeActiveToolCall],
    waiting: &[EdgeWaitingToolCall],
    results: &[EdgeCompletedToolCall],
) -> Result<(), GraphError> {
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
        if tool.name.as_str() != call.1.as_str() || !checkpoint.selected_tools.contains(&tool.name)
        {
            return Err(GraphError::SnapshotValidation(
                "agent tool call does not match a selected tool".into(),
            ));
        }
        if !ids.insert(call.0.as_str()) {
            return Err(GraphError::SnapshotValidation(
                "agent checkpoint contains duplicate tool call ids".into(),
            ));
        }
    }
    for result in results {
        if !ids.insert(result.result.call_id()) {
            return Err(GraphError::SnapshotValidation(
                "agent checkpoint contains duplicate completed tool call ids".into(),
            ));
        }
    }
    Ok(())
}

/// Validates the complete ordered result batch observed at `AfterTools`.
fn validate_completed_results(results: &[AgentToolResult]) -> Result<(), GraphError> {
    let mut ids = BTreeSet::new();
    if results.iter().any(|result| {
        result.call_id().is_empty()
            || result.tool_name().is_empty()
            || !ids.insert(result.call_id())
    }) {
        return Err(GraphError::SnapshotValidation(
            "agent completed tool results are duplicated or invalid".into(),
        ));
    }
    Ok(())
}

/// Validates an agent-owned suspension envelope against its saved checkpoint.
pub(crate) fn validate_agent_suspension(
    payload: &Value,
    checkpoint: &Value,
    suspension_payload: &Value,
    resume_type: &TypeSpec,
) -> Result<bool, GraphError> {
    let is_agent = payload.get("agent_id").is_some() && payload.get("output_schema").is_some();
    if !is_agent {
        return Ok(false);
    }
    let payload = decode_payload(payload)?;
    let checkpoint: EdgeAgentCheckpoint = from_value(checkpoint.clone()).map_err(|err| {
        GraphError::SnapshotValidation(format!("failed to decode agent checkpoint: {err}"))
    })?;
    validate_checkpoint(&payload, &checkpoint)?;
    if payload.control_handler_key.is_none()
        || checkpoint_point_for_validation(&checkpoint.phase).is_none()
    {
        return Err(GraphError::SnapshotValidation(
            "agent suspension is not at a controlled intervention boundary".into(),
        ));
    }
    let expected = TypeSpec::new(AgentResume::schema_name(), schema_for::<AgentResume>());
    if resume_type != &expected {
        return Err(GraphError::SnapshotValidation(
            "agent suspension resume schema is inconsistent".into(),
        ));
    }
    let suspension: AgentSuspension = from_value(suspension_payload.clone()).map_err(|err| {
        GraphError::SnapshotValidation(format!("failed to decode agent suspension: {err}"))
    })?;
    let point = checkpoint_point_for_validation(&checkpoint.phase).ok_or_else(|| {
        GraphError::SnapshotValidation("agent suspension phase is not resumable".into())
    })?;
    if suspension.agent_id() != payload.agent_id
        || suspension.session_id() != checkpoint.session_id
        || suspension.point() != point
    {
        return Err(GraphError::SnapshotValidation(
            "agent suspension identity does not match its checkpoint".into(),
        ));
    }
    Ok(true)
}

fn checkpoint_point_for_validation(phase: &EdgeAgentPhase) -> Option<AgentInterventionPoint> {
    match phase {
        EdgeAgentPhase::BeforeModel => Some(AgentInterventionPoint::BeforeModel),
        EdgeAgentPhase::BeforeTools { .. } => Some(AgentInterventionPoint::BeforeTools),
        EdgeAgentPhase::AfterTools { .. } => Some(AgentInterventionPoint::AfterTools),
        EdgeAgentPhase::Dispatch { .. }
        | EdgeAgentPhase::AcceptedTools { .. }
        | EdgeAgentPhase::PendingTool { .. } => None,
    }
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
        suspension: None,
    })
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
