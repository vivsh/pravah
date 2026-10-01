use super::*;
use crate::graph::model::TypeSpec;

/// Validates and freezes one activation-time agent configuration.
pub(super) async fn resolve_agent_config(
    payload: &AgentPayloadView<'_>,
    config: AgentConfig,
    ctx: &Context,
) -> Result<super::effects::Configured, GraphError> {
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
        max_output_tokens: config.max_output_tokens,
        tools,
        resources,
    };
    Ok(super::effects::Configured {
        key: config.key,
        resolved,
        message: config.message,
        budget,
    })
}

/// Checks activation settings that must hold before history can change.
fn validate_agent_config(
    payload: &AgentPayloadView<'_>,
    config: &AgentConfig,
) -> Result<(), GraphError> {
    config
        .tool_filter
        .validate_names(payload.tools.iter().map(|tool| tool.name.as_str()))
        .map_err(GraphError::AgentConfigValidation)?;
    if config.model.trim().is_empty() {
        return Err(GraphError::AgentConfigValidation(
            "model must not be empty".into(),
        ));
    }
    if config.key.as_ref().is_some_and(|key| key.trim().is_empty()) {
        return Err(GraphError::AgentConfigValidation(
            "conversation key must not be empty".into(),
        ));
    }
    if !matches!(config.message.role, Role::User) {
        return Err(GraphError::AgentConfigValidation(
            "initial agent message must have the user role".into(),
        ));
    }
    validate_budget_config(&payload.tools, config)
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
            render_result: Arc::new(|_| {
                Err(EdgeToolMessageError::Fatal {
                    expected: "valid tool".into(),
                    reason: "invalid tool".into(),
                    raw: "<invalid tool>".into(),
                })
            }),
        }),
    }
}

/// Lowers a standalone asynchronous tool function into a child graph.
pub(super) fn build_function_tool_flow<I, O, Fut>(
    func: fn(I, Context) -> Fut,
) -> Result<CompiledFlow<I, JsonValue>, String>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
{
    let builder = crate::graph::TypedGraphBuilder::<I>::new();
    let output = builder.function_tool::<JsonValue>(super::function_tool::FunctionTool::new(func));
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
pub(super) fn decode_handler_tool_result<O>(
    value: Value,
) -> Result<EdgeRenderedToolResult, EdgeToolMessageError>
where
    O: Serialize + DeserializeOwned + JsonSchema,
{
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

/// Validates and renders one JSON-backed typed tool output.
pub(super) fn decode_json_tool_message<O>(value: JsonValue) -> Result<Message, EdgeToolMessageError>
where
    O: Serialize + DeserializeOwned + JsonSchema,
{
    match serde_json::from_value::<O>(value.clone()) {
        Ok(output) => default_tool_message(&output),
        Err(first) => {
            if let JsonValue::String(text) = &value {
                match serde_json::from_str::<O>(text) {
                    Ok(output) => return default_tool_message(&output),
                    Err(second) => {
                        return Err(EdgeToolMessageError::Fatal {
                            expected: O::schema_name().into_owned(),
                            reason: second.to_string(),
                            raw: preview_display(&value),
                        });
                    }
                }
            }
            Err(EdgeToolMessageError::Fatal {
                expected: O::schema_name().into_owned(),
                reason: first.to_string(),
                raw: preview_display(&value),
            })
        }
    }
}

/// Renders a typed runtime tool value without losing recoverable error tags.
pub(super) fn decode_runtime_tool_result<O>(
    value: Value,
) -> Result<EdgeRenderedToolResult, EdgeToolMessageError>
where
    O: Serialize + DeserializeOwned + JsonSchema,
{
    let output = match from_value::<O>(value.clone()) {
        Ok(output) => output,
        Err(first) => {
            if let Some(text) = value.as_str()
                && let Ok(output) = serde_json::from_str::<O>(text)
            {
                output
            } else {
                return Err(EdgeToolMessageError::Fatal {
                    expected: O::schema_name().into_owned(),
                    reason: first.to_string(),
                    raw: preview_value(&value),
                });
            }
        }
    };
    let message = default_tool_message(&output)?;
    Ok(EdgeRenderedToolResult {
        message,
        value,
        error: false,
    })
}

/// Serializes one typed tool output into the default model-facing JSON message.
fn default_tool_message<O: Serialize + JsonSchema>(
    output: &O,
) -> Result<Message, EdgeToolMessageError> {
    let content = serde_json::to_string(output).map_err(|error| EdgeToolMessageError::Fatal {
        expected: O::schema_name().into_owned(),
        reason: error.to_string(),
        raw: "<tool output>".into(),
    })?;
    Ok(Message::tool_output(String::new(), content))
}

/// Decodes and validates the current serialized agent payload version.
pub(super) fn decode_payload(payload: &Value) -> Result<AgentPayload, GraphError> {
    let decoded: AgentPayload = from_value(payload.clone())
        .map_err(|err| GraphError::Invalid(format!("failed to decode agent payload: {err}")))?;
    super::payload::validate_identity(payload)?;
    Ok(decoded)
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
        checkpoint: Some(checkpoint.into_value()?),
        state: None,
        outputs: Vec::new(),
        writes: Vec::new(),
        child_calls: Vec::new(),
        suspension: None,
        ..Default::default()
    })
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
    super::conversation::validate_empty_state(state)?;
    let payload = decode_payload(payload)?;
    if let Some(checkpoint) = checkpoint {
        if checkpoint.get("effect").is_some() {
            super::effects::validate_effect_checkpoint(&payload.tools, checkpoint)?;
            return Ok(true);
        }
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
        validate_checkpoint(&payload.tools, &checkpoint)?;
    }
    Ok(true)
}

/// Validates all stable agent checkpoint identities and phase relationships.
pub(super) fn validate_checkpoint(
    tools: &[AgentToolPayload],
    checkpoint: &EdgeAgentCheckpoint,
) -> Result<(), GraphError> {
    let resolved = checkpoint.resolved_config()?;
    validate_resolved_config(&resolved)?;
    validate_progress(tools, checkpoint, &resolved.tools)
}

/// Validates immutable configuration at activation and restore, before it becomes trusted state.
pub(super) fn validate_resolved_config(resolved: &ResolvedAgentConfig) -> Result<(), GraphError> {
    validate_resolved_resources(&resolved.resources)?;
    if resolved.max_output_tokens == Some(0) {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint max output tokens is zero".into(),
        ));
    }
    if resolved.model.trim().is_empty() {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint model is empty".into(),
        ));
    }
    Ok(())
}

/// Checks mutable progress without reconstructing configuration validated at activation or restore.
pub(super) fn validate_checkpoint_progress(
    tools: &[AgentToolPayload],
    checkpoint: &EdgeAgentCheckpoint,
) -> Result<(), GraphError> {
    validate_progress(tools, checkpoint, &checkpoint.configured_tools()?)
}

/// Keeps tool, budget and phase relationships checked at every internal transition.
fn validate_progress(
    tools: &[AgentToolPayload],
    checkpoint: &EdgeAgentCheckpoint,
    configured_tools: &[String],
) -> Result<(), GraphError> {
    if checkpoint.session_id.is_empty() {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint session id is empty".into(),
        ));
    }
    validate_resolved_tools(tools, configured_tools)?;
    validate_selected_tools(tools, checkpoint, configured_tools)?;
    validate_budget_state(configured_tools, checkpoint.budget.as_ref())?;
    validate_checkpoint_phase(tools, checkpoint)
}

/// Requires configured tool identities to be unique and in prepared order.
fn validate_resolved_tools(
    tools: &[AgentToolPayload],
    selected: &[String],
) -> Result<(), GraphError> {
    let expected = tools
        .iter()
        .filter(|tool| selected.contains(&tool.name))
        .map(|tool| tool.name.as_str());
    if !expected.eq(selected.iter().map(String::as_str)) {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint tools are unknown, duplicated, or unordered".into(),
        ));
    }
    Ok(())
}

/// Requires controller-selected tools to be an ordered configured subset.
fn validate_selected_tools(
    tools: &[AgentToolPayload],
    checkpoint: &EdgeAgentCheckpoint,
    configured_tools: &[String],
) -> Result<(), GraphError> {
    validate_resolved_tools(tools, &checkpoint.selected_tools)?;
    if checkpoint
        .selected_tools
        .iter()
        .any(|tool| !configured_tools.contains(tool))
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
    tools: &[AgentToolPayload],
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
        } => validate_pending_calls(tools, checkpoint, active, waiting, results),
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
    tools: &[AgentToolPayload],
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
        let tool = tools.get(call.2).ok_or_else(|| {
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
    validate_checkpoint(&payload.tools, &checkpoint)?;
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

pub(super) fn transition_with_children(
    checkpoint: EdgeAgentCheckpoint,
    child_calls: Vec<ContinuationChildCall>,
) -> Result<ContinuationTransition, GraphError> {
    Ok(ContinuationTransition {
        checkpoint: Some(checkpoint.into_value()?),
        state: None,
        outputs: Vec::new(),
        writes: Vec::new(),
        child_calls,
        suspension: None,
        ..Default::default()
    })
}

pub(super) fn schema_for<T: JsonSchema>() -> JsonValue {
    serde_json::to_value(schemars::SchemaGenerator::default().root_schema_for::<T>())
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
