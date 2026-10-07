//! Native JSON tool registration and immutable canonical validation contracts.

use super::*;
use crate::graph::{HandlerKey, NodeKind, TypeSpec, UntypedGraphBuilder};

mod restore;
mod schema;
pub(super) use restore::validate_input as validate_restored_json_input;
pub(crate) use restore::{validate_json_outcome, validate_json_snapshot};

/// Binds authored JSON declarations to the validators shared by codecs and workers.
pub(super) struct JsonToolContract {
    pub(super) definition: Value,
    input_validator: jsonschema::Validator,
    output_validator: jsonschema::Validator,
}

impl Toolset {
    /// Registers a capturing asynchronous JSON handler with a canonical input schema.
    ///
    /// Schemas follow the documented Draft 2020-12 profile and are never normalized.
    /// `None` explicitly permits any successful output; error envelopes bypass that schema.
    /// Invalid definitions accumulate and fail the containing compile/build/restore operation.
    pub fn json<H, Fut>(
        mut self,
        definition: ToolDefinition,
        output_schema: Option<JsonValue>,
        handler: H,
    ) -> Self
    where
        H: Fn(JsonValue, Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<JsonValue, ToolError>> + Send + 'static,
    {
        let spec = build_spec(definition, output_schema, handler, self.tools.len());
        self.tools.push(spec.unwrap_or_else(error_tool_spec));
        self
    }
}

impl JsonToolContract {
    /// Rebuilds validation from an authored contract for operation-local restore checks.
    fn from_definition(definition: &Value) -> Result<Self, GraphError> {
        let invalid = || GraphError::GraphValidation("invalid JSON tool contract".into());
        if definition
            .object_entries()
            .is_none_or(|fields| fields.len() != 4)
        {
            return Err(invalid());
        }
        let name = definition
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(invalid)?;
        let description = definition
            .get("description")
            .and_then(Value::as_str)
            .ok_or_else(invalid)?;
        let input = definition.get("parameters").ok_or_else(invalid)?;
        let output = definition.get("output_schema").ok_or_else(invalid)?;
        let input = serde_json::to_value(input).map_err(|_| invalid())?;
        let output = serde_json::to_value(output).map_err(|_| invalid())?;
        Self::new(
            ToolDefinition::new(name.into(), description.into(), input),
            Some(output),
        )
        .map_err(GraphError::GraphValidation)
    }

    /// Compiles one immutable contract before any graph insertion or handler execution.
    fn new(definition: ToolDefinition, output: Option<JsonValue>) -> Result<Self, String> {
        if definition.name.trim().is_empty() || definition.name == "__rath_final_output" {
            return Err("JSON tool name must be nonempty and not reserved".into());
        }
        let output = output.unwrap_or(JsonValue::Bool(true));
        let input_validator = schema::compile(&definition.parameters, &definition.name, "input")?;
        if definition
            .parameters
            .get("type")
            .and_then(JsonValue::as_str)
            != Some("object")
        {
            return Err(format!(
                "tool '{}' input requires root type object",
                definition.name
            ));
        }
        let output_validator = schema::compile(&output, &definition.name, "output")?;
        let definition = Value::object([
            ("name", definition.name.into()),
            ("description", definition.description.into()),
            (
                "parameters",
                to_value(definition.parameters).map_err(|e| e.to_string())?,
            ),
            (
                "output_schema",
                to_value(output).map_err(|e| e.to_string())?,
            ),
        ])
        .map_err(|e| e.to_string())?;
        Ok(Self {
            definition,
            input_validator,
            output_validator,
        })
    }

    pub(super) fn name(&self) -> &str {
        self.definition
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
    }

    /// Reports only locations, never the instance value or an unbounded preview.
    pub(super) fn validate(&self, value: &JsonValue, output: bool) -> Result<(), String> {
        let (validator, label) = if output {
            (&self.output_validator, "output")
        } else {
            (&self.input_validator, "input")
        };
        validator.validate(value).map_err(|error| {
            format!(
                "tool '{}' {label} violates schema {} at instance {}",
                self.name(),
                error.schema_path(),
                error.instance_path(),
            )
        })
    }

    pub(super) fn validate_input(&self, value: &Value) -> Result<(), GraphError> {
        let json = serde_json::to_value(value).map_err(|e| GraphError::Invalid(e.to_string()))?;
        self.validate(&json, false)
            .map_err(GraphError::AgentRequestValidation)
    }

    /// Rejects a delivered successful result unless it satisfies the output contract.
    pub(super) fn validate_envelope(&self, value: &Value) -> Result<(), GraphError> {
        if let EdgeToolResult::Success { value } = decode_tool_result(value.clone())
            .map_err(|_| GraphError::AgentRequestValidation("invalid JSON tool envelope".into()))?
        {
            self.validate(&value, true).map_err(GraphError::Invalid)?;
        }
        Ok(())
    }
}

/// Builds the graph, worker and codecs around a single shared validation contract.
fn build_spec<H, Fut>(
    definition: ToolDefinition,
    output: Option<JsonValue>,
    handler: H,
    index: usize,
) -> Result<AgentToolSpec, String>
where
    H: Fn(JsonValue, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<JsonValue, ToolError>> + Send + 'static,
{
    let payload = AgentToolPayload {
        name: definition.name.clone(),
        description: definition.description.clone(),
        parameters: definition.parameters.clone(),
        child_index: index,
    };
    let contract = Arc::new(JsonToolContract::new(definition, output)?);
    let function = FunctionTool::json(Arc::clone(&contract), handler);
    let (graph, registry) = child_graph(&contract, function).map_err(|e| e.to_string())?;
    let decoder = Arc::clone(&contract);
    let renderer = Arc::clone(&contract);
    Ok(AgentToolSpec {
        payload,
        graph,
        registry,
        runtime: Arc::new(EdgeAgentToolRuntime {
            json_contract: Some(contract),
            decode_args: Arc::new(move |json| {
                decoder
                    .validate(&json, false)
                    .map_err(ToolError::Validation)?;
                to_value(json).map_err(|e| ToolError::Fatal(e.to_string()))
            }),
            render_result: Arc::new(move |value| render_json(&renderer, value)),
        }),
    })
}

/// Uses the existing lock-free untyped builder for a function-tool continuation.
fn child_graph(
    contract: &JsonToolContract,
    function: FunctionTool,
) -> Result<(UntypedGraph, HandlerRegistry), GraphError> {
    let mut builder = UntypedGraphBuilder::new("json_tool");
    let input = builder.edge(
        "tool_in",
        TypeSpec::new("serde_json::Value", JsonValue::Bool(true)),
    );
    let output = builder.edge(
        "tool_out",
        TypeSpec::new("serde_json::Value", JsonValue::Bool(true)),
    );
    let key = HandlerKey::new("json_tool::function_tool");
    let payload = Value::object([
        ("tool_handler_key", Value::from(key.as_str())),
        ("json_contract", contract.definition.clone()),
    ])
    .map_err(|e| GraphError::Invalid(e.to_string()))?;
    builder.node(
        "tool",
        NodeKind::Continuation {
            key: key.clone(),
            payload,
            children: Vec::new(),
        },
        vec![input],
        vec![output],
    );
    builder.set_entry(input).set_exit(output);
    let mut registry = HandlerRegistry::new();
    registry.insert_effect_continuation(key.as_str(), function)?;
    Ok((builder.build()?, registry))
}

/// Renders exact JSON without the typed API's string reparsing fallback.
fn render_json(
    contract: &JsonToolContract,
    envelope: Value,
) -> Result<EdgeRenderedToolResult, EdgeToolMessageError> {
    let (json, error) = match decode_tool_result(envelope)? {
        EdgeToolResult::Success { value } => {
            contract
                .validate(&value, true)
                .map_err(|reason| EdgeToolMessageError::Fatal {
                    expected: "canonical JSON output schema".into(),
                    reason,
                    raw: "<redacted>".into(),
                })?;
            (value, false)
        }
        EdgeToolResult::Error { value } => (value, true),
    };
    let message = Message::tool_output(String::new(), json.to_string());
    let value = to_value(json).map_err(|e| EdgeToolMessageError::Fatal {
        expected: "JSON output value".into(),
        reason: e.to_string(),
        raw: "<redacted>".into(),
    })?;
    Ok(EdgeRenderedToolResult {
        message,
        value,
        error,
    })
}

/// Binds parent model metadata and format version to registered JSON codecs.
pub(super) fn validate_runtime_contracts(
    tools: &[AgentToolPayload],
    runtimes: &[Arc<EdgeAgentToolRuntime>],
    payload: &Value,
) -> Result<(), GraphError> {
    let has_json = runtimes
        .iter()
        .any(|runtime| runtime.json_contract.is_some());
    let json_version =
        payload.get("version").and_then(Value::as_u64) == Some(u64::from(JSON_PAYLOAD_VERSION));
    if !has_json && !json_version {
        return Ok(());
    }
    if tools.len() != runtimes.len() || has_json != json_version {
        return Err(GraphError::GraphValidation(
            "JSON tool payload binding/version mismatch".into(),
        ));
    }
    for (index, tool) in tools.iter().enumerate() {
        if tool.child_index != index {
            return Err(GraphError::GraphValidation(
                "invalid JSON tool child index".into(),
            ));
        }
        if let Some(contract) = runtimes
            .get(tool.child_index)
            .and_then(|r| r.json_contract.as_ref())
        {
            validate_projection(tool, &contract.definition)?;
        }
    }
    Ok(())
}

/// Checks the existing provider projection against its authoritative child declaration.
fn validate_projection(tool: &AgentToolPayload, definition: &Value) -> Result<(), GraphError> {
    let parameters = to_value(&tool.parameters).map_err(|e| GraphError::Invalid(e.to_string()))?;
    if definition.get("name").and_then(Value::as_str) != Some(tool.name.as_str())
        || definition.get("description").and_then(Value::as_str) != Some(tool.description.as_str())
        || definition.get("parameters") != Some(&parameters)
    {
        return Err(GraphError::GraphValidation(
            "JSON tool definition differs from its projection".into(),
        ));
    }
    Ok(())
}

/// Validates child binding and format gates even for graphs loaded without typed authoring.
pub(crate) fn validate_json_children(
    payload: &Value,
    children: &[UntypedGraph],
) -> Result<(), GraphError> {
    if payload.get("agent_id").is_none() {
        return Ok(());
    }
    let has_json = children.iter().any(|child| child_contract(child).is_some());
    let json_version =
        payload.get("version").and_then(Value::as_u64) == Some(u64::from(JSON_PAYLOAD_VERSION));
    if !has_json && !json_version {
        return Ok(());
    }
    let agent = decode_payload(payload)?;
    if has_json != json_version || agent.tools.len() != children.len() {
        return Err(GraphError::GraphValidation(
            "JSON tool child/version mismatch".into(),
        ));
    }
    for (index, tool) in agent.tools.iter().enumerate() {
        if tool.child_index != index {
            return Err(GraphError::GraphValidation(
                "invalid JSON tool child index".into(),
            ));
        }
        let child = children
            .get(tool.child_index)
            .ok_or_else(|| GraphError::GraphValidation("missing tool child graph".into()))?;
        if let Some(definition) = child_contract(child) {
            validate_projection(tool, definition)?;
        }
    }
    Ok(())
}

fn child_contract(graph: &UntypedGraph) -> Option<&Value> {
    graph.nodes.iter().find_map(|node| match &node.kind {
        NodeKind::Continuation { payload, .. } => payload.get("json_contract"),
        _ => None,
    })
}

#[cfg(test)]
#[path = "tests/json_tool.rs"]
mod tests;
