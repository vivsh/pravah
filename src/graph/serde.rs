use super::{GraphError, HandlerRegistry, UntypedGraph};

/// Serializes an untyped graph in deterministic pretty JSON form.
pub fn to_json_pretty(graph: &UntypedGraph) -> Result<String, GraphError> {
    serde_json::to_string_pretty(graph).map_err(|err| GraphError::JsonEncode {
        target: "graph".into(),
        reason: err.to_string(),
    })
}

/// Loads and shape-validates an untyped graph from JSON.
pub fn from_json(input: &str) -> Result<UntypedGraph, GraphError> {
    let graph: UntypedGraph =
        serde_json::from_str(input).map_err(|err| GraphError::JsonDecode {
            target: "graph".into(),
            reason: err.to_string(),
        })?;
    super::validation::validate_graph_shape(&graph)?;
    Ok(graph)
}

/// Loads a graph from JSON and verifies all referenced handlers exist.
pub fn from_json_with_registry(
    input: &str,
    registry: &HandlerRegistry,
) -> Result<UntypedGraph, GraphError> {
    let graph = from_json(input)?;
    let has_value = |key: &str| registry.has_value(key);
    let has_work = |key: &str| registry.has_work(key);
    let has_continuation = |key: &str| registry.has_continuation(key);
    super::validation::validate_registry_keys(&graph, &has_value, &has_work, &has_continuation)?;
    Ok(graph)
}
