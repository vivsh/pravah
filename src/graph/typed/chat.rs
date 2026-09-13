use super::*;
use crate::graph::NodeId;

/// Builds one prepared chat graph and returns its authored application access points.
pub(crate) fn build_chat_graph<I, O, S>(
    agent: fn(Agent<I>) -> Agent<O>,
) -> Result<(PreparedGraph, VarId, [NodeId; 2]), GraphError>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: JsonSchema,
{
    let root = Flow::<()>::root();
    let state_var = {
        let mut state = root
            .state
            .lock()
            .map_err(|_| GraphError::Invalid("typed chat builder lock is poisoned".into()))?;
        state.builder.variable(
            VarKey::new("pravah.chat", "state"),
            type_spec::<S>(),
            VarScope::Local,
            VarInit::Uninitialized,
        )
    };
    let request = root.clone().suspend::<I>();
    let bootstrap_edge = request.edge;
    let start = request.mark();
    let next_request = request.agent(agent).suspend::<I>();
    let response_edge = next_request.edge;
    let _loop_edge = next_request.goto(start);
    let compiled = root.map(|value| value).finish::<()>()?;
    let graph = compiled.prepared.graph();
    let bootstrap = suspension_producer(graph, bootstrap_edge)?;
    let response = suspension_producer(graph, response_edge)?;
    Ok((compiled.prepared, state_var, [bootstrap, response]))
}

/// Resolves the producing suspend node from the edge recorded during typed construction.
fn suspension_producer(graph: &UntypedGraph, edge: EdgeId) -> Result<NodeId, GraphError> {
    let node = graph
        .edge(edge)
        .and_then(|edge| edge.producer)
        .and_then(|id| graph.node(id))
        .ok_or_else(|| GraphError::GraphValidation("chat suspension producer is missing".into()))?;
    if !matches!(node.kind, NodeKind::Suspend { .. }) {
        return Err(GraphError::GraphValidation(
            "chat boundary is not a suspend node".into(),
        ));
    }
    Ok(node.id)
}
