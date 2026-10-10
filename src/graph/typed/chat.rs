use super::*;
use crate::graph::NodeId;
use crate::graph::chat::ChatRequest;

/// Builds one prepared chat graph and returns its authored application access points.
pub(crate) fn build_chat_graph<I, O>(
    agent: Agent<O>,
) -> Result<(PreparedGraph, [NodeId; 2]), GraphError>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let root = Flow::<()>::root();
    let request = root.clone().suspend::<ChatRequest<I>>();
    let bootstrap_edge = request.edge;
    let start = request.mark();
    let response: Flow<O> = Flow::from_typed(add_agent_node::<ChatRequest<I>, O>(
        Arc::clone(&request.state),
        request.edge,
        agent.for_chat(),
    ));
    let next_request = response.suspend::<ChatRequest<I>>();
    let response_edge = next_request.edge;
    let _loop_edge = next_request.goto(start);
    let compiled = root.map(|value| value).finish::<()>()?;
    let graph = compiled.prepared.graph();
    let bootstrap = suspension_producer(graph, bootstrap_edge)?;
    let response = suspension_producer(graph, response_edge)?;
    Ok((compiled.prepared, [bootstrap, response]))
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
