//! Diagram support for graph-backed Pravah flows.
//!
//! This inspects the serializable [`UntypedGraph`] only. It never requires a
//! runtime, registry, provider, or handler implementation.

use std::collections::HashSet;

use crate::diagram::{Diagram, DiagramEdge, DiagramEdgeKind, DiagramNode, DiagramNodeKind};

use super::error::GraphError;
use super::ids::{EdgeId, MarkId, NodeId};
use super::model::{BuiltinNode, NodeKind, UntypedGraph};
use super::typed::{CompiledFlow, Flow, compile};

/// Snapshot of a graph-backed flow topology for diagram rendering.
#[derive(Debug, Clone)]
pub struct GraphDiagram {
    inner: Diagram,
}

impl GraphDiagram {
    /// Builds a diagram directly from a serializable graph.
    pub fn from_graph(graph: &UntypedGraph) -> Self {
        Self {
            inner: diagram_from_graph(graph),
        }
    }

    /// Compiles a typed graph flow and builds a diagram from its graph.
    pub fn from_flow<I, O>(flow: fn(Flow<I>) -> Flow<O>) -> Result<Self, GraphError>
    where
        I: 'static
            + schemars::JsonSchema
            + serde::Serialize
            + serde::de::DeserializeOwned
            + Send
            + Sync,
        O: 'static
            + schemars::JsonSchema
            + serde::Serialize
            + serde::de::DeserializeOwned
            + Send
            + Sync,
    {
        let flow = compile(flow)?;
        Ok(Self::from_graph(flow.graph()))
    }

    /// Builds a diagram from an already-compiled typed graph flow.
    pub fn from_compiled_flow<I, O>(flow: &CompiledFlow<I, O>) -> Self
    where
        I: 'static + schemars::JsonSchema + serde::Serialize + serde::de::DeserializeOwned,
        O: 'static + schemars::JsonSchema + serde::Serialize + serde::de::DeserializeOwned,
    {
        Self::from_graph(flow.graph())
    }

    /// Returns the entry node id.
    pub fn entry(&self) -> &str {
        self.inner.entry()
    }

    /// Returns all diagram nodes, including terminals.
    pub fn nodes(&self) -> &[DiagramNode] {
        self.inner.nodes()
    }

    /// Returns all directed edges.
    pub fn edges(&self) -> &[DiagramEdge] {
        self.inner.edges()
    }

    /// Renders the graph as Mermaid `flowchart LR` source.
    pub fn mermaid(&self) -> String {
        self.inner.mermaid()
    }

    /// Renders the graph as Graphviz DOT source.
    pub fn dot(&self) -> String {
        self.inner.dot()
    }

    /// Renders the graph as an indented execution tree.
    pub fn render_tree(&self) -> String {
        self.inner.render_tree()
    }
}

fn diagram_from_graph(graph: &UntypedGraph) -> Diagram {
    let mut nodes = Vec::new();
    for node in &graph.nodes {
        nodes.push(DiagramNode {
            id: node_id(node.id),
            label: Some(node_label(graph, node)),
            kind: diagram_kind(&node.kind),
        });
    }

    for mark in &graph.marks {
        nodes.push(DiagramNode {
            id: mark_id(mark.id),
            label: Some(format!("mark {}", concise_type_name(&mark.type_spec.name))),
            kind: DiagramNodeKind::Mark,
        });
    }

    let mut edges = Vec::new();
    let mut terminal_ids = HashSet::new();
    let entry = entry_target(graph);
    add_value_edges(graph, &mut edges);
    add_mark_edges(graph, &mut edges);
    add_exit_edge(graph, &mut edges, &mut terminal_ids);

    for id in terminal_ids {
        nodes.push(DiagramNode {
            id,
            label: Some("exit".into()),
            kind: DiagramNodeKind::Terminal,
        });
    }

    Diagram::new(entry, nodes, edges)
}

fn add_value_edges(graph: &UntypedGraph, edges: &mut Vec<DiagramEdge>) {
    for edge in &graph.edges {
        let Some(from) = edge_source(graph, edge.id) else {
            continue;
        };
        if from == "_start" {
            continue;
        }
        let label = edge_label(graph, edge.id);
        for consumer in &edge.consumers {
            if graph.node(*consumer).is_some() {
                edges.push(DiagramEdge {
                    from: from.clone(),
                    to: node_ref(graph, *consumer),
                    label: label.clone(),
                    kind: DiagramEdgeKind::Data,
                });
            }
        }
    }
}

fn add_mark_edges(graph: &UntypedGraph, edges: &mut Vec<DiagramEdge>) {
    for mark in &graph.marks {
        let mark_node = mark_id(mark.id);
        for consumer in graph
            .edge(mark.target)
            .map(|edge| edge.consumers.as_slice())
            .unwrap_or_default()
        {
            if graph.node(*consumer).is_some() {
                edges.push(DiagramEdge {
                    from: mark_node.clone(),
                    to: node_ref(graph, *consumer),
                    label: "reenter".into(),
                    kind: DiagramEdgeKind::Control,
                });
            }
        }
    }

    for node in &graph.nodes {
        let NodeKind::Goto { mark } = node.kind else {
            continue;
        };
        edges.push(DiagramEdge {
            from: node_id(node.id),
            to: mark_id(mark),
            label: "goto".into(),
            kind: DiagramEdgeKind::Control,
        });
    }
}

fn add_exit_edge(
    graph: &UntypedGraph,
    edges: &mut Vec<DiagramEdge>,
    terminal_ids: &mut HashSet<String>,
) {
    let exit_id = "_exit".to_string();
    terminal_ids.insert(exit_id.clone());
    if let Some(from) = edge_source(graph, graph.exit) {
        edges.push(DiagramEdge {
            from,
            to: exit_id,
            label: edge_label(graph, graph.exit),
            kind: DiagramEdgeKind::Data,
        });
    }
}

fn entry_target(graph: &UntypedGraph) -> String {
    graph
        .edge(graph.entry)
        .and_then(|edge| edge.consumers.first().copied())
        .map(|node| node_ref(graph, node))
        .or_else(|| {
            graph
                .edge(graph.entry)
                .and_then(|edge| edge.producer)
                .map(|node| node_ref(graph, node))
        })
        .unwrap_or_else(|| "_exit".into())
}

fn edge_source(graph: &UntypedGraph, edge: EdgeId) -> Option<String> {
    let edge_data = graph.edge(edge)?;
    if edge == graph.entry && edge_data.producer.is_none() {
        return Some("_start".into());
    }
    edge_data.producer.map(|producer| node_ref(graph, producer))
}

fn node_ref(graph: &UntypedGraph, node: NodeId) -> String {
    graph
        .node(node)
        .map(|node| node_id(node.id))
        .unwrap_or_else(|| node_id(node))
}

fn node_id(id: NodeId) -> String {
    format!("node_{}", id.0)
}

fn mark_id(id: MarkId) -> String {
    format!("mark_{}", id.0)
}

fn edge_label(graph: &UntypedGraph, edge: EdgeId) -> String {
    graph
        .edge(edge)
        .map(|edge| concise_type_name(&edge.type_spec.name))
        .filter(|label| !label.is_empty())
        .or_else(|| graph.edge(edge).and_then(|edge| edge.label.clone()))
        .unwrap_or_else(|| format!("edge_{}", edge.0))
}

fn node_label(graph: &UntypedGraph, node: &super::model::Node) -> String {
    let input_types = node
        .inputs
        .iter()
        .filter_map(|edge| graph.edge(*edge))
        .map(|edge| concise_type_name(&edge.type_spec.name))
        .collect::<Vec<_>>();
    let output_types = node
        .outputs
        .iter()
        .filter_map(|edge| graph.edge(*edge))
        .map(|edge| concise_type_name(&edge.type_spec.name))
        .collect::<Vec<_>>();

    match &node.kind {
        NodeKind::Builtin { op } => match op {
            BuiltinNode::Identity => output_types
                .first()
                .cloned()
                .unwrap_or_else(|| "identity".into()),
            BuiltinNode::FanOut => format_type_flow(&input_types, &output_types),
            BuiltinNode::PackTuple => format!("pack {}", output_types.join(", ")),
            BuiltinNode::UnpackTuple => format!("unpack {}", output_types.join(", ")),
        },
        NodeKind::PureHandler { .. } => format_type_flow(&input_types, &output_types),
        NodeKind::WorkHandler { .. } => format_type_flow(&input_types, &output_types),
        NodeKind::Continuation { payload, .. } if looks_like_agent_payload(payload) => output_types
            .first()
            .cloned()
            .unwrap_or_else(|| "agent".into()),
        NodeKind::Continuation { .. } => format_type_flow(&input_types, &output_types),
        NodeKind::Suspend { .. } => output_types
            .first()
            .cloned()
            .map(|ty| format!("resume {ty}"))
            .unwrap_or_else(|| "suspend".into()),
        NodeKind::Load { var, .. } => graph
            .variable(*var)
            .map(|var| format!("load {}", concise_type_name(&var.type_spec.name)))
            .unwrap_or_else(|| "load".into()),
        NodeKind::Store { var, .. } => graph
            .variable(*var)
            .map(|var| format!("store {}", concise_type_name(&var.type_spec.name)))
            .unwrap_or_else(|| "store".into()),
        NodeKind::Subflow { graph: child } => {
            format!("flow {}", edge_type_name(child, child.exit))
        }
        NodeKind::Either { .. } => output_types
            .first()
            .cloned()
            .map(|ty| format!("choose {ty}"))
            .unwrap_or_else(|| "either".into()),
        NodeKind::Each { graph: child } => {
            format!("each {}", edge_type_name(child, child.exit))
        }
        NodeKind::Goto { mark } => graph
            .mark(*mark)
            .map(|mark| format!("goto {}", concise_type_name(&mark.type_spec.name)))
            .unwrap_or_else(|| "goto".into()),
    }
}

fn format_type_flow(inputs: &[String], outputs: &[String]) -> String {
    match (inputs, outputs) {
        (_, []) => inputs.first().cloned().unwrap_or_else(|| "node".into()),
        ([], outputs) => outputs.join(", "),
        ([input], [output]) if input == output => output.clone(),
        ([input], outputs) => format!("{input} -> {}", outputs.join(", ")),
        (inputs, outputs) => format!("{} -> {}", inputs.join(", "), outputs.join(", ")),
    }
}

fn edge_type_name(graph: &UntypedGraph, edge: EdgeId) -> String {
    graph
        .edge(edge)
        .map(|edge| concise_type_name(&edge.type_spec.name))
        .unwrap_or_else(|| format!("edge_{}", edge.0))
}

fn concise_type_name(name: &str) -> String {
    let mut name = name
        .rsplit("::")
        .next()
        .unwrap_or(name)
        .replace("alloc::vec::Vec", "Vec")
        .replace("std::vec::Vec", "Vec")
        .replace(' ', "");
    if name.len() > 36 {
        name.truncate(33);
        name.push_str("...");
    }
    name
}

fn diagram_kind(kind: &NodeKind) -> DiagramNodeKind {
    match kind {
        NodeKind::Builtin { op } => match op {
            BuiltinNode::Identity
            | BuiltinNode::FanOut
            | BuiltinNode::PackTuple
            | BuiltinNode::UnpackTuple => DiagramNodeKind::Builtin,
        },
        NodeKind::PureHandler { .. } => DiagramNodeKind::Map,
        NodeKind::WorkHandler { .. } => DiagramNodeKind::Work,
        NodeKind::Continuation { payload, .. } if looks_like_agent_payload(payload) => {
            DiagramNodeKind::Agent
        }
        NodeKind::Continuation { .. } => DiagramNodeKind::Continuation,
        NodeKind::Suspend { .. } => DiagramNodeKind::Suspend,
        NodeKind::Load { .. } => DiagramNodeKind::Load,
        NodeKind::Store { .. } => DiagramNodeKind::Store,
        NodeKind::Subflow { .. } => DiagramNodeKind::Flow,
        NodeKind::Either { .. } => DiagramNodeKind::Either,
        NodeKind::Each { .. } => DiagramNodeKind::Each,
        NodeKind::Goto { .. } => DiagramNodeKind::Goto,
    }
}

fn looks_like_agent_payload(payload: &super::Value) -> bool {
    payload.get("agent_id").is_some() && payload.get("output_schema").is_some()
}
