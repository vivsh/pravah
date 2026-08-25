//! Diagram rendering for old flow graphs.
//! [`FlowGraphDiagram`] snapshots old `FlowGraph` topology and renders Mermaid,
//! DOT, or an indented execution tree.
//!
//! ```ignore
//! let diagram = FlowGraphDiagram::from_flow::<MyFlow>()?;
//! println!("{}", diagram.dot());
//! println!("{}", diagram.mermaid());
//! ```

use std::collections::HashSet;

use crate::diagram::Diagram;
pub use crate::diagram::{DiagramEdge, DiagramNode, DiagramNodeKind};
use crate::legacy::{Flow, FlowGraph};

use super::errors::FlowError;
use super::nodes::FlowNode;

/// Snapshot of an old flow graph topology for diagram rendering.
#[derive(Debug, Clone)]
pub struct FlowGraphDiagram {
    inner: Diagram,
}

impl FlowGraphDiagram {
    /// Builds a diagram for flow `F`.
    ///
    /// This only inspects the graph definition; it does not execute the flow.
    pub fn from_flow<F: Flow>() -> Result<Self, FlowError> {
        let graph = FlowGraph::from_flow::<F>()?;
        Ok(diagram_from_graph(&graph))
    }

    /// Constructs a new diagram from normalized shared-renderer data.
    pub(crate) fn new(entry: String, nodes: Vec<DiagramNode>, edges: Vec<DiagramEdge>) -> Self {
        Self {
            inner: Diagram::new(entry, nodes, edges),
        }
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

/// Diagram-ready node description derived from `FlowNode`.
pub(crate) struct NodeDesc {
    pub id: String,
    pub kind: DiagramNodeKind,
    pub succs: Vec<(String, &'static str)>,
}

/// Builds a [`FlowGraphDiagram`] from node descriptions.
pub(crate) fn build_diagram(entry: String, descs: Vec<NodeDesc>) -> FlowGraphDiagram {
    let defined_ids: HashSet<&str> = descs.iter().map(|d| d.id.as_str()).collect();

    let mut nodes: Vec<DiagramNode> = descs
        .iter()
        .map(|d| DiagramNode {
            id: d.id.clone(),
            label: None,
            kind: d.kind.clone(),
        })
        .collect();

    let mut edges: Vec<DiagramEdge> = Vec::new();
    let mut terminal_ids: HashSet<String> = HashSet::new();

    for desc in &descs {
        for (to, label) in &desc.succs {
            edges.push(DiagramEdge {
                from: desc.id.clone(),
                to: to.clone(),
                label: (*label).to_string(),
                kind: crate::diagram::DiagramEdgeKind::Data,
            });
            if !defined_ids.contains(to.as_str()) {
                terminal_ids.insert(to.clone());
            }
        }
    }

    for id in terminal_ids {
        nodes.push(DiagramNode {
            id,
            label: None,
            kind: DiagramNodeKind::Terminal,
        });
    }

    FlowGraphDiagram::new(entry, nodes, edges)
}

fn diagram_from_graph(graph: &FlowGraph) -> FlowGraphDiagram {
    let descs: Vec<NodeDesc> = graph
        .nodes
        .iter()
        .map(|(key, node)| {
            let key_str = graph.interner.name_of(*key).to_string();
            let (kind, succs): (DiagramNodeKind, Vec<(String, &'static str)>) = match node {
                FlowNode::Agent(info) => (
                    DiagramNodeKind::Agent,
                    vec![(graph.interner.name_of(info.exit).to_string(), "agent")],
                ),
                FlowNode::Work(info) => (
                    DiagramNodeKind::Work,
                    vec![(graph.interner.name_of(info.exit_name).to_string(), "work")],
                ),
                FlowNode::ToolWork(info) => (
                    DiagramNodeKind::ToolWork,
                    vec![(
                        graph.interner.name_of(info.exit_name).to_string(),
                        "tool_work",
                    )],
                ),
                FlowNode::Map(info) => (
                    DiagramNodeKind::Map,
                    vec![(graph.interner.name_of(info.exit_name).to_string(), "map")],
                ),
                FlowNode::Suspend(info) => (
                    DiagramNodeKind::Suspend,
                    vec![(graph.interner.name_of(info.exit).to_string(), "suspend")],
                ),
                FlowNode::Fork(info) => (
                    DiagramNodeKind::Fork,
                    info.children
                        .iter()
                        .map(|&c| (graph.interner.name_of(c).to_string(), "fork"))
                        .collect(),
                ),
                FlowNode::Join(info) => (
                    DiagramNodeKind::Join,
                    vec![(graph.interner.name_of(info.target).to_string(), "join")],
                ),
                FlowNode::Either(info) => (
                    DiagramNodeKind::Either,
                    vec![
                        (graph.interner.name_of(info.left_name).to_string(), "either"),
                        (
                            graph.interner.name_of(info.right_name).to_string(),
                            "either",
                        ),
                    ],
                ),
                FlowNode::Flow(inner) => {
                    let exit = inner.exit;
                    let exit_str = inner.interner.name_of(exit).to_string();
                    (DiagramNodeKind::Flow, vec![(exit_str, "flow")])
                }
                FlowNode::Each(info) => {
                    let exit_str = graph.interner.name_of(info.exit).to_string();
                    (DiagramNodeKind::Each, vec![(exit_str, "each")])
                }
            };
            NodeDesc {
                id: key_str,
                kind,
                succs,
            }
        })
        .collect();
    let entry_str = graph.interner.name_of(graph.entry).to_string();
    build_diagram(entry_str, descs)
}

#[cfg(test)]
mod tests {
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::legacy::Node;

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct DiagramInput(i64);

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct DiagramOutput(i64);

    impl Flow for DiagramInput {
        type Output = DiagramOutput;

        fn build(root: Node<Self>) -> Node<Self::Output> {
            root.with_builder(|builder| builder.map(|input: DiagramInput| DiagramOutput(input.0)))
        }
    }

    #[test]
    fn old_flow_diagram_still_renders() {
        let diagram = FlowGraphDiagram::from_flow::<DiagramInput>().expect("diagram should build");
        let mermaid = diagram.mermaid();
        let dot = diagram.dot();

        assert!(mermaid.contains("flowchart LR"));
        assert!(mermaid.contains("map"));
        assert!(dot.contains("digraph"));
        assert!(!diagram.nodes().is_empty());
    }
}
