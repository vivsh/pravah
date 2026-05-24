//! Diagram rendering for flow graphs.
//! [`FlowGraphDiagram`] snapshots graph topology and renders Mermaid or DOT output.
//! With `diagram-text`, the Mermaid source can also be rendered as text.
//!
//! ```ignore
//! let diagram = FlowGraphDiagram::for_flow::<MyFlow>()?;
//! println!("{}", diagram.dot());
//! println!("{}", diagram.mermaid());
//! ```

use std::collections::{HashMap, HashSet};

use crate::flows::{Flow, FlowGraph};

use super::errors::FlowError;
use super::nodes::FlowNode;

/// Node kind used by the diagram renderers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagramNodeKind {
    Agent,
    Work,
    /// Tool-backed work node that routes non-fatal errors back to the model.
    ToolWork,
    /// Pure synchronous transform.
    Map,
    Fork,
    Join,
    Either,
    /// Flow-level suspend point.
    Suspend,
    /// Embedded child flow.
    Flow,
    /// Edge target with no node definition in the graph.
    Terminal,
}

impl DiagramNodeKind {
    fn label_suffix(&self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Work => "work",
            Self::ToolWork => "tool_work",
            Self::Map => "map",
            Self::Fork => "fork",
            Self::Join => "join",
            Self::Either => "either",
            Self::Suspend => "suspend",
            Self::Flow => "flow",
            Self::Terminal => "terminal",
        }
    }
}

/// One diagram node.
#[derive(Debug, Clone)]
pub struct DiagramNode {
    /// Unique node identifier (matches the flow node id).
    pub id: String,
    /// Semantic kind used to choose the rendering shape.
    pub kind: DiagramNodeKind,
}

/// One directed edge in the diagram.
#[derive(Debug, Clone)]
pub struct DiagramEdge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Edge label (e.g. `"ok"`, `"err"`, `"default"`).
    pub label: &'static str,
}

/// Snapshot of a flow graph topology for diagram rendering.
#[derive(Debug, Clone)]
pub struct FlowGraphDiagram {
    entry: String,
    nodes: Vec<DiagramNode>,
    edges: Vec<DiagramEdge>,
}

impl FlowGraphDiagram {
    /// Builds a diagram for flow `F`.
    /// This only inspects the graph definition.
    pub fn from_flow<F: Flow>() -> Result<Self, FlowError> {
        let graph = FlowGraph::from_flow::<F>()?;
        Ok(diagram_from_graph(&graph))
    }

    /// Constructs a new diagram.
    pub(crate) fn new(entry: String, nodes: Vec<DiagramNode>, edges: Vec<DiagramEdge>) -> Self {
        Self {
            entry,
            nodes,
            edges,
        }
    }

    /// Returns the entry node id.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// Returns all diagram nodes, including terminals.
    pub fn nodes(&self) -> &[DiagramNode] {
        &self.nodes
    }

    /// Returns all directed edges.
    pub fn edges(&self) -> &[DiagramEdge] {
        &self.edges
    }

    /// Renders the graph as Mermaid `flowchart LR` source.
    pub fn mermaid(&self) -> String {
        let mut out = String::from("flowchart LR\n");

        out.push_str("    _start(( ))\n");

        for node in &self.nodes {
            let safe_id = mermaid_id(&node.id);
            let decl = match node.kind {
                DiagramNodeKind::Agent
                    | DiagramNodeKind::Work
                    | DiagramNodeKind::ToolWork
                    | DiagramNodeKind::Map => {
                    format!(
                        "    {}[\"{} ({})\"]",
                        safe_id,
                        node.id,
                        node.kind.label_suffix()
                    )
                }
                DiagramNodeKind::Fork | DiagramNodeKind::Either => {
                    format!(
                        "    {}{{\"{} ({})\"}}",
                        safe_id,
                        node.id,
                        node.kind.label_suffix()
                    )
                }
                DiagramNodeKind::Suspend => {
                    format!(
                        "    {}{{{{\"{}  (suspend)\"}}}}",
                        safe_id,
                        node.id
                    )
                }
                DiagramNodeKind::Join => {
                    format!("    {}([\"{}  (join)\"])", safe_id, node.id)
                }
                DiagramNodeKind::Terminal => {
                    format!("    {}([\"{}  ◉\"])", safe_id, node.id)
                }
                DiagramNodeKind::Flow => {
                    format!("    {}[\"\\[{} (flow)\\]\"]", safe_id, node.id)
                }
            };
            out.push_str(&decl);
            out.push('\n');
        }

        out.push_str(&format!("    _start --> {}\n", mermaid_id(&self.entry)));

        for edge in &self.edges {
            out.push_str(&format!(
                "    {} -->|{}| {}\n",
                mermaid_id(&edge.from),
                edge.label,
                mermaid_id(&edge.to)
            ));
        }

        out
    }

    /// Renders the graph as Graphviz DOT source.
    pub fn dot(&self) -> String {
        let mut out = String::from("digraph {\n    rankdir=LR;\n");

        out.push_str(
            "    _start [label=\"\" shape=circle style=filled fillcolor=black width=0.3];\n",
        );

        for node in &self.nodes {
            let safe_id = dot_id(&node.id);
            let attrs = match node.kind {
                DiagramNodeKind::Agent
                    | DiagramNodeKind::Work
                    | DiagramNodeKind::ToolWork
                    | DiagramNodeKind::Map => format!(
                    "label=\"{}\\n({})\" shape=box style=rounded",
                    node.id,
                    node.kind.label_suffix()
                ),
                DiagramNodeKind::Fork | DiagramNodeKind::Either => format!(
                    "label=\"{}\\n({})\" shape=diamond",
                    node.id,
                    node.kind.label_suffix()
                ),
                DiagramNodeKind::Suspend => {
                    format!("label=\"{}\\n(suspend)\" shape=hexagon", node.id)
                }
                DiagramNodeKind::Join => {
                    format!("label=\"{}\\n(join)\" shape=ellipse", node.id)
                }
                DiagramNodeKind::Terminal => {
                    format!("label=\"{}\" shape=doublecircle", node.id)
                }
                DiagramNodeKind::Flow => format!("label=\"{}\\n(flow)\" shape=box3d", node.id),
            };
            out.push_str(&format!("    {} [{}];\n", safe_id, attrs));
        }

        out.push_str(&format!("    _start -> {};\n", dot_id(&self.entry)));

        for edge in &self.edges {
            out.push_str(&format!(
                "    {} -> {} [label=\"{}\"];\n",
                dot_id(&edge.from),
                dot_id(&edge.to),
                edge.label,
            ));
        }

        out.push_str("}\n");
        out
    }

    /// Renders the graph as an indented execution tree.
    /// Revisited nodes are marked with `↩` instead of being expanded again.
    ///
    /// ```text
    /// ● ArticleRequest (fork)
    ///   ├── [fork] AudienceTask (agent)
    ///   │   └── [agent] AudienceProfile (join)
    ///   │       └── [join] ContentBrief (work)
    ///   │           └── [work] ...
    ///   └── [fork] ResearchTask (agent)
    ///       └── [agent] ResearchNotes (join)
    ///           └── [join] ContentBrief (work) ↩
    /// ```
    pub fn render_tree(&self) -> String {
        let mut adj: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
        for node in &self.nodes {
            adj.entry(node.id.as_str()).or_default();
        }
        for edge in &self.edges {
            adj.entry(edge.from.as_str())
                .or_default()
                .push((edge.label, edge.to.as_str()));
        }
        for succs in adj.values_mut() {
            succs.sort_by_key(|(_, to)| *to);
        }

        let node_kind: HashMap<&str, &DiagramNodeKind> = self
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), &n.kind))
            .collect();

        let mut visited: HashSet<String> = HashSet::new();
        let mut out = String::new();

        tree_write_node(
            &self.entry,
            "",
            true,
            true,
            None,
            &mut visited,
            &adj,
            &node_kind,
            &mut out,
        );

        out
    }
}

/// Sanitizes a node id for Mermaid.
fn mermaid_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Wraps a node id for DOT output.
fn dot_id(id: &str) -> String {
    format!("\"{}\"", id.replace('"', "\\\""))
}

#[allow(clippy::too_many_arguments)]
fn tree_write_node(
    id: &str,
    prefix: &str,
    is_root: bool,
    is_last: bool,
    edge_label: Option<&str>,
    visited: &mut HashSet<String>,
    adj: &HashMap<&str, Vec<(&str, &str)>>,
    node_kind: &HashMap<&str, &DiagramNodeKind>,
    out: &mut String,
) {
    let repeated = visited.contains(id);

    let kind_tag = match node_kind.get(id).copied() {
        Some(DiagramNodeKind::Agent) => " (agent)",
        Some(DiagramNodeKind::Work) => " (work)",
        Some(DiagramNodeKind::ToolWork) => " (tool_work)",
        Some(DiagramNodeKind::Map) => " (map)",
        Some(DiagramNodeKind::Fork) => " (fork)",
        Some(DiagramNodeKind::Join) => " (join)",
        Some(DiagramNodeKind::Either) => " (either)",
        Some(DiagramNodeKind::Suspend) => " (suspend)",
        Some(DiagramNodeKind::Flow) => " (flow)",
        Some(DiagramNodeKind::Terminal) => " ◉",
        None => "",
    };
    let display = if repeated {
        format!("{}{} ↩", id, kind_tag)
    } else {
        format!("{}{}", id, kind_tag)
    };

    if is_root {
        out.push_str(&format!("● {}\n", display));
    } else {
        let connector = if is_last { "└── " } else { "├── " };
        let edge_part = match edge_label {
            Some(l) if !l.is_empty() => format!("[{}] ", l),
            _ => String::new(),
        };
        out.push_str(&format!(
            "{}{}{}{}\n",
            prefix, connector, edge_part, display
        ));
    }

    if repeated {
        return;
    }
    visited.insert(id.to_string());

    let succs = match adj.get(id) {
        Some(v) if !v.is_empty() => v,
        _ => return,
    };

    let child_prefix = if is_root {
        "  ".to_string()
    } else if is_last {
        format!("{}    ", prefix)
    } else {
        format!("{}│   ", prefix)
    };

    for (i, (label, to)) in succs.iter().enumerate() {
        let is_last_child = i == succs.len() - 1;
        tree_write_node(
            to,
            &child_prefix,
            false,
            is_last_child,
            Some(label),
            visited,
            adj,
            node_kind,
            out,
        );
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
            kind: d.kind.clone(),
        })
        .collect();

    let mut edges: Vec<DiagramEdge> = Vec::new();
    let mut terminal_ids: HashSet<String> = HashSet::new();

    // Mark edge targets without node definitions as terminals.
    for desc in &descs {
        for (to, label) in &desc.succs {
            edges.push(DiagramEdge {
                from: desc.id.clone(),
                to: to.clone(),
                label,
            });
            if !defined_ids.contains(to.as_str()) {
                terminal_ids.insert(to.clone());
            }
        }
    }

    // Add terminal nodes
    for id in terminal_ids {
        nodes.push(DiagramNode {
            id,
            kind: DiagramNodeKind::Terminal,
        });
    }

    FlowGraphDiagram::new(entry, nodes, edges)
}

fn diagram_from_graph(graph: &FlowGraph) -> FlowGraphDiagram {
    let descs: Vec<NodeDesc> = graph
        .nodes
        .iter()
        .filter_map(|(key, node)| {
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
                    vec![(graph.interner.name_of(info.exit_name).to_string(), "tool_work")],
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
                        (graph.interner.name_of(info.right_name).to_string(), "either"),
                    ],
                ),
                FlowNode::Flow(inner) => {
                    let exit = inner
                        .exit;
                    let exit_str = inner.interner.name_of(exit).to_string();
                    (DiagramNodeKind::Flow, vec![(exit_str, "flow")])
                }
            };
            Some(NodeDesc {
                id: key_str,
                kind,
                succs,
            })
        })
        .collect();
    let entry_str = graph.interner.name_of(graph.entry).to_string();
    build_diagram(entry_str, descs)
}
