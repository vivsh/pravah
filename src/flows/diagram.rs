//! Flow graph diagram generation.
//!
//! [`FlowGraphDiagram`] holds a snapshot of a flow graph's topology and can
//! render it as a Graphviz DOT file or a Mermaid flowchart.  With the
//! `diagram-text` feature enabled it can also render the Mermaid source to
//! Unicode box-drawing text or plain ASCII via the `mermaid-text` crate.
//!
//! # Example
//! ```ignore
//! let diagram = FlowGraphDiagram::for_flow::<MyFlow>()?;
//! println!("{}", diagram.dot());
//! println!("{}", diagram.mermaid());
//! ```

use std::collections::{HashMap, HashSet};

use crate::flows::FlowGraph;
use crate::flows::flows::FlowNode;

use super::errors::FlowError;
use super::flows::Flow;

// ── Public data model ──────────────────────────────────────────────────────

/// The kind of node in the flow graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagramNodeKind {
    Agent,
    Work,
    Fork,
    Join,
    Either,
    /// An embedded sub-flow node.
    Flow,
    /// A node that is the target of an edge but has no definition in the graph
    /// (i.e. the flow terminates there).
    Terminal,
}

impl DiagramNodeKind {
    fn label_suffix(&self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Work => "work",
            Self::Fork => "fork",
            Self::Join => "join",
            Self::Either => "either",
            Self::Flow => "flow",
            Self::Terminal => "terminal",
        }
    }
}

/// A single node in the flow graph.
#[derive(Debug, Clone)]
pub struct DiagramNode {
    pub id: String,
    pub kind: DiagramNodeKind,
}

/// A directed edge between two nodes.
#[derive(Debug, Clone)]
pub struct DiagramEdge {
    pub from: String,
    pub to: String,
    pub label: &'static str,
}

/// A snapshot of a flow graph's topology suitable for diagram rendering.
///
/// Obtain via [`FlowGraph::diagram`](super::flows::FlowGraph::diagram).
#[derive(Debug, Clone)]
pub struct FlowGraphDiagram {
    entry: String,
    nodes: Vec<DiagramNode>,
    edges: Vec<DiagramEdge>,
}

impl FlowGraphDiagram {
    /// Build and return a diagram for flow `F`.
    ///
    /// Calls `F::build()`, validates the graph, and snapshots the topology.
    /// No LLM calls are made.
    pub fn from_flow<F: Flow>() -> Result<Self, FlowError> {
        let graph = FlowGraph::from_flow::<F>()?;
        Ok(diagram_from_graph(&graph))
    }

    /// Construct a new diagram. Called by [`FlowGraph::diagram`].
    pub(crate) fn new(entry: String, nodes: Vec<DiagramNode>, edges: Vec<DiagramEdge>) -> Self {
        Self {
            entry,
            nodes,
            edges,
        }
    }

    /// The entry node id.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// All nodes in the diagram (including terminal nodes).
    pub fn nodes(&self) -> &[DiagramNode] {
        &self.nodes
    }

    /// All directed edges in the diagram.
    pub fn edges(&self) -> &[DiagramEdge] {
        &self.edges
    }

    // ── Mermaid ────────────────────────────────────────────────────────────

    /// Render the graph as a Mermaid `flowchart LR` source string.
    ///
    /// The output can be pasted into [mermaid.live](https://mermaid.live) or
    /// embedded in Markdown. With the `diagram-text` feature, pass the result
    /// to [`Self::render_text`] / [`Self::render_ascii`].
    pub fn mermaid(&self) -> String {
        let mut out = String::from("flowchart LR\n");

        // Sentinel start node
        out.push_str("    _start(( ))\n");

        // Node declarations
        for node in &self.nodes {
            let safe_id = mermaid_id(&node.id);
            let decl = match node.kind {
                // Rectangle with label + kind
                DiagramNodeKind::Agent | DiagramNodeKind::Work => {
                    format!(
                        "    {}[\"{} ({})\"]",
                        safe_id,
                        node.id,
                        node.kind.label_suffix()
                    )
                }
                // Diamond for fork / either (branching)
                DiagramNodeKind::Fork | DiagramNodeKind::Either => {
                    format!(
                        "    {}{{\"{} ({})\"}}",
                        safe_id,
                        node.id,
                        node.kind.label_suffix()
                    )
                }
                // Stadium for join / terminal
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

        // Entry edge from sentinel
        out.push_str(&format!("    _start --> {}\n", mermaid_id(&self.entry)));

        // Graph edges
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

    // ── DOT ───────────────────────────────────────────────────────────────

    /// Render the graph as a Graphviz DOT source string.
    ///
    /// Pass the result to `dot -Tpng -o flow.png` or similar.
    pub fn dot(&self) -> String {
        let mut out = String::from("digraph {\n    rankdir=LR;\n");

        // Sentinel start node
        out.push_str(
            "    _start [label=\"\" shape=circle style=filled fillcolor=black width=0.3];\n",
        );

        // Node declarations
        for node in &self.nodes {
            let safe_id = dot_id(&node.id);
            let attrs = match node.kind {
                DiagramNodeKind::Agent | DiagramNodeKind::Work => format!(
                    "label=\"{}\\n({})\" shape=box style=rounded",
                    node.id,
                    node.kind.label_suffix()
                ),
                DiagramNodeKind::Fork | DiagramNodeKind::Either => format!(
                    "label=\"{}\\n({})\" shape=diamond",
                    node.id,
                    node.kind.label_suffix()
                ),
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

        // Entry edge from sentinel
        out.push_str(&format!("    _start -> {};\n", dot_id(&self.entry)));

        // Graph edges
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

    /// Render the graph as an indented tree showing the execution path from
    /// the entry node through all branches and convergence points.
    ///
    /// Nodes reached via multiple branches (e.g. join targets) are rendered
    /// in full on the first visit and marked `↩` on subsequent visits,
    /// so the complete topology can be read top-to-bottom without loops.
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

// ── Helpers ────────────────────────────────────────────────────────────────

/// Sanitise a node id for use as a Mermaid node identifier.
/// Mermaid identifiers must be alphanumeric + underscore only.
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

/// Sanitise a node id for use as a DOT identifier (wrap in quotes).
fn dot_id(id: &str) -> String {
    // DOT allows any string inside double-quotes; escape existing quotes.
    format!("\"{}\"", id.replace('"', "\\\""))
}

// ── Tree renderer helper ───────────────────────────────────────────────────

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
        Some(DiagramNodeKind::Fork) => " (fork)",
        Some(DiagramNodeKind::Join) => " (join)",
        Some(DiagramNodeKind::Either) => " (either)",
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

// ── Build helper (called from flows.rs) ────────────────────────────────────

/// Snapshot node kinds from the private `FlowNode` enum.
/// We pass an iterator of `(id, kind, edges)` tuples.
pub(crate) struct NodeDesc {
    pub id: String,
    pub kind: DiagramNodeKind,
    pub succs: Vec<(String, &'static str)>,
}

/// Build a [`FlowGraphDiagram`] from a description of nodes.
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

    // Collect edges; mark targets that are not registered nodes as terminals.
    // Join nodes are registered under each *parent* key, both emitting an edge
    // to the same target — dedup the terminal detection but keep both edges.
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
                // Tool nodes are implementation details not shown in diagrams.
                FlowNode::Tool(_) => return None,
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
