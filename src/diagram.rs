//! Shared diagram rendering primitives.
//!
//! Runtime-specific modules build a neutral [`Diagram`], then reuse these
//! Mermaid, DOT, and tree renderers.

use std::collections::{HashMap, HashSet};

/// Node kind used by the diagram renderers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagramNodeKind {
    Agent,
    /// Generic multi-step continuation node.
    Continuation,
    Work,
    /// Tool-backed work node that routes non-fatal errors back to the model.
    ToolWork,
    /// Pure synchronous transform.
    Map,
    /// Built-in VM data-shaping node.
    Builtin,
    Fork,
    Join,
    Either,
    /// Flow-level suspend point.
    Suspend,
    /// Embedded child flow.
    Flow,
    /// Fan-out node.
    Each,
    /// Variable read/update transform.
    Load,
    /// Variable update node.
    Store,
    /// Control-flow re-entry write.
    Goto,
    /// Control-flow re-entry target.
    Mark,
    /// Edge target with no node definition in the graph.
    Terminal,
}

/// How an edge should be treated by renderers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagramEdgeKind {
    /// Normal value/data dependency.
    Data,
    /// Control-flow hint such as mark/goto/reenter.
    Control,
}

impl DiagramNodeKind {
    fn label_suffix(&self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Continuation => "continuation",
            Self::Work => "work",
            Self::ToolWork => "tool_work",
            Self::Map => "map",
            Self::Builtin => "builtin",
            Self::Fork => "fork",
            Self::Join => "join",
            Self::Either => "either",
            Self::Suspend => "suspend",
            Self::Flow => "flow",
            Self::Each => "each",
            Self::Load => "load",
            Self::Store => "store",
            Self::Goto => "goto",
            Self::Mark => "mark",
            Self::Terminal => "terminal",
        }
    }
}

/// One diagram node.
#[derive(Debug, Clone)]
pub struct DiagramNode {
    /// Unique node identifier.
    pub id: String,
    /// Human-facing label. Falls back to `id` when absent.
    pub label: Option<String>,
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
    /// Edge label.
    pub label: String,
    /// Rendering treatment for layout and styling.
    pub kind: DiagramEdgeKind,
}

/// Snapshot of a graph topology for diagram rendering.
#[derive(Debug, Clone)]
pub struct Diagram {
    entry: String,
    nodes: Vec<DiagramNode>,
    edges: Vec<DiagramEdge>,
}

impl Diagram {
    /// Constructs a diagram from already-normalized nodes and edges.
    pub fn new(entry: String, nodes: Vec<DiagramNode>, edges: Vec<DiagramEdge>) -> Self {
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
            let display = node_label(node);
            let decl = match node.kind {
                DiagramNodeKind::Agent
                | DiagramNodeKind::Continuation
                | DiagramNodeKind::Work
                | DiagramNodeKind::ToolWork
                | DiagramNodeKind::Map
                | DiagramNodeKind::Builtin
                | DiagramNodeKind::Load
                | DiagramNodeKind::Store => {
                    format!(
                        "    {}[\"{} ({})\"]",
                        safe_id,
                        display,
                        node.kind.label_suffix()
                    )
                }
                DiagramNodeKind::Fork | DiagramNodeKind::Either | DiagramNodeKind::Goto => {
                    format!(
                        "    {}{{\"{} ({})\"}}",
                        safe_id,
                        display,
                        node.kind.label_suffix()
                    )
                }
                DiagramNodeKind::Suspend => {
                    format!("    {}{{{{\"{}  (suspend)\"}}}}", safe_id, display)
                }
                DiagramNodeKind::Join => {
                    format!("    {}([\"{}  (join)\"])", safe_id, display)
                }
                DiagramNodeKind::Terminal => {
                    format!("    {}([\"{}  ◉\"])", safe_id, display)
                }
                DiagramNodeKind::Flow => {
                    format!("    {}[\"\\[{} (flow)\\]\"]", safe_id, display)
                }
                DiagramNodeKind::Each => {
                    format!("    {}[\"\\[{} (each)\\]\"]", safe_id, display)
                }
                DiagramNodeKind::Mark => {
                    format!("    {}((\"{}\"))", safe_id, display)
                }
            };
            out.push_str(&decl);
            out.push('\n');
        }

        out.push_str(&format!("    _start --> {}\n", mermaid_id(&self.entry)));

        for edge in &self.edges {
            let arrow = match edge.kind {
                DiagramEdgeKind::Data => "-->",
                DiagramEdgeKind::Control => "-.->",
            };
            if edge.label.is_empty() {
                out.push_str(&format!(
                    "    {} {} {}\n",
                    mermaid_id(&edge.from),
                    arrow,
                    mermaid_id(&edge.to)
                ));
            } else {
                out.push_str(&format!(
                    "    {} {}|{}| {}\n",
                    mermaid_id(&edge.from),
                    arrow,
                    edge.label,
                    mermaid_id(&edge.to)
                ));
            }
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
            let display = node_label(node);
            let attrs = match node.kind {
                DiagramNodeKind::Agent
                | DiagramNodeKind::Continuation
                | DiagramNodeKind::Work
                | DiagramNodeKind::ToolWork
                | DiagramNodeKind::Map
                | DiagramNodeKind::Builtin
                | DiagramNodeKind::Load
                | DiagramNodeKind::Store => format!(
                    "label=\"{}\\n({})\" shape=box style=rounded",
                    dot_escape(display),
                    node.kind.label_suffix()
                ),
                DiagramNodeKind::Fork | DiagramNodeKind::Either | DiagramNodeKind::Goto => {
                    format!(
                        "label=\"{}\\n({})\" shape=diamond",
                        dot_escape(display),
                        node.kind.label_suffix()
                    )
                }
                DiagramNodeKind::Suspend => {
                    format!(
                        "label=\"{}\\n(suspend)\" shape=hexagon",
                        dot_escape(display)
                    )
                }
                DiagramNodeKind::Join => {
                    format!("label=\"{}\\n(join)\" shape=ellipse", dot_escape(display))
                }
                DiagramNodeKind::Terminal => {
                    format!("label=\"{}\" shape=doublecircle", dot_escape(display))
                }
                DiagramNodeKind::Flow => {
                    format!("label=\"{}\\n(flow)\" shape=box3d", dot_escape(display))
                }
                DiagramNodeKind::Each => {
                    format!("label=\"{}\\n(each)\" shape=box3d", dot_escape(display))
                }
                DiagramNodeKind::Mark => format!(
                    "label=\"\" xlabel=\"{}\" shape=point width=0.08 height=0.08 color=\"#8a6a00\"",
                    dot_escape(display)
                ),
            };
            out.push_str(&format!("    {} [{}];\n", safe_id, attrs));
        }

        out.push_str(&format!("    _start -> {};\n", dot_id(&self.entry)));

        for edge in &self.edges {
            let attrs = match edge.kind {
                DiagramEdgeKind::Data => {
                    if edge.label.is_empty() {
                        String::new()
                    } else {
                        format!("label=\"{}\"", dot_escape(&edge.label))
                    }
                }
                DiagramEdgeKind::Control => {
                    let label = if edge.label.is_empty() {
                        String::new()
                    } else {
                        format!("label=\"{}\" ", dot_escape(&edge.label))
                    };
                    format!(
                        "{label}style=dashed color=\"#8a6a00\" fontcolor=\"#8a6a00\" constraint=false"
                    )
                }
            };
            if attrs.is_empty() {
                out.push_str(&format!(
                    "    {} -> {};\n",
                    dot_id(&edge.from),
                    dot_id(&edge.to),
                ));
            } else {
                out.push_str(&format!(
                    "    {} -> {} [{}];\n",
                    dot_id(&edge.from),
                    dot_id(&edge.to),
                    attrs,
                ));
            }
        }

        out.push_str("}\n");
        out
    }

    /// Renders the graph as an indented execution tree.
    ///
    /// Revisited nodes are marked with `↩` instead of being expanded again.
    pub fn render_tree(&self) -> String {
        let mut adj: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
        for node in &self.nodes {
            adj.entry(node.id.as_str()).or_default();
        }
        for edge in &self.edges {
            adj.entry(edge.from.as_str())
                .or_default()
                .push((edge.label.as_str(), edge.to.as_str()));
        }
        for succs in adj.values_mut() {
            succs.sort_by_key(|(_, to)| *to);
        }

        let node_kind: HashMap<&str, &DiagramNodeKind> = self
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), &n.kind))
            .collect();
        let node_labels: HashMap<&str, &str> = self
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), node_label(n)))
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
            &node_labels,
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
    format!("\"{}\"", dot_escape(id))
}

fn dot_escape(input: &str) -> String {
    input.replace('"', "\\\"")
}

fn node_label(node: &DiagramNode) -> &str {
    node.label.as_deref().unwrap_or(&node.id)
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
    node_labels: &HashMap<&str, &str>,
    out: &mut String,
) {
    let repeated = visited.contains(id);

    let kind_tag = match node_kind.get(id).copied() {
        Some(kind) => match kind {
            DiagramNodeKind::Terminal => " ◉".to_string(),
            other => format!(" ({})", other.label_suffix()),
        },
        None => String::new(),
    };
    let label = node_labels.get(id).copied().unwrap_or(id);
    let display = if repeated {
        format!("{label}{kind_tag} ↩")
    } else {
        format!("{label}{kind_tag}")
    };

    if is_root {
        out.push_str(&format!("● {display}\n"));
    } else {
        let connector = if is_last { "└── " } else { "├── " };
        let edge_part = match edge_label {
            Some(l) if !l.is_empty() => format!("[{l}] "),
            _ => String::new(),
        };
        out.push_str(&format!("{prefix}{connector}{edge_part}{display}\n"));
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
        format!("{prefix}    ")
    } else {
        format!("{prefix}│   ")
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
            node_labels,
            out,
        );
    }
}
