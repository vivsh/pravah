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

use std::collections::HashSet;

use super::flows::{Flow, FlowError};

// ── Public data model ──────────────────────────────────────────────────────

/// The kind of node in the flow graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagramNodeKind {
    Agent,
    Work,
    Fork,
    Join,
    Either,
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
    pub fn for_flow<F: Flow>() -> Result<Self, FlowError> {
        let graph = F::build()?.with_entry(F::node_id())?;
        Ok(graph.diagram())
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
                    format!(
                        "    {}([\"{}  (join)\"])",
                        safe_id, node.id
                    )
                }
                DiagramNodeKind::Terminal => {
                    format!("    {}([\"{}  ◉\"])", safe_id, node.id)
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

    // ── mermaid-text (feature-gated) ───────────────────────────────────────

    /// Render to Unicode box-drawing text using the `mermaid-text` crate.
    ///
    /// Requires the `diagram-text` feature.
    #[cfg(feature = "diagram-text")]
    pub fn render_text(&self) -> Result<String, mermaid_text::Error> {
        mermaid_text::render(&self.mermaid())
    }

    /// Render to plain ASCII using the `mermaid-text` crate.
    ///
    /// Requires the `diagram-text` feature.
    #[cfg(feature = "diagram-text")]
    pub fn render_ascii(&self) -> Result<String, mermaid_text::Error> {
        mermaid_text::render_ascii(&self.mermaid())
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Sanitise a node id for use as a Mermaid node identifier.
/// Mermaid identifiers must be alphanumeric + underscore only.
fn mermaid_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Sanitise a node id for use as a DOT identifier (wrap in quotes).
fn dot_id(id: &str) -> String {
    // DOT allows any string inside double-quotes; escape existing quotes.
    format!("\"{}\"", id.replace('"', "\\\""))
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
