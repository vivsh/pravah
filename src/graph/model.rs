use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use super::ids::{EdgeId, HandlerKey, MarkId, NodeId, VarId};
use super::value::Value;

/// Current serialized untyped graph schema version.
pub const UNTYPED_GRAPH_SCHEMA_VERSION: u32 = 1;

/// JSON Schema metadata associated with an edge or variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeSpec {
    /// Human-readable type name used in errors and metadata.
    pub name: String,
    /// JSON-schema-like metadata used for cheap runtime shape checks.
    pub schema: JsonValue,
}

impl TypeSpec {
    /// Creates lightweight schema metadata for an edge or variable.
    pub fn new(name: impl Into<String>, schema: JsonValue) -> Self {
        Self {
            name: name.into(),
            schema,
        }
    }
}

/// Serializable variable identity. Rust fluent variables derive this from type
/// metadata; JSON graphs provide it directly.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VarKey {
    /// Key namespace, such as `rust` for typed fluent variables.
    pub namespace: String,
    /// Stable type name within the namespace.
    pub type_name: String,
}

impl VarKey {
    /// Creates a stable variable key for typed or JSON graph frontends.
    pub fn new(namespace: impl Into<String>, type_name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            type_name: type_name.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Variable visibility inside the VM frame stack.
///
/// `Local` is frame-owned. `Inherit` copies a matching parent variable into a
/// child frame, or falls back to its declared initializer.
pub enum VarScope {
    /// Variable belongs to the current frame only.
    Local,
    /// Variable may copy a matching parent variable on child-frame creation.
    Inherit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Deterministic variable initialization policy.
pub enum VarInit {
    /// Initialize the variable with this serialized value.
    Value(Value),
    /// Leave the variable unset until workflow logic writes it.
    Uninitialized,
}

/// A typed value channel in the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    /// Dense edge id matching this edge's vector position.
    pub id: EdgeId,
    /// Optional label for diagnostics and serialized graph readability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Type metadata for values carried by this edge.
    pub type_spec: TypeSpec,
    /// Node that writes this edge, absent only for entry/external edges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer: Option<NodeId>,
    /// Nodes that read this edge.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub consumers: Vec<NodeId>,
}

/// A typed workflow variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variable {
    /// Dense variable id matching this variable's vector position.
    pub id: VarId,
    /// Serializable variable identity.
    pub key: VarKey,
    /// Type metadata for stored variable values.
    pub type_spec: TypeSpec,
    /// Frame inheritance behavior.
    pub scope: VarScope,
    /// Initial value policy.
    pub init: VarInit,
}

/// A typed re-entry point for imperative control flow.
///
/// `goto(mark)` writes a fresh generation to the target edge, which lets
/// downstream nodes run again without inventing wrapper types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mark {
    /// Dense mark id matching this mark's vector position.
    pub id: MarkId,
    /// Optional label for diagnostics and serialized graph readability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Edge re-entered by `goto` nodes.
    pub target: EdgeId,
    /// Type metadata copied from the target edge.
    pub type_spec: TypeSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Built-in pure data-shaping operations understood by the VM.
///
/// These need no runtime registry because their behavior is fully described by
/// the graph.
pub enum BuiltinNode {
    /// Passes one input value to one output edge.
    Identity,
    /// Clones one input value to all output edges.
    FanOut,
    /// Packs multiple input values into one JSON array.
    PackTuple,
    /// Unpacks one JSON array into multiple output edges.
    UnpackTuple,
}

/// Serializable operation kind. User code is referenced by handler key only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NodeKind {
    /// Built-in pure control/data-shaping node.
    Builtin { op: BuiltinNode },
    /// Synchronous pure transform resolved from the value-handler registry.
    PureHandler { key: HandlerKey },
    /// One-shot async operation resolved from the work-handler registry.
    WorkHandler { key: HandlerKey },
    /// Multi-step VM operation resolved from the continuation registry.
    ///
    /// Continuations can keep checkpoints and call child graphs, but external
    /// pause/resume belongs to `Suspend`.
    Continuation {
        key: HandlerKey,
        #[serde(default)]
        payload: Value,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        children: Vec<UntypedGraph>,
    },
    /// First-class VM suspension point.
    ///
    /// `next()` returns the payload, and `resume()` writes the resume value to
    /// this node's output edge after type validation.
    Suspend {
        resume_type: String,
        #[serde(default)]
        payload: Value,
    },
    /// Reads a frame variable and transforms the input value.
    Load { var: VarId, key: HandlerKey },
    /// Updates a frame variable while passing the input value onward.
    Store { var: VarId, key: HandlerKey },
    /// Calls a child graph by pushing a child VM frame.
    Subflow { graph: Box<UntypedGraph> },
    /// Routes one value through either the left or right child graph.
    Either {
        key: HandlerKey,
        left: Box<UntypedGraph>,
        right: Box<UntypedGraph>,
    },
    /// Runs a child graph sequentially for each item in an input array.
    Each { graph: Box<UntypedGraph> },
    /// Writes this node's input value to a marked re-entry edge.
    Goto { mark: MarkId },
}

/// A graph operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Dense node id matching this node's vector position.
    pub id: NodeId,
    /// Diagnostic/display name.
    pub name: String,
    /// Operation behavior.
    pub kind: NodeKind,
    /// Input edges consumed by this node.
    pub inputs: Vec<EdgeId>,
    /// Output edges produced by this node.
    pub outputs: Vec<EdgeId>,
}

/// Fully serializable graph data. Runtime handlers and state live elsewhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UntypedGraph {
    /// Serialized graph schema version.
    pub schema_version: u32,
    /// Diagnostic graph name.
    pub name: String,
    /// Ordered edge table; ids must match positions.
    pub edges: Vec<Edge>,
    /// Ordered variable table; ids must match positions.
    pub variables: Vec<Variable>,
    /// Ordered mark table; ids must match positions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub marks: Vec<Mark>,
    /// Ordered node table; ids must match positions.
    pub nodes: Vec<Node>,
    /// Entry edge written before execution starts.
    pub entry: EdgeId,
    /// Exit edge whose readiness completes a frame.
    pub exit: EdgeId,
}

impl UntypedGraph {
    /// Looks up an edge by dense id and verifies the id matches its slot.
    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(id.0).filter(|edge| edge.id == id)
    }

    /// Looks up a variable by dense id and verifies the id matches its slot.
    pub fn variable(&self, id: VarId) -> Option<&Variable> {
        self.variables.get(id.0).filter(|var| var.id == id)
    }

    /// Looks up a mark by dense id and verifies the id matches its slot.
    pub fn mark(&self, id: MarkId) -> Option<&Mark> {
        self.marks.get(id.0).filter(|mark| mark.id == id)
    }

    /// Looks up a node by dense id and verifies the id matches its slot.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0).filter(|node| node.id == id)
    }
}
