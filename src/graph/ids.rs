use serde::{Deserialize, Serialize};

/// Dense declaration-order identifier for an operation node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);

/// Dense declaration-order identifier for a typed value edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub usize);

/// Dense declaration-order identifier for a workflow variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VarId(pub usize);

/// Dense declaration-order identifier for a control-flow mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MarkId(pub usize);

/// Serializable key used to resolve runtime handlers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HandlerKey(pub String);

impl HandlerKey {
    /// Creates a serializable registry key from a deterministic string.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// Returns the key string used for registry lookup.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for HandlerKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
