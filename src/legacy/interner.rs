use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct NodeId(pub(crate) usize);

pub struct Interner {
    pub(crate) fwd: HashMap<String, NodeId>,
    pub(crate) rev: Vec<String>,
}

impl Interner {
    pub fn new() -> Self {
        Self {
            fwd: HashMap::new(),
            rev: Vec::new(),
        }
    }

    pub fn intern(&mut self, s: &str) -> NodeId {
        if let Some(&id) = self.fwd.get(s) {
            return id;
        }
        let id = NodeId(self.rev.len());
        self.rev.push(s.to_string());
        self.fwd.insert(s.to_string(), id);
        id
    }

    pub fn intern_get(&self, s: &str) -> Option<NodeId> {
        self.fwd.get(s).copied()
    }

    pub fn name_of(&self, id: NodeId) -> &str {
        self.rev
            .get(id.0)
            .map(String::as_str)
            .unwrap_or("<unknown>")
    }
}
