use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::NodeId;

/// Stable authored identity of an embedded graph call site.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CallSite {
    Subflow { node: NodeId },
    Each { node: NodeId },
    EitherLeft { node: NodeId },
    EitherRight { node: NodeId },
    Continuation { node: NodeId, child_index: usize },
}

/// Stable path from the root graph to one embedded authored graph.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GraphPath(Arc<[CallSite]>);

impl GraphPath {
    pub(super) fn root() -> Self {
        Self::default()
    }

    pub(super) fn child(&self, call_site: CallSite) -> Self {
        let mut path = self.0.to_vec();
        path.push(call_site);
        Self(Arc::from(path))
    }
}
