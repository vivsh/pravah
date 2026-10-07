//! Execution intent for durable conversation maintenance, independent of worker services.

use serde::{Deserialize, Serialize};

/// Selects history operations for an execution. Workers must supply the requested services.
/// Defaults to runtime-owned history without external maintenance. Unkeyed
/// conversations are released at frame exit; keyed histories grow until dropped or compacted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryPolicy {
    /// Persist accepted original conversation entries during agent execution and completion.
    /// Frame-exit cleanup and explicit conversation dropping never wait for persistence.
    pub persist: bool,
    /// Load absent keyed conversations from the configured store during activation.
    pub load: bool,
    /// Consult the worker's compactor before generation, protecting the current exchange.
    pub compact: bool,
}
