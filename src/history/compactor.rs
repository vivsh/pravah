use async_trait::async_trait;

use crate::Context;
use crate::clients::{ClientOptions, Message};
use crate::graph::GraphError;

use super::HistoryEntry;

/// Borrowed context for one upcoming model request, before attachment materialization.
/// Provider wire transformations and exact token counts are not represented here.
pub struct CompactionRequest<'a> {
    pub(crate) session_id: &'a str,
    pub(crate) model: &'a str,
    pub(crate) options: &'a ClientOptions,
    pub(crate) framework_messages: &'a [Message],
    pub(crate) committed: &'a [&'a HistoryEntry],
    pub(crate) protected: &'a [&'a HistoryEntry],
}

impl CompactionRequest<'_> {
    /// Borrows the first committed entry's summary text without its framework wrapper.
    /// Preserves whitespace; returns None when absent or malformed, without modifying history.
    pub fn summary(&self) -> Option<&str> {
        self.committed
            .first()
            .and_then(|entry| super::inspection::summary_text(entry))
    }

    /// Borrows committed messages excluding framework summaries and tool calls/results.
    /// Omits the newest `skip_recent` exposed messages; indices retain their positions in `committed()`.
    /// Protected input is never included; enumeration alone does not select a safe eviction prefix.
    pub fn enum_messages(&self, skip_recent: usize) -> impl Iterator<Item = (usize, &Message)> {
        super::inspection::enum_messages(self.committed.iter().copied(), skip_recent)
    }

    /// Returns the compact JSON array size of all committed messages, including tools and attachments.
    /// Excludes protected input, framework guidance and entry metadata; not a token or provider-size estimate.
    /// Serialization or size overflow can fail; no encoded buffer is retained.
    pub fn byte_size(&self) -> Result<usize, serde_json::Error> {
        super::inspection::byte_size(self.committed.iter().map(|entry| &entry.message))
    }

    /// Counts committed user exchanges ending in an assistant reply, not intermediate tool rounds.
    pub fn turn_count(&self) -> usize {
        super::inspection::turn_count(self.committed.iter().map(|entry| &entry.message))
    }

    /// Returns the history session being prepared.
    pub fn session_id(&self) -> &str {
        self.session_id
    }

    /// Returns the invocation's resolved model URL.
    pub fn model(&self) -> &str {
        self.model
    }

    /// Borrows effective client options, including preamble, schemas and active tools.
    pub fn options(&self) -> &ClientOptions {
        self.options
    }

    /// Returns additional guidance in the order it will follow conversation history.
    pub fn framework_messages(&self) -> &[Message] {
        self.framework_messages
    }

    /// Returns prior completed exchanges eligible for prefix replacement.
    pub fn committed(&self) -> &[&HistoryEntry] {
        self.committed
    }

    /// Returns the current user input and its ongoing exchange, which cannot be changed.
    pub fn protected(&self) -> &[&HistoryEntry] {
        self.protected
    }
}

/// Replaces a sorted, contiguous prefix of `committed()` with optional plain text memory.
/// An empty decision retains history. A summary requires at least one replaced entry.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompactionResult {
    /// Zero-based indices of committed entries; must be exactly `0..n`.
    pub evict_indices: Vec<usize>,
    /// Non-empty memory text, rendered as a tagged system message before retained history.
    pub summary: Option<String>,
}

/// Fallible application policy invoked once before each model execution attempt.
/// Errors prevent execution and leave history unchanged. External work should be idempotent.
/// The supplied context belongs to the execution, including fresh dependencies after restore.
pub trait Compactor: Send + Sync {
    /// Application error preserved as the source of `GraphError::HistoryCompaction`.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Chooses a safe replacement for completed history using the upcoming request context.
    /// `ctx` provides runtime dependencies; external writes are not atomic with history replacement.
    fn compact(
        &self,
        request: CompactionRequest<'_>,
        ctx: Context,
    ) -> impl std::future::Future<Output = Result<CompactionResult, Self::Error>> + Send;
}

#[async_trait]
pub(crate) trait DynCompactor: Send + Sync {
    async fn compact_dyn(
        &self,
        request: CompactionRequest<'_>,
        ctx: Context,
    ) -> Result<CompactionResult, GraphError>;
}

#[async_trait]
impl<T: Compactor> DynCompactor for T {
    /// Forwards request dependencies and retains the application's error as its source.
    async fn compact_dyn(
        &self,
        request: CompactionRequest<'_>,
        ctx: Context,
    ) -> Result<CompactionResult, GraphError> {
        let session_id = request.session_id;
        self.compact(request, ctx)
            .await
            .map_err(|source| GraphError::HistoryCompaction {
                session_id: session_id.to_owned(),
                source: Box::new(source),
            })
    }
}
