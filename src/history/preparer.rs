use async_trait::async_trait;

use crate::Context;
use crate::clients::{ClientOptions, Message};
use crate::graph::GraphError;

use super::HistoryEntry;

/// Borrowed context for one upcoming model request, before attachment materialization.
/// Provider wire transformations and exact token counts are not represented here.
pub struct HistoryPreparation<'a> {
    pub(crate) session_id: &'a str,
    pub(crate) model: &'a str,
    pub(crate) options: &'a ClientOptions,
    pub(crate) framework_messages: &'a [Message],
    pub(crate) committed: &'a [&'a HistoryEntry],
    pub(crate) protected: &'a [&'a HistoryEntry],
}

impl HistoryPreparation<'_> {
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
#[derive(Debug, Default)]
pub struct HistoryReplacement {
    /// Zero-based indices of committed entries; must be exactly `0..n`.
    pub evict_indices: Vec<usize>,
    /// Non-empty memory text, rendered as a tagged system message before retained history.
    pub summary: Option<String>,
}

/// Fallible application policy invoked once before each model execution attempt.
/// Errors prevent execution and leave history unchanged. External work should be idempotent.
/// The supplied context belongs to the execution, including fresh dependencies after restore.
pub trait HistoryPreparer: Send + Sync {
    /// Application error preserved as the source of `GraphError::HistoryPreparation`.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Chooses a safe replacement for completed history using the upcoming request context.
    /// `ctx` provides runtime dependencies; external writes are not atomic with history replacement.
    fn prepare(
        &self,
        request: HistoryPreparation<'_>,
        ctx: Context,
    ) -> impl std::future::Future<Output = Result<HistoryReplacement, Self::Error>> + Send;
}

#[async_trait]
pub(crate) trait DynHistoryPreparer: Send + Sync {
    async fn prepare_dyn(
        &self,
        request: HistoryPreparation<'_>,
        ctx: Context,
    ) -> Result<HistoryReplacement, GraphError>;
}

#[async_trait]
impl<T: HistoryPreparer> DynHistoryPreparer for T {
    /// Forwards request dependencies and retains the application's error as its source.
    async fn prepare_dyn(
        &self,
        request: HistoryPreparation<'_>,
        ctx: Context,
    ) -> Result<HistoryReplacement, GraphError> {
        let session_id = request.session_id;
        self.prepare(request, ctx)
            .await
            .map_err(|source| GraphError::HistoryPreparation {
                session_id: session_id.to_owned(),
                source: Box::new(source),
            })
    }
}
