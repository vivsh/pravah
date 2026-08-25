use chrono::Utc;

use super::{MemoryManagerError, MemoryRetriever};
use crate::{RecallCandidate, RecallReceipt, SearchRequest, SearchResult, TrackedSearch};

impl MemoryRetriever {
    /// Runs default hybrid retrieval and returns an optional-reporting receipt.
    pub async fn search_tracked(
        &self,
        text: impl Into<String>,
    ) -> Result<TrackedSearch, MemoryManagerError> {
        self.search_tracked_with(SearchRequest::new(text)?).await
    }

    /// Runs configurable retrieval and returns the unchanged results plus a receipt.
    pub async fn search_tracked_with(
        &self,
        request: SearchRequest,
    ) -> Result<TrackedSearch, MemoryManagerError> {
        let results = self.search_with(request).await?;
        tracked_search(&self.user_key, &self.agent_key, results)
    }
}

/// Projects final result order into a validated, one-based recall receipt.
fn tracked_search(
    user_key: &str,
    agent_key: &str,
    results: Vec<SearchResult>,
) -> Result<TrackedSearch, MemoryManagerError> {
    let candidates = results
        .iter()
        .enumerate()
        .map(|(position, result)| {
            let rank = u32::try_from(position + 1).map_err(|_| {
                MemoryManagerError::InvalidInput("tracked search result count exceeds u32".into())
            })?;
            Ok(RecallCandidate {
                memory_id: result.memory.id,
                rank,
            })
        })
        .collect::<Result<Vec<_>, MemoryManagerError>>()?;
    let receipt = RecallReceipt::at(user_key, agent_key, candidates, Utc::now())?;
    Ok(TrackedSearch { receipt, results })
}
