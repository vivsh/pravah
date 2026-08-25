//! Runs evidence ingestion, reconciliation, and retrieval with small deterministic providers.

use async_trait::async_trait;
#[cfg(feature = "recall-postgres")]
use pravah_memory::context::AssembledContext;
use pravah_memory::context::{ContextAssembler, ContextError, ContextOptions};
#[cfg(feature = "recall-postgres")]
use pravah_memory::postgres::{RecallStore, RecallStoreError};
use pravah_memory::{
    Embedding, EmbeddingProfile, EmbeddingProvider, ExtractedMemory, ExtractionRequest,
    MemoryExtractor, MemoryKind, MemoryManager, MemoryManagerError, MemoryReconciler,
    ProviderError, ReconciliationDecision, ReconciliationGroup, TemporalMetadata,
};
#[cfg(feature = "recall-postgres")]
use pravah_memory::{RecallBatch, RecallError, TrackedSearch};
use thiserror::Error;

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    Database(#[from] mool::DbError),
    #[error(transparent)]
    Memory(#[from] MemoryManagerError),
    #[error(transparent)]
    Context(#[from] ContextError),
    #[cfg(feature = "recall-postgres")]
    #[error(transparent)]
    Recall(#[from] RecallError),
    #[cfg(feature = "recall-postgres")]
    #[error(transparent)]
    RecallStore(#[from] RecallStoreError),
    #[cfg(feature = "recall-postgres")]
    #[error(transparent)]
    RecallFlush(#[from] tokio::task::JoinError),
}

struct Extractor;

#[async_trait]
impl MemoryExtractor for Extractor {
    fn revision(&self) -> &str {
        "example-extractor-v1"
    }

    async fn extract(
        &self,
        request: ExtractionRequest,
    ) -> Result<Vec<ExtractedMemory>, ProviderError> {
        Ok(vec![ExtractedMemory {
            text: request.evidence,
            entities: Some(Vec::new()),
            kind: MemoryKind::Fact,
            temporal: TemporalMetadata::default(),
            metadata: serde_json::json!({}),
        }])
    }
}

struct Embedder;

#[async_trait]
impl EmbeddingProvider for Embedder {
    fn profile(&self) -> EmbeddingProfile {
        EmbeddingProfile {
            model: "example-embedding".to_owned(),
            revision: "v1".to_owned(),
            dimensions: 3,
            document_format_revision: "claim-text-v1".to_owned(),
        }
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, ProviderError> {
        texts
            .iter()
            .map(|text| {
                let length = text.len() as f32;
                Embedding::new(vec![length, 1.0, 0.5]).map_err(|error| {
                    ProviderError::new("invalid_embedding", error.to_string(), false)
                })
            })
            .collect()
    }
}

struct Reconciler;

#[async_trait]
impl MemoryReconciler for Reconciler {
    fn revision(&self) -> &str {
        "example-reconciler-v1"
    }

    async fn reconcile(
        &self,
        _group: ReconciliationGroup,
    ) -> Result<Vec<ReconciliationDecision>, ProviderError> {
        Ok(Vec::new())
    }
}

#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    let conf = mool::DbConf::from_env()?;
    let pool = mool::DbPool::from_conf(&conf).await?;
    let manager = MemoryManager::builder(pool.clone())
        .memory_extractor(Extractor)
        .embedding_provider(Embedder)
        .reconciler(Reconciler)
        .build()
        .await?;

    let ingestor = manager.ingestor("user-42", "companion")?;
    ingestor
        .submit("profile:42:v1", "The user prefers aisle seats.")
        .await?;
    manager
        .reconciler("user-42", "companion")?
        .reconcile_pending(16)
        .await?;
    let tracked = manager
        .retriever("user-42", "companion")?
        .search_tracked("What seating does the user prefer?")
        .await?;
    let context =
        ContextAssembler::compact().assemble(&tracked.results, ContextOptions::default())?;

    #[cfg(feature = "recall-postgres")]
    let recall_flush = queue_recall_flush(pool, &tracked, &context)?;

    for result in &tracked.results {
        println!("{:.4}: {}", result.score, result.memory.text);
    }
    println!("\n{}", context.rendered);

    #[cfg(feature = "recall-postgres")]
    recall_flush.await??;
    Ok(())
}

/// Starts an application-owned flush without adding a write to retrieval latency.
#[cfg(feature = "recall-postgres")]
fn queue_recall_flush(
    pool: mool::DbPool,
    tracked: &TrackedSearch,
    context: &AssembledContext,
) -> Result<tokio::task::JoinHandle<Result<u64, RecallStoreError>>, RecallError> {
    let batch = RecallBatch::used(
        &tracked.receipt,
        context.selected_memory_ids.iter().copied(),
    )?;
    let user_key = tracked.receipt.user_key.clone();
    let agent_key = tracked.receipt.agent_key.clone();
    Ok(tokio::spawn(async move {
        let store = RecallStore::builder(pool).build().await?;
        store
            .recorder(user_key, agent_key)?
            .record_many(&[batch])
            .await
    }))
}
