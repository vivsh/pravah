//! Caller-owned asynchronous maintenance of runtime-owned history.

use std::{collections::BTreeMap, sync::Arc};
use uuid::Uuid;

use super::{
    CompactionRequest, Compactor, DynCompactor, DynHistoryStore, HistoryStore, protected_start,
};
use crate::{
    Context,
    graph::{GraphError, Runtime},
};

/// Persists original messages before optionally replacing completed working history.
///
/// This runtime-only owner stores acknowledgements, not messages or VM state. Reinstall
/// it after restoration; stores must deduplicate redelivery by entry identity. Coordinate
/// concurrent writers in the host. There are no automatic retries or background tasks.
#[derive(Default)]
pub struct HistoryManager {
    store: Option<Arc<dyn DynHistoryStore>>,
    compactor: Option<Arc<dyn DynCompactor>>,
    persisted: BTreeMap<Uuid, u64>,
}

impl HistoryManager {
    /// Creates an inert manager; neither persistence nor compaction is enabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the store and forgets previous acknowledgements for that store.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.store = Some(Arc::new(store));
        self.persisted.clear();
        self
    }

    /// Replaces the application compaction policy; no history is retained here.
    pub fn with_compactor(mut self, compactor: impl Compactor + 'static) -> Self {
        self.compactor = Some(Arc::new(compactor));
        self
    }

    /// Checks application policy at an upcoming agent dispatch without allocating a request.
    /// This does not report pending persistence or count exact tokens.
    pub fn needs_compaction(&self, runtime: &Runtime) -> bool {
        match (
            self.compactor.as_deref(),
            runtime.history_dispatch_session(),
        ) {
            (Some(policy), Some(session)) => {
                policy.needs_compaction_dyn(runtime.history(), session)
            }
            _ => false,
        }
    }

    /// Flushes new rows, then prepares working memory before the next model dispatch.
    ///
    /// Call before stepping and after the final step. Failed persistence prevents pruning;
    /// successful partial writes remain acknowledged. A policy/validation failure leaves
    /// history and VM state unchanged. Retry maintenance, not a completed model operation.
    /// Repeated explicit calls at the same dispatch may invoke the policy again.
    pub async fn maintain(
        &mut self,
        runtime: &mut Runtime,
        ctx: Context,
    ) -> Result<(), GraphError> {
        self.persist(runtime).await?;
        if !self.needs_compaction(runtime) {
            return Ok(());
        }
        self.compact(runtime, ctx).await
    }

    /// Builds a borrowed policy view from the next agent dispatch and commits only a valid decision.
    async fn compact(&self, runtime: &mut Runtime, ctx: Context) -> Result<(), GraphError> {
        let Some((session, request, mut guidance, budget_conclusion)) =
            runtime.history_dispatch_request()?
        else {
            return Ok(());
        };
        let client = preparation_client(&request, &ctx).await?;
        if budget_conclusion {
            guidance.push(crate::clients::Message::user(
                crate::clients::conclusion_message(&client.provider()),
            ));
        }
        let entries = runtime.history().session_entries(&session);
        let (committed, protected) = entries.split_at(protected_start(&entries));
        let Some(policy) = self.compactor.as_deref() else {
            return Ok(());
        };
        let decision = policy
            .compact_dyn(
                CompactionRequest {
                    session_id: &session,
                    model: request.model(),
                    options: client.options(),
                    framework_messages: &guidance,
                    committed,
                    protected,
                },
                ctx,
            )
            .await?;
        runtime.compact_history(&session, decision)
    }

    /// Saves each original row in order and advances only the manager's acknowledgement.
    async fn persist(&mut self, runtime: &Runtime) -> Result<(), GraphError> {
        let Some(store) = self.store.as_deref() else {
            return Ok(());
        };
        let execution = runtime.state().execution_id();
        let cursor = self.persisted.get(&execution).copied().unwrap_or(0);
        let entries = runtime.history().entries();
        let start = entries.partition_point(|entry| entry.position < cursor);
        for entry in entries.get(start..).unwrap_or_default() {
            if entry.agent_id == super::inspection::SUMMARY_AGENT_ID {
                continue;
            }
            let next = entry.position.checked_add(1).ok_or_else(|| {
                GraphError::HistoryPersistence("history positions exhausted".into())
            })?;
            store
                .record_dyn(entry)
                .await
                .map_err(|error| GraphError::HistoryPersistence(error.to_string()))?;
            self.persisted.insert(execution, next);
        }
        Ok(())
    }
}

/// Observes provider-effective options using operation-local construction dependencies.
async fn preparation_client(
    request: &crate::graph::fetch::rath::RathRequest,
    ctx: &Context,
) -> Result<crate::clients::Client, GraphError> {
    ctx.providers()
        .llm(request.model(), request.options().clone())
        .await
        .map_err(|source| GraphError::AgentClient {
            operation: crate::graph::AgentClientOperation::Create,
            source,
        })
}
