//! Stateless worker execution: dependencies are shared, completion state belongs to the VM.

use super::{AgentError, AgentOperation, AgentRequest, AgentResponse};
use crate::graph::{AgentClientOperation, GraphError, HandlerRegistry, Value};
use crate::history::{DynCompactor, DynHistoryStore};
use crate::{Compactor, Context, HistoryStore};
use futures::future::BoxFuture;
use std::sync::Arc;

/// An agent or tool callback executed outside the synchronous VM.
pub trait DynAgentHandler: Send + Sync {
    /// Executes one callback; it owns no history or pending-operation state.
    fn execute<'a>(
        &'a self,
        request: &'a AgentRequest,
        context: Context,
    ) -> BoxFuture<'a, Result<Value, GraphError>>;
}

/// Reusable agent worker dependencies, independent of any runtime or graph lifetime.
pub struct AgentExecutor {
    context: Context,
    registry: Arc<HandlerRegistry>,
    pub(crate) store: Option<Arc<dyn DynHistoryStore>>,
    pub(crate) compactor: Option<Arc<dyn DynCompactor>>,
}

impl AgentExecutor {
    /// Creates a graph-independent worker; callbacks require a matching registry.
    pub fn new(context: Context) -> Self {
        Self::from_registry(context, Arc::new(HandlerRegistry::new()))
    }
    pub(crate) fn from_registry(context: Context, registry: Arc<HandlerRegistry>) -> Self {
        Self {
            context,
            registry,
            store: None,
            compactor: None,
        }
    }
    /// Installs immutable compatible callbacks without retaining a graph or runtime.
    pub fn with_registry(mut self, registry: Arc<HandlerRegistry>) -> Self {
        self.registry = registry;
        self
    }
    /// Installs a conversation store. Restore workers must reinstall dependencies.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.store = Some(Arc::new(store));
        self
    }
    /// Installs a request-aware compactor; runtime policy determines whether it runs.
    pub fn with_compactor(mut self, compactor: impl Compactor + 'static) -> Self {
        self.compactor = Some(Arc::new(compactor));
        self
    }
    /// Borrows runtime dependencies without exposing an execution owner.
    pub fn context(&self) -> &Context {
        &self.context
    }
    /// Borrows compatible immutable handlers.
    pub fn registry(&self) -> &HandlerRegistry {
        &self.registry
    }

    /// Completes one operation without retry or fallback. Later failures retain earlier acknowledgements.
    pub async fn execute(&self, request: &AgentRequest) -> AgentResponse {
        let mut response = AgentResponse::new(request.id(), Ok(Value::NULL));
        let result = self.execute_stages(request, &mut response).await;
        response.outcome = result.map_err(|error| {
            AgentError::from_execution_error(&error).unwrap_or_else(|_| {
                AgentError::new("diagnostics", "could not encode failure diagnostics")
            })
        });
        response
    }

    /// Persists accepted rows before any preparation can replace them.
    async fn execute_stages(
        &self,
        request: &AgentRequest,
        response: &mut AgentResponse,
    ) -> Result<Value, GraphError> {
        self.persist(request.persist.as_deref().unwrap_or_default(), response)
            .await?;
        match request.operation.as_ref() {
            AgentOperation::Generate { .. } => self.generate(request, response).await,
            AgentOperation::PersistHistory => Ok(Value::NULL),
            _ => self.callback(request, response).await,
        }
    }

    /// Delivers callbacks through immutable handlers, then loads a newly configured keyed conversation.
    async fn callback(
        &self,
        request: &AgentRequest,
        response: &mut AgentResponse,
    ) -> Result<Value, GraphError> {
        let key = request.handler().ok_or_else(|| {
            GraphError::AgentRequestValidation("missing callback identity".into())
        })?;
        let handler = self
            .registry
            .agent(key)
            .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
        let value = handler.execute(request, self.context.clone()).await?;
        if let AgentOperation::Configure {
            load_history: true,
            loaded_keys,
            ..
        } = request.operation.as_ref()
            && let Some(key) = value.get("key").and_then(Value::as_str)
            && !loaded_keys.contains(key)
        {
            response.loaded = Some((key.to_owned(), self.load(key).await?));
        }
        Ok(value)
    }

    /// Advances the returned durable cursor only after each individual store acknowledgement.
    async fn persist(
        &self,
        entries: &[crate::HistoryEntry],
        response: &mut AgentResponse,
    ) -> Result<(), GraphError> {
        if entries.is_empty() {
            return Ok(());
        }
        let store = self.store.as_deref().ok_or_else(|| {
            GraphError::HistoryPersistence(
                "history persistence is enabled but no store is installed".into(),
            )
        })?;
        for entry in entries {
            let next = entry.position.checked_add(1).ok_or_else(|| {
                GraphError::HistoryPersistence("history positions exhausted".into())
            })?;
            store
                .record_dyn(entry)
                .await
                .map_err(|error| GraphError::HistoryPersistence(error.to_string()))?;
            response.persisted_through = Some(next);
        }
        Ok(())
    }

    /// Retrieves original completed context; the VM validates and assigns local append positions.
    async fn load(&self, key: &str) -> Result<Vec<crate::HistoryEntry>, GraphError> {
        let store = self.store.as_deref().ok_or_else(|| {
            GraphError::HistoryPersistence(
                "history loading is enabled but no store is installed".into(),
            )
        })?;
        store
            .load_dyn(key)
            .await
            .map_err(|error| GraphError::HistoryPersistence(error.to_string()))
    }

    /// Performs request preparation and one ordinary Rath generation without an intermediate request.
    async fn generate(
        &self,
        request: &AgentRequest,
        completion: &mut AgentResponse,
    ) -> Result<Value, GraphError> {
        let AgentOperation::Generate { model, options, .. } = request.operation.as_ref() else {
            return Err(GraphError::AgentRequestValidation(
                "expected generation".into(),
            ));
        };
        let options = super::options::deserialize(options)
            .map_err(|error| GraphError::AgentRequestValidation(error.to_string()))?;
        let messages = self
            .prepare_messages(request.operation.as_ref(), &options, completion)
            .await?;
        let client = self.client(model, options).await?;
        let response = client.execute(&messages).await.map_err(|source| {
            let source = if source.provider().is_none() {
                source.with_context(client.provider(), "generation")
            } else {
                source
            };
            GraphError::AgentClient {
                operation: AgentClientOperation::Execute,
                source,
            }
        })?;
        let response = super::client_response::serialize_value(&response)?;
        Ok(response)
    }

    /// Constructs through the authoritative Rath registry without fallback or retained clients.
    async fn client(
        &self,
        model: &str,
        options: crate::clients::ClientOptions,
    ) -> Result<crate::clients::Client, GraphError> {
        self.context
            .providers()
            .llm(model, options)
            .await
            .map_err(|source| GraphError::AgentClient {
                operation: AgentClientOperation::Create,
                source,
            })
    }
}

impl AgentExecutor {
    /// Applies a validated operation-local candidate without retaining client or history state.
    async fn prepare_messages(
        &self,
        operation: &AgentOperation,
        options: &crate::clients::ClientOptions,
        completion: &mut AgentResponse,
    ) -> Result<Vec<crate::clients::Message>, GraphError> {
        let AgentOperation::Generate {
            session_id,
            entries,
            guidance,
            budget_conclusion,
            compact,
            ..
        } = operation
        else {
            return Err(GraphError::AgentRequestValidation(
                "expected generation".into(),
            ));
        };
        let mut borrowed = entries.iter().collect::<Vec<_>>();
        let mut guidance = guidance.clone();
        let policy = self.compaction_policy(*compact)?;
        let view = compaction_view(operation, options, &guidance, &borrowed)?;
        let policy = policy.filter(|policy| policy.needs_compaction_dyn(&view));
        let replacement = if policy.is_some() || *budget_conclusion {
            self.prepare_candidate(
                operation,
                options,
                &mut guidance,
                &borrowed,
                policy,
                completion,
            )
            .await?
        } else {
            None
        };
        if let Some(replacement) = &replacement {
            replacement.preview(session_id, &mut borrowed);
        }
        self.materialize_history(&borrowed, guidance).await
    }

    /// Obtains effective options only when a compactor or provider conclusion reminder needs them.
    async fn prepare_candidate(
        &self,
        operation: &AgentOperation,
        options: &crate::clients::ClientOptions,
        guidance: &mut Vec<crate::clients::Message>,
        entries: &[&crate::HistoryEntry],
        policy: Option<&dyn DynCompactor>,
        completion: &mut AgentResponse,
    ) -> Result<Option<crate::history::ValidatedCompactionResult>, GraphError> {
        let AgentOperation::Generate {
            model,
            session_id,
            budget_conclusion,
            ..
        } = operation
        else {
            return Err(GraphError::AgentRequestValidation(
                "expected generation".into(),
            ));
        };
        let client = self.client(model, options.clone()).await?;
        if *budget_conclusion {
            guidance.push(crate::clients::Message::user(
                crate::clients::conclusion_message(&client.provider()),
            ));
        }
        let Some(policy) = policy else {
            return Ok(None);
        };
        let view = compaction_view(operation, client.options(), guidance, entries)?;
        let decision = policy.compact_dyn(view, self.context.clone()).await?;
        let replacement = crate::history::prepare_compaction(
            session_id,
            entries,
            decision.clone(),
            Some(completion.id()),
        )
        .map_err(|reason| GraphError::HistoryCompactionValidation {
            session_id: session_id.clone(),
            reason,
        })?;
        completion.compaction = Some((session_id.clone(), decision));
        Ok(Some(replacement))
    }

    /// Enforced execution intent cannot silently become a no-op when a worker lacks its service.
    fn compaction_policy(&self, enabled: bool) -> Result<Option<&dyn DynCompactor>, GraphError> {
        if !enabled {
            return Ok(None);
        }
        self.compactor.as_deref().map(Some).ok_or_else(|| {
            GraphError::HistoryPersistence(
                "compaction is enabled but no compactor is installed".into(),
            )
        })
    }

    /// Renders the selected candidate once; attachments are resolved outside the VM.
    async fn materialize_history(
        &self,
        entries: &[&crate::HistoryEntry],
        guidance: Vec<crate::clients::Message>,
    ) -> Result<Vec<crate::clients::Message>, GraphError> {
        let mut messages = entries
            .iter()
            .map(|entry| entry.message.clone())
            .collect::<Vec<_>>();
        messages.extend(guidance);
        crate::clients::materialize_owned_messages(messages, &self.context)
            .await
            .map_err(|error| {
                GraphError::AgentRequestValidation(format!(
                    "message materialization failed: {error}"
                ))
            })
    }
}

/// Borrows generation inputs and separates completed exchanges from the protected current exchange.
fn compaction_view<'a>(
    operation: &'a AgentOperation,
    options: &'a crate::clients::ClientOptions,
    guidance: &'a [crate::clients::Message],
    entries: &'a [&'a crate::HistoryEntry],
) -> Result<crate::CompactionRequest<'a>, GraphError> {
    let AgentOperation::Generate {
        session_id,
        model,
        last_usage,
        total_input,
        total_output,
        ..
    } = operation
    else {
        return Err(GraphError::AgentRequestValidation(
            "expected generation".into(),
        ));
    };
    let (committed, protected) = entries.split_at(crate::history::protected_start(entries));
    Ok(crate::CompactionRequest {
        session_id,
        model,
        options,
        framework_messages: guidance,
        committed,
        protected,
        last_usage: *last_usage,
        total_input: *total_input,
        total_output: *total_output,
    })
}
