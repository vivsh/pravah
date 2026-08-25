use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;

use futures::future::BoxFuture;
use serde_json::Value;

use crate::commons::Agent;
use crate::context::Context;

/// Convenience alias for the return type of [`MemoryFactory::retrieve`].
///
/// `Ok(Some(string))` — inject this string into the system prompt.
/// `Ok(None)` — no context to inject; proceed without it.
/// `Err(e)` — retrieval failed; the flow will halt with [`FlowError::MemoryError`](crate::legacy::FlowError).
pub type MemoryResult = Result<Option<String>, Box<dyn std::error::Error + Send + Sync>>;

/// Query passed to [`MemoryFactory::retrieve`].
///
/// Carries all per-invocation context needed to retrieve relevant memories or
/// dynamic context for an agent's system prompt.
pub struct MemoryQuery<'a> {
    /// Registered node id of the agent being dispatched.
    ///
    /// Use this to route different retrieval strategies per agent.
    pub agent_name: &'a str,
    /// Raw agent input as a JSON value.
    ///
    /// Extract query text, user id, or any other field your schema exposes:
    /// `query.input["user_id"].as_str()`.
    pub input: &'a Value,
    /// Shared execution context.
    ///
    /// Provides an HTTP client (`ctx.http_client()`), working directory, and
    /// registered external dependencies — enough to call any retrieval service.
    pub ctx: &'a Context,
}

/// Injects dynamic context into an agent's system prompt before each invocation.
///
/// Implement this trait to retrieve memories, RAG results, or any other
/// per-request context from an external source. The returned string is inserted
/// between the static preamble and the input-schema hint in the system prompt.
///
/// The result is cached in [`AgentState`](crate::legacy::FlowRuntime) for the
/// lifetime of a single agent invocation, so `retrieve` is called at most once
/// per agent dispatch regardless of how many LLM turns occur.
///
/// # Example
///
/// ```rust,no_run
/// use pravah::legacy::memory::{MemoryFactory, MemoryQuery, MemoryResult};
///
/// struct MyRetriever;
///
/// impl MemoryFactory for MyRetriever {
///     async fn retrieve(&self, query: &MemoryQuery<'_>) -> MemoryResult {
///         // Call your vector DB, REST API, local cache, etc.
///         let text = query.input["query"].as_str().ok_or("missing query field")?;
///         Ok(Some(format!("Relevant context: {text}")))
///     }
/// }
/// ```
pub trait MemoryFactory {
    fn retrieve(&self, query: &MemoryQuery<'_>) -> impl Future<Output = MemoryResult> + Send;
}

/// A [`MemoryFactory`] that always returns `None`. Default for [`FlowRuntime`](crate::legacy::FlowRuntime).
pub struct NoopMemoryFactory;

impl MemoryFactory for NoopMemoryFactory {
    async fn retrieve(&self, _query: &MemoryQuery<'_>) -> MemoryResult {
        Ok(None)
    }
}

/// Object-safe version of [`MemoryFactory`] used internally for type erasure.
pub(crate) trait DynMemoryFactory: Send + Sync + 'static {
    fn retrieve_dyn<'a>(&'a self, query: &'a MemoryQuery<'a>) -> BoxFuture<'a, MemoryResult>;
}

impl<T> DynMemoryFactory for T
where
    T: MemoryFactory + Send + Sync + 'static,
{
    fn retrieve_dyn<'a>(&'a self, query: &'a MemoryQuery<'a>) -> BoxFuture<'a, MemoryResult> {
        Box::pin(self.retrieve(query))
    }
}

/// Typed per-agent memory retrieval.
///
/// Implement this for a specific agent type `A` to receive the deserialized
/// input struct rather than raw JSON. Register with [`MemoryRegistry::for_agent`].
pub trait AgentMemory<A: Agent> {
    /// Retrieve context for one invocation of agent `A`.
    ///
    /// `input` is the deserialized agent input. Return `Ok(None)` when nothing
    /// is relevant; return `Err(...)` to halt the flow with
    /// [`FlowError::MemoryError`](crate::legacy::FlowError).
    fn retrieve(&self, input: &A, ctx: &Context) -> impl Future<Output = MemoryResult> + Send;
}

struct AgentMemoryShim<A, F> {
    factory: F,
    _marker: PhantomData<fn() -> A>,
}

impl<A, F> MemoryFactory for AgentMemoryShim<A, F>
where
    A: Agent,
    F: AgentMemory<A> + Send + Sync + 'static,
{
    async fn retrieve(&self, query: &MemoryQuery<'_>) -> MemoryResult {
        let input: A = match serde_json::from_value(query.input.clone()) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        self.factory.retrieve(&input, query.ctx).await
    }
}

/// Dispatch table that routes memory retrieval to per-agent stores.
///
/// Build with [`MemoryRegistry::new`], register typed stores with
/// [`for_agent`](MemoryRegistry::for_agent), and optionally set a
/// [`with_fallback`](MemoryRegistry::with_fallback) for agents without a
/// dedicated entry.
///
/// `MemoryRegistry` implements [`MemoryFactory`], so pass it directly to
/// [`FlowRuntime::with_memory`](crate::legacy::FlowRuntime::with_memory).
///
/// # Example
///
/// ```rust,no_run
/// use pravah::legacy::memory::{AgentMemory, MemoryRegistry, MemoryResult};
/// use pravah::context::Context;
/// # use schemars::JsonSchema;
/// # use serde::{Deserialize, Serialize};
/// # use pravah::legacy::{Agent, AgentConfig};
/// # #[derive(Serialize, Deserialize, JsonSchema)]
/// # struct Search { query: String }
/// # impl Agent for Search {
/// #     type Output = Search;
/// #     fn configure() -> AgentConfig { AgentConfig::new("", "openai:///gpt-4o") }
/// # }
/// # struct VectorDB;
/// impl AgentMemory<Search> for VectorDB {
///     async fn retrieve(&self, input: &Search, _ctx: &Context) -> MemoryResult {
///         Ok(Some(format!("results for: {}", input.query)))
///     }
/// }
///
/// let registry = MemoryRegistry::new()
///     .for_agent::<Search, _>(VectorDB);
/// ```
pub struct MemoryRegistry {
    entries: HashMap<String, Box<dyn DynMemoryFactory>>,
    fallback: Option<Box<dyn DynMemoryFactory>>,
}

impl Default for MemoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryRegistry {
    /// Creates an empty registry with no registered agents and no fallback.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            fallback: None,
        }
    }

    /// Registers a typed memory store for agent `A`.
    ///
    /// When agent `A` is dispatched, `factory.retrieve(&input, ctx)` is called
    /// with the deserialized input struct. Overwrites any previously registered
    /// factory for the same agent.
    pub fn for_agent<A, F>(mut self, factory: F) -> Self
    where
        A: Agent,
        F: AgentMemory<A> + Send + Sync + 'static,
    {
        let shim = AgentMemoryShim {
            factory,
            _marker: PhantomData,
        };
        self.entries.insert(A::node_id(), Box::new(shim));
        self
    }

    /// Sets a fallback factory for agents without a registered entry.
    ///
    /// The fallback receives the untyped [`MemoryQuery`] (including
    /// `agent_name`) so it can apply its own routing logic if needed.
    pub fn with_fallback<F>(mut self, fallback: F) -> Self
    where
        F: MemoryFactory + Send + Sync + 'static,
    {
        self.fallback = Some(Box::new(fallback));
        self
    }
}

impl MemoryFactory for MemoryRegistry {
    async fn retrieve(&self, query: &MemoryQuery<'_>) -> MemoryResult {
        if let Some(entry) = self.entries.get(query.agent_name) {
            return entry.retrieve_dyn(query).await;
        }
        if let Some(fallback) = &self.fallback {
            return fallback.retrieve_dyn(query).await;
        }
        Ok(None)
    }
}
