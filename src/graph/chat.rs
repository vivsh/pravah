use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};

use crate::Context;
use crate::legacy::{HistoryCompactor, HistoryStore};

use super::agent::Agent;
use super::error::GraphError;
use super::registry::RuntimeServices;
use super::runtime::{Runtime, Snapshot};
use super::state::Step;
use super::typed::{CompiledFlow, Flow};
use super::value::{from_value, to_value};

/// One assistant response produced by graph chat.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatTurn<O> {
    /// Decoded assistant response.
    pub output: O,
}

impl<O> ChatTurn<O> {
    /// Consumes the turn and returns the response value.
    pub fn into_output(self) -> O {
        self.output
    }
}

/// Runtime-backed chat loop built from one function-defined graph agent.
///
/// The agent configuration should enable `keep_alive` when later turns must
/// retain the same model-visible conversation session.
pub struct Chat<I, O> {
    agent: fn(Agent<I>) -> Agent<O>,
    runtime: Option<Runtime>,
    services: RuntimeServices,
    _marker: PhantomData<fn(I) -> O>,
}

impl<I, O> Chat<I, O>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Creates a chat session from a function-defined agent.
    pub fn new(agent: fn(Agent<I>) -> Agent<O>) -> Self {
        Self {
            agent,
            runtime: None,
            services: RuntimeServices::new(),
            _marker: PhantomData,
        }
    }

    /// Restores a chat using the same agent definition as the saved snapshot.
    pub fn from_snapshot(
        agent: fn(Agent<I>) -> Agent<O>,
        snapshot: Snapshot,
    ) -> Result<Self, GraphError> {
        let flow = build_chat_flow(agent)?;
        let runtime = flow.prepared().restore(snapshot)?;
        Ok(Self {
            agent,
            runtime: Some(runtime),
            services: RuntimeServices::new(),
            _marker: PhantomData,
        })
    }

    /// Replaces the history compactor used by the chat runtime.
    pub fn with_compactor(mut self, compactor: impl HistoryCompactor + 'static) -> Self {
        self.services = self.services.with_compactor(compactor);
        self
    }

    /// Replaces the history store used to record chat messages.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        self.services = self.services.with_store(store);
        self
    }

    /// Captures the underlying graph runtime snapshot.
    pub fn snapshot(&self) -> Result<Snapshot, GraphError> {
        self.runtime
            .as_ref()
            .ok_or_else(|| GraphError::Invalid("chat runtime has not started".into()))?
            .snapshot()
    }

    /// Sends one input and returns the next assistant response.
    pub async fn send(&mut self, input: I, ctx: Context) -> Result<ChatTurn<O>, GraphError> {
        let input = to_value(input).map_err(|err| GraphError::ValueConversion {
            target: "chat input".into(),
            reason: err.to_string(),
        })?;
        if self.runtime.is_none() {
            let flow = build_chat_flow(self.agent)?;
            self.runtime = Some(
                flow.prepared()
                    .start(input)?
                    .with_runtime_services(self.services.clone()),
            );
        } else {
            let step = self.runtime_mut()?.resume(input, ctx.clone()).await?;
            if let Some(turn) = decode_chat_step(step)? {
                return Ok(turn);
            }
        }

        loop {
            let step = self.runtime_mut()?.next(ctx.clone()).await?;
            if let Some(turn) = decode_chat_step(step)? {
                return Ok(turn);
            }
        }
    }

    fn runtime_mut(&mut self) -> Result<&mut Runtime, GraphError> {
        self.runtime
            .as_mut()
            .ok_or_else(|| GraphError::Invalid("chat runtime disappeared".into()))
    }
}

fn build_chat_flow<I, O>(agent: fn(Agent<I>) -> Agent<O>) -> Result<CompiledFlow<I, I>, GraphError>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let root = Flow::<I>::root();
    let start = root.mark();
    let _loop_edge = root.clone().agent(agent).suspend::<I>().goto(start);
    root.map(|value| value).finish::<I>()
}

fn decode_chat_step<O>(step: Step) -> Result<Option<ChatTurn<O>>, GraphError>
where
    O: DeserializeOwned,
{
    match step {
        Step::Continue => Ok(None),
        Step::Suspend(value) => from_value(value)
            .map(|output| Some(ChatTurn { output }))
            .map_err(|err| GraphError::ValueConversion {
                target: "chat response".into(),
                reason: err.to_string(),
            }),
        Step::Done(value) => Err(GraphError::Invalid(format!(
            "chat runtime completed unexpectedly with {value}"
        ))),
    }
}
