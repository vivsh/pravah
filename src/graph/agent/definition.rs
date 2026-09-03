use std::error::Error;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use futures::future::{BoxFuture, FutureExt};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::Context;

use super::control::{AgentDecision, AgentLoop, AgentLoopData};
use super::{AgentConfig, Toolset};
use crate::graph::error::GraphError;
use crate::graph::value::{Value, from_value};

type ConfigureCall =
    dyn Fn(Value, Context) -> BoxFuture<'static, Result<AgentConfig, GraphError>> + Send + Sync;

type ControlCall = dyn Fn(AgentLoopData, Context) -> BoxFuture<'static, Result<AgentDecision, GraphError>>
    + Send
    + Sync;

#[derive(Clone)]
pub(crate) struct AgentConfigurator {
    call: Arc<ConfigureCall>,
}

impl AgentConfigurator {
    pub(crate) fn missing() -> Self {
        Self {
            call: Arc::new(|_, _| {
                async {
                    Err(GraphError::AgentConfigValidation(
                        "agent configure function is missing".into(),
                    ))
                }
                .boxed()
            }),
        }
    }

    /// Resolves one invocation's complete dynamic agent configuration.
    pub(crate) async fn configure(
        &self,
        input: Value,
        ctx: Context,
    ) -> Result<AgentConfig, GraphError> {
        (self.call)(input, ctx).await
    }
}

#[derive(Clone)]
pub(crate) struct AgentController {
    call: Arc<ControlCall>,
}

impl AgentController {
    /// Evaluates one explicit agent-loop intervention boundary.
    pub(crate) async fn control(
        &self,
        data: AgentLoopData,
        ctx: Context,
    ) -> Result<AgentDecision, GraphError> {
        (self.call)(data, ctx).await
    }
}

struct AgentDefinition {
    tools: Toolset,
    controller: Option<AgentController>,
    configure: Option<AgentConfigurator>,
    errors: Vec<String>,
}

/// Typed root used to define one agent as `Agent<Input> -> Agent<Output>`.
///
/// Candidate tools are structural and prepared with the graph. The terminal
/// configuration function resolves all invocation-specific behavior once.
pub struct Agent<T> {
    definition: AgentDefinition,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Agent<T> {
    pub(crate) fn root() -> Self {
        Self {
            definition: AgentDefinition {
                tools: Toolset::default(),
                controller: None,
                configure: None,
                errors: Vec::new(),
            },
            _marker: PhantomData,
        }
    }

    /// Declares every tool graph this agent may expose at runtime.
    pub fn tools(mut self, build: fn(Toolset) -> Toolset) -> Self {
        if self.definition.configure.is_some() {
            self.definition
                .errors
                .push("agent tools must be declared before configure".into());
            return self;
        }
        self.definition.tools = build(self.definition.tools);
        self
    }

    /// Registers an optional asynchronous agent-loop controller.
    ///
    /// The controller receives typed invocation input and checkpointed loop
    /// observations at explicit boundaries. It must be declared before the
    /// terminal configuration function.
    pub fn control<Fut, E>(mut self, control: fn(AgentLoop<T>, Context) -> Fut) -> Self
    where
        T: 'static + DeserializeOwned + JsonSchema + Send + Sync,
        Fut: Future<Output = Result<AgentDecision, E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        if self.definition.configure.is_some() {
            self.definition
                .errors
                .push("agent control must be declared before configure".into());
        } else if self.definition.controller.is_some() {
            self.definition
                .errors
                .push("agent control may only be declared once".into());
        } else {
            self.definition.controller = Some(AgentController {
                call: Arc::new(move |data, ctx| {
                    async move {
                        let input = from_value::<T>(data.input.clone()).map_err(|err| {
                            GraphError::AgentControl {
                                agent: data.agent_id.clone(),
                                reason: format!(
                                    "failed to decode agent input '{}': {err}",
                                    T::schema_name()
                                ),
                            }
                        })?;
                        let agent = data.agent_id.clone();
                        control(AgentLoop::from_data(input, data), ctx)
                            .await
                            .map_err(|err| GraphError::AgentControl {
                                agent,
                                reason: err.to_string(),
                            })
                    }
                    .boxed()
                }),
            });
        }
        self
    }

    /// Registers the terminal asynchronous configuration function.
    ///
    /// The function receives owned input and a cheap clone of the runtime
    /// context. Its resolved configuration is checkpointed and is not rerun
    /// after restoration.
    pub fn configure<O, Fut, E>(mut self, configure: fn(T, Context) -> Fut) -> Agent<O>
    where
        T: 'static + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + DeserializeOwned + JsonSchema + Send + Sync,
        Fut: Future<Output = Result<AgentConfig, E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        if self.definition.configure.is_some() {
            self.definition
                .errors
                .push("agent configure may only be declared once".into());
        } else {
            let agent = O::schema_name();
            self.definition.configure = Some(AgentConfigurator {
                call: Arc::new(move |value, ctx| {
                    let agent = agent.clone();
                    async move {
                        let input = from_value::<T>(value).map_err(|err| {
                            GraphError::AgentConfigValidation(format!(
                                "failed to decode agent input '{}': {err}",
                                T::schema_name()
                            ))
                        })?;
                        configure(input, ctx)
                            .await
                            .map_err(|err| GraphError::AgentConfiguration {
                                agent,
                                reason: err.to_string(),
                            })
                    }
                    .boxed()
                }),
            });
        }
        Agent {
            definition: self.definition,
            _marker: PhantomData,
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Toolset,
        Option<AgentController>,
        Option<AgentConfigurator>,
        Vec<String>,
    ) {
        (
            self.definition.tools,
            self.definition.controller,
            self.definition.configure,
            self.definition.errors,
        )
    }
}
