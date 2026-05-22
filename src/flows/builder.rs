use std::collections::HashMap;
use std::sync::Arc;

use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::flow::{Flow, FlowGraph};
use super::nary::{MergeInputs, SplitOutputs};
use super::nodes::{
    AgentInfo, EitherInfo, FlowNode, ForkInfo, JoinInfo, MapInfo, StateNode, SuspendInfo,
    ToolInfo, WorkInfo, build_tool_definition, node,
};
use crate::flows::NodeId;
use crate::flows::errors::{BuildError, FlowError};
use crate::flows::validation::validate_nodes;
use crate::{
    clients::Message,
    commons::{Agent, make_agent_message},
    context::Context,
    tools::base::pascal_to_snake,
    tools::{SuspendedValue, ToolError, ToolOutput},
};

pub struct FlowBuilder {
    flow: FlowGraph,
    errors: Vec<String>,
}

impl FlowBuilder {
    pub(crate) fn new() -> Self {
        Self {
            flow: FlowGraph::new(),
            errors: Vec::new(),
        }
    }

    /// Applies `f` to this builder, enabling modular flow composition.
    pub fn pipe(self, f: impl FnOnce(Self) -> Self) -> Self {
        f(self)
    }

    /// Registers an agent node keyed by `A::node_id()`.
    pub fn agent<A: Agent>(mut self) -> Self {
        let name_str = A::node_id();
        let name = self.flow.interner.intern(&name_str);
        if self.flow.nodes.contains_key(&name) {
            self.errors
                .push(format!("agent '{}': duplicate node key", name_str));
            return self;
        }
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let input_schema = match serde_json::to_value(schema_gen.root_schema_for::<A>()) {
            Ok(v) => v,
            Err(e) => {
                self.errors
                    .push(format!("agent '{}' input schema: {e}", name_str));
                return self;
            }
        };
        let output_schema = match serde_json::to_value(schema_gen.root_schema_for::<A::Output>()) {
            Ok(v) => v,
            Err(e) => {
                self.errors
                    .push(format!("agent '{}' output schema: {e}", name_str));
                return self;
            }
        };
        let config = A::build();
        let output_str = A::Output::schema_name();
        let output_id = self.flow.interner.intern(&output_str);
        let agent_info = AgentInfo {
            id: name,
            tools: Vec::new(),
            make_message: make_agent_message::<A>,
            preamble: config.preamble,
            input_schema,
            model: config.model_url,
            exit: output_id,
            output_schema,
            tool_lookup: HashMap::new(),
            keep_alive: config.keep_alive,
            turn_budget: config.turn_budget,
            turn_budget_message: config.turn_budget_message,
        };
        self.flow
            .nodes
            .insert(name, FlowNode::Agent(Arc::new(agent_info)));
        self
    }

    /// Attaches a tool to agent `A`.
    pub fn tool<A, I, O>(mut self) -> Self
    where
        A: Agent,
        I: 'static + DeserializeOwned + JsonSchema + Send,
        O: ToolOutput,
    {
        let agent_str = A::node_id();
        let agent_id = self.flow.interner.intern(&agent_str);

        let definition = match build_tool_definition::<I>() {
            Ok(d) => d,
            Err(e) => {
                self.errors.push(format!("tool '{}' schema: {e}", pascal_to_snake(&I::schema_name())));
                return self;
            }
        };
        let tool_name = definition.name.clone();

        let entry_id = self.flow.interner.intern(&I::schema_name());
        let exit_id = self.flow.interner.intern(&O::schema_name());
        let to_message: Box<dyn Fn(Value) -> Result<Message, ToolError> + Send + Sync> =
            Box::new(|value: Value| -> Result<Message, ToolError> {
                let o: O = serde_json::from_value(value).map_err(ToolError::Deserialize)?;
                o.to_message()
            });

        match self.flow.nodes.get_mut(&agent_id) {
            Some(FlowNode::Agent(arc)) => match Arc::get_mut(arc) {
                Some(info) => {
                    info.tools.push(ToolInfo {
                        definition,
                        exit_id,
                        to_message,
                    });
                    info.tool_lookup.insert(tool_name, (entry_id, exit_id));
                    let synthetic_name = pascal_to_snake(&A::Output::schema_name());
                    if !info.tool_lookup.contains_key(&synthetic_name) {
                        let synthetic_exit = info.exit;
                        match build_tool_definition::<A::Output>() {
                            Ok(def) => {
                                info.tools.push(ToolInfo {
                                    definition: def,
                                    exit_id: synthetic_exit,
                                    to_message: Box::new(|value: Value| {
                                        Ok(Message::tool_output(
                                            String::new(),
                                            serde_json::to_string(&value).unwrap_or_default(),
                                        ))
                                    }),
                                });
                                info.tool_lookup
                                    .insert(synthetic_name, (synthetic_exit, synthetic_exit));
                            }
                            Err(e) => {
                                self.errors.push(format!(
                                    "agent '{}': synthetic exit tool '{}' schema: {e}",
                                    agent_str, synthetic_name
                                ));
                            }
                        }
                    }
                }
                None => {
                    self.errors.push(format!(
                        "tool: agent '{}' Arc is shared; cannot mutate",
                        agent_str
                    ));
                }
            },
            _ => {
                self.errors.push(format!(
                    "tool: agent '{}' not found (register it with .agent::<A>() first)",
                    agent_str
                ));
            }
        }

        self
    }

    /// Registers a pure branch node.
    pub fn either<From, A, B, H>(mut self, func: H) -> Self
    where
        From: Serialize + DeserializeOwned + JsonSchema,
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(From) -> Either<A, B> + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("either '{}': duplicate node key", from_id_str));
            return self;
        }
        let left_name = self.flow.interner.intern(&A::schema_name());
        let right_name = self.flow.interner.intern(&B::schema_name());
        let shim: Box<dyn Fn(&Value) -> Result<(NodeId, Value), FlowError> + Send + Sync> =
            Box::new(move |value: &Value| {
                let typed: From =
                    serde_json::from_value(value.clone()).map_err(FlowError::Deserialize)?;
                match func(typed) {
                    Either::Left(a) => {
                        let v = serde_json::to_value(&a).map_err(FlowError::Serialize)?;
                        Ok((left_name, v))
                    }
                    Either::Right(b) => {
                        let v = serde_json::to_value(&b).map_err(FlowError::Serialize)?;
                        Ok((right_name, v))
                    }
                }
            });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Either(EitherInfo {
                entry: from_id,
                left_name,
                right_name,
                func: shim,
            }),
        );

        self
    }

    /// Registers a pure 1->2 fan-out node.
    pub fn fork<From, A, B, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(From) -> (A, B) + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("fork '{}': duplicate node key", from_id_str));
            return self;
        }
        let shim: Box<dyn Fn(&Value) -> Result<Vec<StateNode>, FlowError> + Send + Sync> =
            Box::new(move |value: &Value| {
                let typed: From =
                    serde_json::from_value(value.clone()).map_err(FlowError::Deserialize)?;
                let (a, b) = func(typed);
                Ok(vec![node(a)?, node(b)?])
            });
        let a_child = self.flow.interner.intern(&A::schema_name());
        let b_child = self.flow.interner.intern(&B::schema_name());
        self.flow.nodes.insert(
            from_id,
            FlowNode::Fork(ForkInfo {
                name: from_id,
                children: vec![a_child, b_child],
                func: shim,
            }),
        );
        self
    }

    /// Registers a pure 2->1 join node.
    pub fn join<A, B, Out, H>(mut self, func: H) -> Self
    where
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(A, B) -> Out + Send + Sync + 'static,
    {
        let a_id_str = A::schema_name();
        let b_id_str = B::schema_name();
        let a_id = self.flow.interner.intern(&a_id_str);
        let b_id = self.flow.interner.intern(&b_id_str);
        for (id, id_str) in [(a_id, &a_id_str), (b_id, &b_id_str)] {
            if self.flow.nodes.contains_key(&id) {
                self.errors
                    .push(format!("join: duplicate node key '{}'", id_str));
                return self;
            }
        }
        let target_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Arc<dyn Fn(&[Value]) -> Result<StateNode, FlowError> + Send + Sync> =
            Arc::new(move |inputs: &[Value]| {
                let a: A =
                    serde_json::from_value(inputs[0].clone()).map_err(FlowError::Deserialize)?;
                let b: B =
                    serde_json::from_value(inputs[1].clone()).map_err(FlowError::Deserialize)?;
                node(func(a, b))
            });
        self.flow.nodes.insert(
            a_id,
            FlowNode::Join(JoinInfo {
                parents: vec![a_id, b_id],
                target: target_id,
                func: Arc::clone(&shim),
            }),
        );
        self.flow.nodes.insert(
            b_id,
            FlowNode::Join(JoinInfo {
                parents: vec![a_id, b_id],
                target: target_id,
                func: shim,
            }),
        );
        self
    }

    /// Registers a pure 1->N fan-out node.
    pub fn split<From, Out, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: SplitOutputs,
        H: Fn(From) -> Out + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("split '{}': duplicate node key", from_id_str));
            return self;
        }
        let children: Vec<NodeId> = Out::schema_names()
            .into_iter()
            .map(|s| self.flow.interner.intern(&s))
            .collect();
        let shim: Box<dyn Fn(&Value) -> Result<Vec<StateNode>, FlowError> + Send + Sync> =
            Box::new(move |value: &Value| {
                let typed: From =
                    serde_json::from_value(value.clone()).map_err(FlowError::Deserialize)?;
                func(typed).into_nodes()
            });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Fork(ForkInfo {
                name: from_id,
                children,
                func: shim,
            }),
        );
        self
    }

    /// Registers a pure N->1 join node.
    pub fn merge<In, Out, H>(mut self, func: H) -> Self
    where
        In: MergeInputs,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(In) -> Out + Send + Sync + 'static,
    {
        let parent_names = In::schema_names();
        let parent_ids: Vec<NodeId> = parent_names
            .iter()
            .map(|s| self.flow.interner.intern(s))
            .collect();
        for (id, name) in parent_ids.iter().zip(&parent_names) {
            if self.flow.nodes.contains_key(id) {
                self.errors
                    .push(format!("merge: duplicate node key '{}'", name));
                return self;
            }
        }
        let target_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Arc<dyn Fn(&[Value]) -> Result<StateNode, FlowError> + Send + Sync> =
            Arc::new(move |inputs: &[Value]| {
                let typed = In::from_values(inputs)?;
                node(func(typed))
            });
        for &pid in &parent_ids {
            self.flow.nodes.insert(
                pid,
                FlowNode::Join(JoinInfo {
                    parents: parent_ids.clone(),
                    target: target_id,
                    func: Arc::clone(&shim),
                }),
            );
        }
        self
    }

    /// Embeds a child flow.
    pub fn flow<F: Flow>(mut self) -> Self {
        let input_str = F::schema_name();
        let output_str = F::Output::schema_name();
        let input_id = self.flow.interner.intern(&input_str);
        let output_id = self.flow.interner.intern(&output_str);

        if self.flow.nodes.contains_key(&input_id) {
            self.errors
                .push(format!("flow '{}': duplicate node key", input_str));
            return self;
        }

        let mut inner = match FlowGraph::from_flow::<F>() {
            Ok(g) => g,
            Err(e) => {
                self.errors.push(format!("flow '{}': {e}", input_str));
                return self;
            }
        };

        inner.parent_entry = Some(input_id);
        inner.parent_exit = Some(output_id);

        self.flow
            .nodes
            .insert(input_id, FlowNode::Flow(Arc::new(inner)));

        self
    }

    /// Registers an async work node.
    pub fn work<From, Out, Fut, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        Fut: std::future::Future<Output = Result<Out, FlowError>> + Send + 'static,
        H: Fn(From, Context) -> Fut + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("work '{}': duplicate node key", from_id_str));
            return self;
        }
        let exit_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Box<
            dyn Fn(&Value, Context) -> BoxFuture<'static, Result<Value, FlowError>> + Send + Sync,
        > = Box::new(move |value: &Value, ctx: Context| {
            let typed: From = match serde_json::from_value(value.clone()) {
                Ok(v) => v,
                Err(e) => {
                    let err = FlowError::Deserialize(e);
                    return Box::pin(async move { Err(err) });
                }
            };
            let fut = func(typed, ctx);
            Box::pin(async move {
                let out = fut.await?;
                serde_json::to_value(&out).map_err(FlowError::Serialize)
            })
        });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Work(WorkInfo {
                name: from_id,
                exit_name: exit_id,
                func: shim,
            }),
        );
        self
    }

    /// Registers a pure synchronous transform node.
    pub fn map<From, Out, H>(mut self, func: H) -> Self
    where
        From: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(From) -> Out + Send + Sync + 'static,
    {
        let from_id_str = From::schema_name();
        let from_id = self.flow.interner.intern(&from_id_str);
        if self.flow.nodes.contains_key(&from_id) {
            self.errors
                .push(format!("map '{}': duplicate node key", from_id_str));
            return self;
        }
        let exit_id = self.flow.interner.intern(&Out::schema_name());
        let shim: Box<dyn Fn(&Value) -> Result<Value, FlowError> + Send + Sync> =
            Box::new(move |value: &Value| {
                let typed: From =
                    serde_json::from_value(value.clone()).map_err(FlowError::Deserialize)?;
                let out = func(typed);
                serde_json::to_value(&out).map_err(FlowError::Serialize)
            });
        self.flow.nodes.insert(
            from_id,
            FlowNode::Map(MapInfo {
                name: from_id,
                exit_name: exit_id,
                func: shim,
            }),
        );
        self
    }

    /// Registers a flow-level suspend point.
    pub fn suspend<I, O>(mut self) -> Self
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send,
        O: 'static + Serialize + DeserializeOwned + JsonSchema,
    {
        let entry_str = I::schema_name();
        let exit_str = O::schema_name();
        let entry = self.flow.interner.intern(&entry_str);
        let exit = self.flow.interner.intern(&exit_str);
        if self.flow.nodes.contains_key(&entry) {
            self.errors
                .push(format!("suspend '{}': duplicate node key", entry_str));
            return self;
        }
        let output_type = exit_str.to_string();
        let deserialize: Box<
            dyn Fn(Value) -> Result<SuspendedValue, serde_json::Error> + Send + Sync,
        > = Box::new(|v| serde_json::from_value::<I>(v).map(SuspendedValue::new));
        self.flow.nodes.insert(
            entry,
            FlowNode::Suspend(SuspendInfo {
                entry,
                exit,
                output_type,
                deserialize,
            }),
        );
        self
    }

    /// Validates structural rules and returns the graph.
    pub fn build(self) -> Result<FlowGraph, FlowError> {
        if !self.errors.is_empty() {
            return Err(BuildError::Invalid(self.errors).into());
        }
        validate_nodes(&self.flow.nodes, &self.flow)?;
        Ok(self.flow)
    }
}