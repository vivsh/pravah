use std::collections::BTreeMap;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};

use super::agent::{Agent, build_agent};
use super::builder::UntypedGraphBuilder;
use super::error::GraphError;
use super::ids::{EdgeId, HandlerKey, MarkId, VarId};
use super::model::{NodeKind, TypeSpec, UntypedGraph, VarInit, VarKey, VarScope};
use super::registry::{ContinuationHandler, HandlerRegistry};
use super::runtime::{PreparedGraph, Runtime, Snapshot};
use super::value::{Value, from_value, to_value};
use crate::Context;

mod build;
mod support;
mod tuples;

use build::*;
use support::*;
pub use tuples::{MergeFlows, SplitOutputs};

#[doc(hidden)]
pub struct TypedBuildState {
    builder: UntypedGraphBuilder,
    registry: HandlerRegistry,
    handler_counter: usize,
    graph_name: String,
    variables: BTreeMap<VarKey, VarId>,
    errors: Vec<String>,
}

/// String-free typed graph builder over the untyped graph core.
///
/// This is the extension layer intended for alternate Rust frontends. It keeps
/// user-facing APIs typed while generating deterministic internal graph labels
/// and handler keys from declaration order and Rust type metadata.
pub struct TypedGraphBuilder<I> {
    state: Arc<Mutex<TypedBuildState>>,
    root: EdgeId,
    _marker: PhantomData<fn() -> I>,
}

impl<I> Clone for TypedGraphBuilder<I> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            root: self.root,
            _marker: PhantomData,
        }
    }
}

impl<I> Default for TypedGraphBuilder<I>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Typed handle to one value edge in a [`TypedGraphBuilder`].
pub struct TypedEdge<T> {
    state: Arc<Mutex<TypedBuildState>>,
    edge: EdgeId,
    _marker: PhantomData<fn() -> T>,
}

/// Typed handle to a frame variable declared in a [`TypedGraphBuilder`].
pub struct TypedVar<T> {
    state: Arc<Mutex<TypedBuildState>>,
    var: Option<VarId>,
    _marker: PhantomData<fn() -> T>,
}

/// Typed handle to a graph re-entry mark.
///
/// Use `flow.mark()` to declare one and `other.goto(mark)` to loop or re-enter
/// that edge with a fresh value generation.
pub struct TypedMark<T> {
    state: Arc<Mutex<TypedBuildState>>,
    mark: Option<MarkId>,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for TypedEdge<T> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            edge: self.edge,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for TypedVar<T> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            var: self.var,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for TypedMark<T> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            mark: self.mark,
            _marker: PhantomData,
        }
    }
}

impl<I> TypedGraphBuilder<I>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Starts a typed builder using the input type as the graph name.
    pub fn new() -> Self {
        Self::new_with_internal_name(I::schema_name())
    }

    fn new_with_internal_name(name: impl Into<String>) -> Self {
        let name = name.into();
        let mut builder = UntypedGraphBuilder::new(&name);
        let entry = builder.edge("entry", type_spec::<I>());
        builder.set_entry(entry);
        Self {
            state: Arc::new(Mutex::new(TypedBuildState {
                builder,
                registry: HandlerRegistry::new(),
                handler_counter: 0,
                graph_name: name,
                variables: BTreeMap::new(),
                errors: Vec::new(),
            })),
            root: entry,
            _marker: PhantomData,
        }
    }

    /// Returns the typed entry edge for the graph.
    pub fn root(&self) -> TypedEdge<I> {
        typed_edge(Arc::clone(&self.state), self.root)
    }

    /// Declares a frame-local typed variable with an initial value.
    pub fn local<S>(&self, value: S) -> TypedVar<S>
    where
        S: 'static + Serialize + JsonSchema,
    {
        declare_variable::<S>(&self.state, value, VarScope::Local)
    }

    /// Declares an inherited typed variable with a fallback value.
    ///
    /// Child frames copy a matching parent variable when one exists; writes stay
    /// local to the child frame.
    pub fn inherit<S>(&self, value: S) -> TypedVar<S>
    where
        S: 'static + Serialize + JsonSchema,
    {
        declare_variable::<S>(&self.state, value, VarScope::Inherit)
    }

    /// Declares a typed re-entry mark on an existing edge.
    pub fn mark<T>(&self, input: TypedEdge<T>) -> TypedMark<T>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        if !self.same_graph(&input) {
            push_error(
                &self.state,
                "mark: input edge belongs to another typed builder",
            );
            return typed_mark(Arc::clone(&self.state), None);
        }
        add_mark(input.state, input.edge)
    }

    /// Writes this edge's next value to a typed re-entry mark.
    pub fn goto<T>(&self, input: TypedEdge<T>, mark: TypedMark<T>) -> TypedEdge<T>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        if !self.same_graph(&input) || !self.same_mark_graph(&mark) {
            push_error(
                &self.state,
                "goto: input edge and mark must belong to this typed builder",
            );
            return typed_edge(Arc::clone(&self.state), input.edge);
        }
        add_goto_node(input.state, input.edge, mark.mark)
    }

    /// Adds a pure typed transform node to the untyped graph.
    pub fn map<T, P, H>(&self, input: TypedEdge<T>, func: H) -> TypedEdge<P>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn(T) -> P + Send + Sync + 'static,
    {
        if !self.same_graph(&input) {
            push_error(
                &self.state,
                "map: input edge belongs to another typed builder",
            );
            return typed_edge(Arc::clone(&self.state), input.edge);
        }
        add_map_node(input.state, input.edge, func)
    }

    /// Splits one typed value into two typed output edges.
    pub fn split<T, A, B, H>(&self, input: TypedEdge<T>, func: H) -> (TypedEdge<A>, TypedEdge<B>)
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        A: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        B: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn(T) -> (A, B) + Send + Sync + 'static,
    {
        if !self.same_graph(&input) {
            push_error(
                &self.state,
                "split: input edge belongs to another typed builder",
            );
            return (
                typed_edge(Arc::clone(&self.state), input.edge),
                typed_edge(Arc::clone(&self.state), input.edge),
            );
        }
        let (left, right): (Flow<A>, Flow<B>) =
            add_split_node::<T, (A, B), H>(input.state, input.edge, func);
        (
            typed_edge(left.state, left.edge),
            typed_edge(right.state, right.edge),
        )
    }

    /// Merges two typed edges into one output value.
    pub fn merge<A, B, Out, H>(
        &self,
        left: TypedEdge<A>,
        right: TypedEdge<B>,
        func: H,
    ) -> TypedEdge<Out>
    where
        A: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        B: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn((A, B)) -> Out + Send + Sync + 'static,
    {
        if !self.same_graph(&left) || !self.same_graph(&right) {
            push_error(
                &self.state,
                "merge: input edges must belong to this typed builder",
            );
            return typed_edge(Arc::clone(&self.state), left.edge);
        }
        add_merge_node::<(A, B), Out, H>(left.state, vec![left.edge, right.edge], func)
    }

    /// Adds a one-shot async work node.
    pub fn work<T, P, Fut, H>(&self, input: TypedEdge<T>, func: H) -> TypedEdge<P>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Fut: Future<Output = Result<P, GraphError>> + Send + 'static,
        H: Fn(T, Context) -> Fut + Send + Sync + 'static,
    {
        if !self.same_graph(&input) {
            push_error(
                &self.state,
                "work: input edge belongs to another typed builder",
            );
            return typed_edge(Arc::clone(&self.state), input.edge);
        }
        add_work_node(input.state, input.edge, func)
    }

    /// Adds a multi-step continuation node by handler type.
    ///
    /// Use this as the low-level hook for higher-level facades such as agents.
    pub fn continuation<T, O, H, P>(&self, input: TypedEdge<T>, payload: P) -> TypedEdge<O>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: ContinuationHandler + Default + 'static,
        P: Serialize,
    {
        if !self.same_graph(&input) {
            push_error(
                &self.state,
                "continuation: input edge belongs to another typed builder",
            );
            return typed_edge(Arc::clone(&self.state), input.edge);
        }
        add_continuation_node::<O, H, P>(input.state, input.edge, payload)
    }

    /// Reads a typed variable and computes the next edge value.
    pub fn load<T, S, O, H>(&self, input: TypedEdge<T>, var: TypedVar<S>, func: H) -> TypedEdge<O>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        S: 'static + Serialize + DeserializeOwned + JsonSchema,
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn(T, S) -> O + Send + Sync + 'static,
    {
        if !self.same_graph(&input) || !self.same_var_graph(&var) {
            push_error(
                &self.state,
                "load: input edge and variable must belong to this typed builder",
            );
            return typed_edge(Arc::clone(&self.state), input.edge);
        }
        add_load_node::<T, S, O, H>(input.state, input.edge, var.var, func)
    }

    /// Updates a typed variable while passing the input edge value through.
    pub fn store<T, S, H>(&self, input: TypedEdge<T>, var: TypedVar<S>, func: H) -> TypedEdge<T>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        S: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(T, S) -> S + Send + Sync + 'static,
    {
        if !self.same_graph(&input) || !self.same_var_graph(&var) {
            push_error(
                &self.state,
                "store: input edge and variable must belong to this typed builder",
            );
            return typed_edge(Arc::clone(&self.state), input.edge);
        }
        add_store_node::<T, S, H>(input.state, input.edge, var.var, func)
    }

    /// Calls another typed flow as a child frame.
    pub fn flow<T, O>(&self, input: TypedEdge<T>, flow: fn(Flow<T>) -> Flow<O>) -> TypedEdge<O>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        if !self.same_graph(&input) {
            push_error(
                &self.state,
                "flow: input edge belongs to another typed builder",
            );
            return typed_edge(Arc::clone(&self.state), input.edge);
        }
        add_flow_node::<T, O>(input.state, input.edge, flow)
    }

    /// Runs a typed child flow sequentially for each vector item.
    pub fn each<T, O>(
        &self,
        input: TypedEdge<Vec<T>>,
        flow: fn(Flow<T>) -> Flow<O>,
    ) -> TypedEdge<Vec<O>>
    where
        T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Vec<T>: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Vec<O>: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        if !self.same_graph(&input) {
            push_error(
                &self.state,
                "each: input edge belongs to another typed builder",
            );
            return typed_edge(Arc::clone(&self.state), input.edge);
        }
        add_each_node::<T, O>(input.state, input.edge, flow)
    }

    /// Finalizes the typed builder into a graph plus registry.
    pub fn finish<O>(self, output: TypedEdge<O>) -> Result<CompiledFlow<I, O>, GraphError>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema,
    {
        if !self.same_graph(&output) {
            return Err(GraphError::Invalid(
                "finish: output edge belongs to another typed builder".into(),
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| GraphError::Invalid("typed graph builder lock is poisoned".into()))?;
        if !state.errors.is_empty() {
            return Err(GraphError::Invalid(state.errors.join("; ")));
        }
        state.builder.set_exit(output.edge);
        let replacement = UntypedGraphBuilder::new(format!("{}_finished", state.graph_name));
        let builder = std::mem::replace(&mut state.builder, replacement);
        let graph = builder.build()?;
        let prepared = PreparedGraph::new(graph, state.registry.clone())?;
        Ok(CompiledFlow {
            prepared,
            _marker: PhantomData,
        })
    }

    fn same_graph<T>(&self, edge: &TypedEdge<T>) -> bool {
        Arc::ptr_eq(&self.state, &edge.state)
    }

    fn same_var_graph<T>(&self, var: &TypedVar<T>) -> bool {
        Arc::ptr_eq(&self.state, &var.state)
    }

    fn same_mark_graph<T>(&self, mark: &TypedMark<T>) -> bool {
        Arc::ptr_eq(&self.state, &mark.state)
    }
}

fn typed_edge<T>(state: Arc<Mutex<TypedBuildState>>, edge: EdgeId) -> TypedEdge<T> {
    TypedEdge {
        state,
        edge,
        _marker: PhantomData,
    }
}

fn typed_var<T>(state: Arc<Mutex<TypedBuildState>>, var: Option<VarId>) -> TypedVar<T> {
    TypedVar {
        state,
        var,
        _marker: PhantomData,
    }
}

fn typed_mark<T>(state: Arc<Mutex<TypedBuildState>>, mark: Option<MarkId>) -> TypedMark<T> {
    TypedMark {
        state,
        mark,
        _marker: PhantomData,
    }
}

/// Typed handle over the graph backend.
///
/// Normal methods do not require strings and do not return `Result`. Build
/// errors accumulate and are surfaced when the graph is finalized.
pub struct Flow<T> {
    state: Arc<Mutex<TypedBuildState>>,
    edge: EdgeId,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Flow<T> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            edge: self.edge,
            _marker: PhantomData,
        }
    }
}

/// Typed conditional branch handle produced by [`Flow::either`].
pub struct EitherFlow<T, A, B, H> {
    state: Arc<Mutex<TypedBuildState>>,
    edge: EdgeId,
    choose: H,
    _marker: PhantomData<fn(T) -> (A, B)>,
}

/// Compiled typed flow plus the runtime registry needed to execute handlers.
pub struct CompiledFlow<I, O> {
    prepared: PreparedGraph,
    _marker: PhantomData<fn(I) -> O>,
}

/// Compiles a function-defined typed flow into the canonical graph runtime.
pub fn compile<I, O>(flow: fn(Flow<I>) -> Flow<O>) -> Result<CompiledFlow<I, O>, GraphError>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let name = format!("{}_to_{}", I::schema_name(), O::schema_name());
    flow(Flow::<I>::root_named(name)).finish::<I>()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum BranchSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Serialize)]
struct BranchChoice {
    side: BranchSide,
    value: Value,
}

impl<T> Flow<T>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Creates a root flow using the type name as the graph name.
    pub fn root() -> Self {
        Self::root_named(T::schema_name())
    }

    /// Creates a root flow with an internally derived graph name.
    pub(crate) fn root_named(name: impl Into<String>) -> Self {
        let builder = TypedGraphBuilder::<T>::new_with_internal_name(name);
        let root = builder.root();
        let state = root.state;
        let edge = root.edge;
        Self {
            state,
            edge,
            _marker: PhantomData,
        }
    }

    /// Declares a local variable on this flow's builder.
    pub fn local<S>(&self, value: S) -> TypedVar<S>
    where
        S: 'static + Serialize + JsonSchema,
    {
        declare_variable::<S>(&self.state, value, VarScope::Local)
    }

    /// Declares an inherited variable on this flow's builder.
    pub fn inherit<S>(&self, value: S) -> TypedVar<S>
    where
        S: 'static + Serialize + JsonSchema,
    {
        declare_variable::<S>(&self.state, value, VarScope::Inherit)
    }

    /// Declares this edge as a typed re-entry point.
    pub fn mark(&self) -> TypedMark<T> {
        add_mark::<T>(Arc::clone(&self.state), self.edge)
    }

    /// Writes this edge value to a previously declared mark.
    pub fn goto(self, mark: TypedMark<T>) -> Flow<T> {
        if !Arc::ptr_eq(&self.state, &mark.state) {
            push_error(
                &self.state,
                "goto: input edge and mark must belong to this typed builder",
            );
            return self;
        }
        Flow::from_typed(add_goto_node::<T>(
            Arc::clone(&self.state),
            self.edge,
            mark.mark,
        ))
    }

    /// Adds a pure map node in fluent style.
    pub fn map<P, H>(self, func: H) -> Flow<P>
    where
        P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn(T) -> P + Send + Sync + 'static,
    {
        self.map_named(format!("map_to_{}", P::schema_name()), func)
    }

    /// Adds a pure map node with an internal display name.
    pub fn map_named<P, H>(self, name: impl Into<String>, func: H) -> Flow<P>
    where
        P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn(T) -> P + Send + Sync + 'static,
    {
        let edge =
            add_map_node_named::<T, P, H>(Arc::clone(&self.state), self.edge, name.into(), func);
        Flow::from_typed(edge)
    }

    /// Adds a one-shot async work node in fluent style.
    pub fn work<P, Fut, H>(self, func: H) -> Flow<P>
    where
        P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Fut: Future<Output = Result<P, GraphError>> + Send + 'static,
        H: Fn(T, Context) -> Fut + Send + Sync + 'static,
    {
        self.work_named(format!("work_to_{}", P::schema_name()), func)
    }

    /// Adds a one-shot async work node with an internal display name.
    pub fn work_named<P, Fut, H>(self, name: impl Into<String>, func: H) -> Flow<P>
    where
        P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Fut: Future<Output = Result<P, GraphError>> + Send + 'static,
        H: Fn(T, Context) -> Fut + Send + Sync + 'static,
    {
        let edge = add_work_node_named::<T, P, Fut, H>(
            Arc::clone(&self.state),
            self.edge,
            name.into(),
            func,
        );
        Flow::from_typed(edge)
    }

    /// Adds a first-class suspend point.
    ///
    /// `next()` pauses here and `resume()` supplies the output value.
    pub fn suspend<R>(self) -> Flow<R>
    where
        R: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let edge = add_suspend_node::<R>(Arc::clone(&self.state), self.edge);
        Flow::from_typed(edge)
    }

    /// Splits one value into multiple branch values.
    pub fn split<Out, H>(self, func: H) -> Out::Flows
    where
        Out: SplitOutputs,
        H: Fn(T) -> Out + Send + Sync + 'static,
    {
        add_split_node::<T, Out, H>(Arc::clone(&self.state), self.edge, func)
    }

    /// Merges this flow with additional typed flows.
    pub fn merge<Others, Out, H>(self, others: Others, func: H) -> Flow<Out>
    where
        Others: MergeFlows<T>,
        Others::Values: DeserializeOwned,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn(Others::Values) -> Out + Send + Sync + 'static,
    {
        let state = Arc::clone(&self.state);
        if !others.same_graph(&self.state) {
            push_error(
                &state,
                "merge: branches do not share the same edge graph builder",
            );
            return Flow::new(state, self.edge);
        }
        let mut input_edges = vec![self.edge];
        input_edges.append(&mut others.edge_ids());
        Flow::from_typed(add_merge_node::<Others::Values, Out, _>(
            state,
            input_edges,
            func,
        ))
    }

    /// Starts an either branch declaration.
    pub fn either<A, B, H>(self, func: H) -> EitherFlow<T, A, B, H>
    where
        A: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        B: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn(T) -> Either<A, B> + Send + Sync + 'static,
    {
        EitherFlow {
            state: self.state,
            edge: self.edge,
            choose: func,
            _marker: PhantomData,
        }
    }

    /// Loads a typed variable and computes the next flow value.
    pub fn load<S, O>(
        self,
        var: TypedVar<S>,
        func: impl Fn(T, S) -> O + Send + Sync + 'static,
    ) -> Flow<O>
    where
        S: 'static + Serialize + DeserializeOwned + JsonSchema,
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        self.load_named::<S, O, _>(format!("load_{}", S::schema_name()), var, func)
    }

    fn load_named<S, O, H>(self, name: impl Into<String>, var: TypedVar<S>, func: H) -> Flow<O>
    where
        S: 'static + Serialize + DeserializeOwned + JsonSchema,
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        H: Fn(T, S) -> O + Send + Sync + 'static,
    {
        let state = Arc::clone(&self.state);
        if !Arc::ptr_eq(&self.state, &var.state) {
            push_error(
                &state,
                "load: input edge and variable must belong to the same edge graph builder",
            );
            return Flow::new(state, self.edge);
        }
        let edge = add_load_node_named::<T, S, O, H>(
            Arc::clone(&self.state),
            self.edge,
            var.var,
            name.into(),
            func,
        );
        Flow::from_typed(edge)
    }

    /// Stores a typed variable and passes this input onward.
    pub fn store<S>(
        self,
        var: TypedVar<S>,
        func: impl Fn(T, S) -> S + Send + Sync + 'static,
    ) -> Self
    where
        S: 'static + Serialize + DeserializeOwned + JsonSchema,
    {
        self.store_named::<S, _>(format!("store_{}", S::schema_name()), var, func)
    }

    fn store_named<S, H>(self, name: impl Into<String>, var: TypedVar<S>, func: H) -> Self
    where
        S: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(T, S) -> S + Send + Sync + 'static,
    {
        let state = Arc::clone(&self.state);
        if !Arc::ptr_eq(&self.state, &var.state) {
            push_error(
                &state,
                "store: input edge and variable must belong to the same edge graph builder",
            );
            return Self::new(state, self.edge);
        }
        let edge = add_store_node_named::<T, S, H>(
            Arc::clone(&self.state),
            self.edge,
            var.var,
            name.into(),
            func,
        );
        Flow::from_typed(edge)
    }

    /// Calls a function-defined typed flow as a child frame.
    pub fn flow<O>(self, flow: fn(Flow<T>) -> Flow<O>) -> Flow<O>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        Flow::from_typed(add_flow_node::<T, O>(
            Arc::clone(&self.state),
            self.edge,
            flow,
        ))
    }

    /// Finalizes the fluent flow using this value as graph output.
    pub fn finish<I>(self) -> Result<CompiledFlow<I, T>, GraphError>
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let builder = TypedGraphBuilder::<I> {
            state: Arc::clone(&self.state),
            root: EdgeId(0),
            _marker: PhantomData,
        };
        builder.finish(typed_edge(self.state, self.edge))
    }

    fn new(state: Arc<Mutex<TypedBuildState>>, edge: EdgeId) -> Self {
        Self {
            state,
            edge,
            _marker: PhantomData,
        }
    }

    fn from_typed(edge: TypedEdge<T>) -> Self {
        Self {
            state: edge.state,
            edge: edge.edge,
            _marker: PhantomData,
        }
    }
}

impl<T, A, B, H> EitherFlow<T, A, B, H>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    A: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    B: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    H: Fn(T) -> Either<A, B> + Send + Sync + 'static,
{
    /// Completes the either declaration with left and right child flows.
    pub fn branch<Out, Left, Right>(self, left: Left, right: Right) -> Flow<Out>
    where
        Out: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Left: FnOnce(Flow<A>) -> Flow<Out>,
        Right: FnOnce(Flow<B>) -> Flow<Out>,
    {
        Flow::from_typed(add_either_node::<T, A, B, Out, H, Left, Right>(
            self.state,
            self.edge,
            self.choose,
            left,
            right,
        ))
    }
}

impl<T> Flow<Vec<T>>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Vec<T>: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Runs a child flow once per vector element, sequentially in the same VM.
    pub fn each<O>(self, flow: fn(Flow<T>) -> Flow<O>) -> Flow<Vec<O>>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Vec<O>: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        Flow::from_typed(add_each_node::<T, O>(
            Arc::clone(&self.state),
            self.edge,
            flow,
        ))
    }
}

impl<I> Flow<I>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    /// Adds a function-defined agent continuation node.
    pub fn agent<O>(self, build: fn(Agent<I>) -> Agent<O>) -> Flow<O>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        Flow::from_typed(add_agent_node::<I, O>(
            Arc::clone(&self.state),
            self.edge,
            build(Agent::root()),
        ))
    }
}

impl<I, O> CompiledFlow<I, O> {
    /// Returns the compiled untyped graph.
    pub fn graph(&self) -> &UntypedGraph {
        self.prepared.graph()
    }

    /// Returns the handler registry needed to execute the graph.
    pub fn registry(&self) -> &HandlerRegistry {
        self.prepared.registry()
    }

    /// Returns the reusable prepared graph backing this typed flow.
    pub fn prepared(&self) -> &PreparedGraph {
        &self.prepared
    }

    /// Splits the compiled flow into graph and registry parts.
    pub fn into_parts(self) -> (UntypedGraph, HandlerRegistry) {
        (
            self.prepared.graph().clone(),
            self.prepared.registry().clone(),
        )
    }
}

impl<I, O> CompiledFlow<I, O>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema,
    O: 'static + Serialize + DeserializeOwned + JsonSchema,
{
    /// Starts an isolated execution with its invocation context.
    ///
    /// Fails before execution exists when the typed input cannot enter the VM
    /// value domain or does not satisfy the graph entry schema.
    pub fn start(&self, input: I, ctx: Context) -> Result<Runtime, GraphError> {
        let input = to_value(input).map_err(|err| GraphError::ValueConversion {
            target: "workflow input".into(),
            reason: err.to_string(),
        })?;
        self.prepared.start(input, ctx)
    }

    /// Restores an execution and attaches its new runtime-only context.
    ///
    /// Fails when the snapshot version, graph fingerprint, or VM state is
    /// incompatible; the supplied snapshot is never partially restored.
    pub fn restore(&self, snapshot: Snapshot, ctx: Context) -> Result<Runtime, GraphError> {
        self.prepared.restore(snapshot, ctx)
    }

    /// Decodes a raw runtime output value into the typed output.
    pub fn decode_output(&self, value: Value) -> Result<O, GraphError> {
        from_value(value).map_err(|err| {
            GraphError::Invalid(format!(
                "failed to decode typed edge output '{}': {err}",
                O::schema_name()
            ))
        })
    }
}
