use either::Either;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use futures::future::Future;
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};

use super::builder::FlowBuilder;
use super::flow::Flow;
use super::nary::{MergeInputs, SplitOutputs};
use crate::commons::Agent;
use crate::context::Context;
use crate::legacy::errors::FlowError;
use crate::legacy::{Tool, ToolOutput};
use crate::tools::ToolError;

/// A typed handle to a position in a flow graph under construction.
///
/// `O` is the type produced at this point in the chain. Methods consume `self`
/// and return a new `Node` with an updated type parameter, encoding the I/O
/// contract of each step at compile time.
///
/// Obtain a root node via [`Flow::build`] and end
/// the chain at `Node<F::Output>`. [`finalize`](Node::finalize) exists as the
/// low-level bridge back to [`FlowBuilder`] for internal paths.
pub struct Node<O> {
    builder: Arc<Mutex<FlowBuilder>>,
    _marker: PhantomData<fn() -> O>,
}

impl<O> Node<O> {
    fn with_arc(builder: Arc<Mutex<FlowBuilder>>) -> Self {
        Node {
            builder,
            _marker: PhantomData,
        }
    }

    /// Wraps an existing builder into a root node for flow compilation.
    pub(crate) fn from_builder(builder: FlowBuilder) -> Self {
        Node {
            builder: Arc::new(Mutex::new(builder)),
            _marker: PhantomData,
        }
    }

    /// Applies low-level [`FlowBuilder`] operations from this point in the chain.
    ///
    /// Prefer the typed node methods when they cover the shape you need. This
    /// escape hatch exists for advanced or transitional cases that still need
    /// direct builder access.
    pub fn with_builder<P>(self, f: impl FnOnce(FlowBuilder) -> FlowBuilder) -> Node<P> {
        self.mutate_into(f)
    }

    fn mutate(self, f: impl FnOnce(FlowBuilder) -> FlowBuilder) -> Self {
        {
            let mut guard = self.builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = f(b);
        }
        self
    }

    fn mutate_into<P>(self, f: impl FnOnce(FlowBuilder) -> FlowBuilder) -> Node<P> {
        {
            let mut guard = self.builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = f(b);
        }
        Node::with_arc(self.builder)
    }

    /// Adds an async work node. `func` receives `O` and must return `Result<P, FlowError>`.
    pub fn work<P, Fut, H>(self, func: H) -> Node<P>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema,
        P: 'static + Serialize + DeserializeOwned + JsonSchema,
        Fut: Future<Output = Result<P, FlowError>> + Send + 'static,
        H: Fn(O, Context) -> Fut + Send + Sync + 'static,
    {
        self.mutate_into(|b| b.work(func))
    }

    /// Adds a pure synchronous transform node.
    pub fn map<P, H>(self, func: H) -> Node<P>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema,
        P: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(O) -> P + Send + Sync + 'static,
    {
        self.mutate_into(|b| b.map(func))
    }

    /// Adds a suspend point. The flow pauses here and resumes with a value of type `R`.
    pub fn suspend<R>(self) -> Node<R>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema + Send,
        R: 'static + Serialize + DeserializeOwned + JsonSchema,
    {
        self.mutate_into(|b| b.suspend::<O, R>())
    }

    /// Splits the current output into N parallel branches.
    ///
    /// Returns a tuple of `Node`s — one per branch — all sharing the same
    /// underlying builder. Develop each branch independently, then combine them
    /// with [`merge`](Node::merge) or discard sidecars with [`hold`](Node::hold).
    pub fn split<Out, H>(self, func: H) -> Out::Branches
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema,
        Out: SplitOutputs + IntoBranches,
        H: Fn(O) -> Out + Send + Sync + 'static,
    {
        {
            let mut guard = self.builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = b.split(func);
        }
        Out::into_branches(self.builder)
    }

    /// Branches into one of two typed paths.
    ///
    /// Use [`EitherNode::branch`] to define both paths when they later converge
    /// to the same output type, or select one arm with [`EitherNode::left`] or
    /// [`EitherNode::right`] when one branch continues through existing nodes.
    pub fn either<A, B, H>(self, func: H) -> EitherNode<A, B>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema,
        A: 'static + Serialize + DeserializeOwned + JsonSchema,
        B: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(O) -> Either<A, B> + Send + Sync + 'static,
    {
        {
            let mut guard = self.builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = b.either(func);
        }
        EitherNode::with_arc(self.builder)
    }

    /// Merges this branch with one or more sibling branches using `func`.
    ///
    /// `others` may be a single [`Node`] or a tuple of `Node`s. All branches
    /// must originate from the same [`split`](Node::split) call.
    ///
    /// # Errors
    ///
    /// Accumulates a build error if `self` and `others` do not share the same builder.
    pub fn merge<M, Out, H>(self, others: M, func: H) -> Node<Out>
    where
        O: 'static + Serialize + DeserializeOwned + JsonSchema,
        M: MergeBranches<O>,
        M::Inputs: MergeInputs,
        Out: 'static + Serialize + DeserializeOwned + JsonSchema,
        H: Fn(M::Inputs) -> Out + Send + Sync + 'static,
    {
        if !others.same_builder_as(&self) {
            return self.mutate_into(|b| {
                b.error("merge: branches do not share the same builder (originated from different splits)")
            });
        }
        drop(others);
        self.mutate_into(|b| b.merge(func))
    }

    /// Registers an orphan branch without affecting the current output type.
    ///
    /// Orphan branches — those not part of a [`merge`](Node::merge) — should be
    /// passed to `hold` before returning the terminal node from `build` to make
    /// the intent explicit and ensure the `Arc` is dropped cleanly.
    ///
    /// # Errors
    ///
    /// Accumulates a build error if `orphan` does not share the same builder.
    pub fn hold<X>(self, orphan: Node<X>) -> Self {
        if !Arc::ptr_eq(&self.builder, &orphan.builder) {
            drop(orphan);
            return self.mutate(|b| b.error("hold: orphan node does not share the same builder"));
        }
        drop(orphan);
        self
    }

    /// Extracts the completed [`FlowBuilder`] from this node.
    pub fn finalize(self) -> FlowBuilder {
        let mut guard = self.builder.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *guard)
    }
}

/// A typed conditional branch produced by [`Node::either`].
///
/// The left and right branches share the same underlying builder. Use
/// [`branch`](EitherNode::branch) when both arms converge to the same output
/// type. Use [`left`](EitherNode::left) or [`right`](EitherNode::right) when
/// one arm should continue through nodes that are already part of the graph.
pub struct EitherNode<A, B> {
    builder: Arc<Mutex<FlowBuilder>>,
    _marker: PhantomData<fn() -> (A, B)>,
}

impl<A, B> EitherNode<A, B> {
    fn with_arc(builder: Arc<Mutex<FlowBuilder>>) -> Self {
        Self {
            builder,
            _marker: PhantomData,
        }
    }

    /// Continues from the left branch as the current typed node.
    pub fn left(self) -> Node<A> {
        Node::with_arc(self.builder)
    }

    /// Continues from the right branch as the current typed node.
    pub fn right(self) -> Node<B> {
        Node::with_arc(self.builder)
    }

    /// Builds both branches and rejoins them at a common output type.
    pub fn branch<Out, Left, Right>(self, left: Left, right: Right) -> Node<Out>
    where
        Left: FnOnce(Node<A>) -> Node<Out>,
        Right: FnOnce(Node<B>) -> Node<Out>,
    {
        let left_out = left(Node::with_arc(Arc::clone(&self.builder)));
        let right_out = right(Node::with_arc(Arc::clone(&self.builder)));
        let left_ok = Arc::ptr_eq(&left_out.builder, &self.builder);
        let right_ok = Arc::ptr_eq(&right_out.builder, &self.builder);
        drop(left_out);
        drop(right_out);

        if !left_ok || !right_ok {
            let mut guard = self.builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = b.error("either.branch: branch returned a node from a different builder");
        }

        Node::with_arc(self.builder)
    }

    /// Extracts the underlying builder for low-level or internal paths.
    pub fn finalize(self) -> FlowBuilder {
        let mut guard = self.builder.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *guard)
    }
}

/// Scoped agent tool configuration for [`Node::agent_with`].
///
/// `Toolbox<A>` can only attach tools to agent `A`. It is not a graph step and
/// does not change the flow output type; it exists solely to keep tool
/// registration local to the agent that owns those tools.
pub struct Toolbox<A> {
    builder: Arc<Mutex<FlowBuilder>>,
    _marker: PhantomData<fn() -> A>,
}

impl<A> Toolbox<A> {
    fn with_arc(builder: Arc<Mutex<FlowBuilder>>) -> Self {
        Self {
            builder,
            _marker: PhantomData,
        }
    }

    fn mutate(self, f: impl FnOnce(FlowBuilder) -> FlowBuilder) -> Self {
        {
            let mut guard = self.builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = f(b);
        }
        self
    }
}

impl<A: Agent> Toolbox<A> {
    /// Attaches tool `T` to agent `A`.
    pub fn tool<T: Tool>(self) -> Self {
        self.mutate(|b| b.tool::<A, T>())
    }

    /// Attaches a tool whose implementation is built from an arbitrary subgraph.
    ///
    /// The closure starts at `Node<I>` and must return `Node<O>`, allowing the
    /// tool to use any flow composition that is valid in the normal fluent API.
    pub fn tool_with<I, O, Build>(self, build: Build) -> Self
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send,
        O: ToolOutput,
        Build: FnOnce(Node<I>) -> Node<O>,
    {
        let builder = self.builder;
        {
            let mut guard = builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = b.tool_with::<A, I, O>();
        }

        let output = build(Node::with_arc(Arc::clone(&builder)));
        let same_builder = Arc::ptr_eq(&output.builder, &builder);
        drop(output);

        if !same_builder {
            let mut guard = builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = b.error("tool_with: tool returned a node from a different builder");
        }

        Toolbox::with_arc(builder)
    }

    /// Attaches a low-level one-shot tool handler with explicit input and output types.
    pub fn tool_handler<I, O, Fut, H>(self, func: H) -> Self
    where
        I: 'static + Serialize + DeserializeOwned + JsonSchema + Send,
        O: ToolOutput,
        Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
        H: Fn(I, Context) -> Fut + Send + Sync + 'static,
    {
        self.mutate(|b| b.tool_with_handler::<A, I, O, Fut, H>(func))
    }

    /// Attaches sub-flow `F` as a tool to agent `A`.
    pub fn tool_flow<F: Flow>(self) -> Self
    where
        F::Output: ToolOutput,
    {
        self.tool_with::<F, F::Output, _>(|tool| tool.flow())
    }
}

/// Methods available when the current output `O` itself implements [`Agent`].
///
/// This enforces at compile time that `.agent()` is only callable when the
/// node's current type matches the agent's expected input.
impl<O: Agent> Node<O> {
    /// Runs agent `O` with no tools, advancing the chain to `O::Output`.
    pub fn agent(self) -> Node<O::Output> {
        self.agent_with(|toolbox| toolbox)
    }

    /// Runs agent `O` and configures its tools within a scoped [`Toolbox`].
    pub fn agent_with(self, build_tools: impl FnOnce(Toolbox<O>) -> Toolbox<O>) -> Node<O::Output> {
        let builder = self.builder;
        {
            let mut guard = builder.lock().unwrap_or_else(|e| e.into_inner());
            let b = std::mem::take(&mut *guard);
            *guard = b.agent::<O>();
        }
        drop(build_tools(Toolbox::with_arc(Arc::clone(&builder))));
        Node::with_arc(builder)
    }
}

/// Methods available when the current output `O` itself implements [`Flow`].
impl<O: Flow> Node<O> {
    /// Embeds child flow `O`, advancing the chain to `O::Output`.
    pub fn flow(self) -> Node<O::Output> {
        self.mutate_into(|b| b.flow::<O>())
    }
}

/// Methods available when the current output is `Vec<F>` for some [`Flow`] `F`.
impl<F: Flow> Node<Vec<F>>
where
    Vec<F>: JsonSchema + 'static,
    Vec<F::Output>: JsonSchema + 'static,
{
    /// Fans out over `Vec<F>`, running sub-flow `F` for each element sequentially.
    ///
    /// Output advances to `Vec<F::Output>`.
    pub fn each(self) -> Node<Vec<F::Output>> {
        self.mutate_into(|b| b.each::<F>())
    }
}

/// Helper for [`Node::merge`].
///
/// Implemented for a single sibling [`Node`] and tuples of sibling `Node`s.
#[doc(hidden)]
pub trait MergeBranches<Head>: sealed::Sealed {
    type Inputs;

    fn same_builder_as(&self, node: &Node<Head>) -> bool;
}

/// Converts a split output tuple type into a tuple of [`Node`]s sharing the same builder.
///
/// Implemented for arities 2–8. Sealed: external crates cannot implement it.
pub trait IntoBranches: sealed::Sealed {
    type Branches;
    fn into_branches(builder: Arc<Mutex<FlowBuilder>>) -> Self::Branches;
}

mod sealed {
    pub trait Sealed {}
}

impl<Branch> sealed::Sealed for Node<Branch> {}

impl<Head, Branch> MergeBranches<Head> for Node<Branch> {
    type Inputs = (Head, Branch);

    fn same_builder_as(&self, node: &Node<Head>) -> bool {
        Arc::ptr_eq(&self.builder, &node.builder)
    }
}

macro_rules! impl_into_branches {
    ($($T:ident),+) => {
        impl<$($T),+> sealed::Sealed for ($($T,)+) {}

        impl<$($T),+> IntoBranches for ($($T,)+) {
            type Branches = ($(Node<$T>,)+);

            fn into_branches(builder: Arc<Mutex<FlowBuilder>>) -> Self::Branches {
                ($( Node::<$T>::with_arc(Arc::clone(&builder)), )+)
            }
        }
    };
}

macro_rules! impl_merge_branches {
    ($($T:ident : $idx:tt),+) => {
        impl<Head, $($T),+> MergeBranches<Head> for ($(Node<$T>,)+) {
            type Inputs = (Head, $($T,)+);

            fn same_builder_as(&self, node: &Node<Head>) -> bool {
                true $(&& Arc::ptr_eq(&self.$idx.builder, &node.builder))+
            }
        }
    };
}

impl_into_branches!(A, B);
impl_into_branches!(A, B, C);
impl_into_branches!(A, B, C, D);
impl_into_branches!(A, B, C, D, E);
impl_into_branches!(A, B, C, D, E, F);
impl_into_branches!(A, B, C, D, E, F, G);
impl_into_branches!(A, B, C, D, E, F, G, H);

impl_merge_branches!(A:0, B:1);
impl_merge_branches!(A:0, B:1, C:2);
impl_merge_branches!(A:0, B:1, C:2, D:3);
impl_merge_branches!(A:0, B:1, C:2, D:3, E:4);
impl_merge_branches!(A:0, B:1, C:2, D:3, E:4, F:5);
impl_merge_branches!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
impl_merge_branches!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
