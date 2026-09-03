use super::*;

pub(super) fn declare_variable<S>(
    state: &Arc<Mutex<TypedBuildState>>,
    value: S,
    scope: VarScope,
) -> TypedVar<S>
where
    S: 'static + Serialize + JsonSchema,
{
    let var = with_state(state, |guard| {
        let key = variable_key::<S>();
        if guard.variables.contains_key(&key) {
            guard.errors.push(format!(
                "variable '{}::{}' is already declared",
                key.namespace, key.type_name
            ));
            return guard.variables.get(&key).copied();
        }
        let value = match to_value(value) {
            Ok(value) => value,
            Err(err) => {
                guard
                    .errors
                    .push(format!("failed to serialize variable value: {err}"));
                return None;
            }
        };
        let id =
            guard
                .builder
                .variable(key.clone(), type_spec::<S>(), scope, VarInit::Value(value));
        guard.variables.insert(key, id);
        Some(id)
    })
    .flatten();
    typed_var(Arc::clone(state), var)
}

pub(super) fn add_mark<T>(state: Arc<Mutex<TypedBuildState>>, input_edge: EdgeId) -> TypedMark<T>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let mark = with_state(&state, |guard| {
        Some(
            guard
                .builder
                .mark_with_label(Some(format!("mark_{}", T::schema_name())), input_edge),
        )
    })
    .flatten();
    typed_mark(state, mark)
}

pub(super) fn add_goto_node<T>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    mark: Option<MarkId>,
) -> TypedEdge<T>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let Some(mark) = mark else {
        push_error(
            &state,
            format!("goto: mark for '{}' is not available", T::schema_name()),
        );
        return typed_edge(state, input_edge);
    };
    with_state(&state, |guard| {
        guard
            .builder
            .goto(format!("goto_{}", T::schema_name()), input_edge, mark);
    });
    typed_edge(state, input_edge)
}

pub(super) fn add_map_node<T, P, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    func: H,
) -> TypedEdge<P>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    H: Fn(T) -> P + Send + Sync + 'static,
{
    let name = format!("map_to_{}", P::schema_name());
    add_map_node_named(state, input_edge, name, func)
}

pub(super) fn add_map_node_named<T, P, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    name: String,
    func: H,
) -> TypedEdge<P>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    H: Fn(T) -> P + Send + Sync + 'static,
{
    let output = with_state(&state, |guard| {
        let key = guard.next_handler_key(&name);
        let handler_name = name.clone();
        if let Err(err) = guard.registry.insert_value(key.as_str(), move |inputs| {
            let input = decode_one::<T>(inputs, &handler_name)?;
            encode_one(Ok(func(input)), &handler_name)
        }) {
            guard.errors.push(err.to_string());
        }
        let output = guard.builder.edge(format!("{name}_out"), type_spec::<P>());
        guard.builder.node(
            name,
            NodeKind::PureHandler { key },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_split_node<T, Out, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    func: H,
) -> Out::Flows
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Out: SplitOutputs,
    H: Fn(T) -> Out + Send + Sync + 'static,
{
    let schema_names = Out::schema_names();
    let name = format!("split_to_{}", schema_names.join("_"));
    let edges = with_state(&state, |guard| {
        let key = guard.next_handler_key(&name);
        let handler_name = name.clone();
        if let Err(err) = guard.registry.insert_value(key.as_str(), move |inputs| {
            let input = decode_one::<T>(inputs, &handler_name)?;
            func(input).encode_outputs(&handler_name)
        }) {
            guard.errors.push(err.to_string());
        }
        let outputs = Out::type_specs()
            .into_iter()
            .enumerate()
            .map(|(index, type_spec)| guard.builder.edge(format!("{name}_{index}"), type_spec))
            .collect::<Vec<_>>();
        guard.builder.node(
            name,
            NodeKind::PureHandler { key },
            vec![input_edge],
            outputs.clone(),
        );
        outputs
    });
    match edges {
        Some(edges) => Out::into_flows(state, edges),
        None => Out::fallback_flows(state, input_edge),
    }
}

pub(super) fn add_merge_node<Inputs, Out, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edges: Vec<EdgeId>,
    func: H,
) -> TypedEdge<Out>
where
    Inputs: 'static + DeserializeOwned,
    Out: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    H: Fn(Inputs) -> Out + Send + Sync + 'static,
{
    let name = format!("merge_to_{}", Out::schema_name());
    let output = with_state(&state, |guard| {
        let key = guard.next_handler_key(&name);
        let handler_name = name.clone();
        if let Err(err) = guard.registry.insert_value(key.as_str(), move |inputs| {
            let inputs = from_value::<Inputs>(Value::array(inputs)).map_err(|err| {
                GraphError::Invalid(format!(
                    "node '{handler_name}' failed to decode merge inputs: {err}"
                ))
            })?;
            encode_one(Ok(func(inputs)), &handler_name)
        }) {
            guard.errors.push(err.to_string());
        }
        let output = guard
            .builder
            .edge(format!("{name}_out"), type_spec::<Out>());
        guard.builder.node(
            name,
            NodeKind::PureHandler { key },
            input_edges,
            vec![output],
        );
        output
    })
    .unwrap_or(EdgeId(0));
    typed_edge(state, output)
}

pub(super) fn add_work_node<T, P, Fut, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    func: H,
) -> TypedEdge<P>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Fut: Future<Output = Result<P, GraphError>> + Send + 'static,
    H: Fn(T, Context) -> Fut + Send + Sync + 'static,
{
    let name = format!("work_to_{}", P::schema_name());
    add_work_node_named(state, input_edge, name, func)
}

pub(super) fn add_work_node_named<T, P, Fut, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    name: String,
    func: H,
) -> TypedEdge<P>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Fut: Future<Output = Result<P, GraphError>> + Send + 'static,
    H: Fn(T, Context) -> Fut + Send + Sync + 'static,
{
    let func = Arc::new(func);
    let output = with_state(&state, |guard| {
        let key = guard.next_handler_key(&name);
        let handler_name = name.clone();
        if let Err(err) = guard
            .registry
            .insert_work(key.as_str(), move |inputs, ctx| {
                let func = Arc::clone(&func);
                let handler_name = handler_name.clone();
                let fut: BoxFuture<'static, Result<Vec<Value>, GraphError>> =
                    Box::pin(async move {
                        let input = decode_one::<T>(inputs, &handler_name)?;
                        encode_one(func(input, ctx).await, &handler_name)
                    });
                fut
            })
        {
            guard.errors.push(err.to_string());
        }
        let output = guard.builder.edge(format!("{name}_out"), type_spec::<P>());
        guard.builder.node(
            name,
            NodeKind::WorkHandler { key },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_continuation_node<O, H, P>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    payload: P,
) -> TypedEdge<O>
where
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    H: ContinuationHandler + Default + 'static,
    P: Serialize,
{
    let name = format!("continuation_to_{}", O::schema_name());
    let output = with_state(&state, |guard| {
        let payload = match to_value(payload) {
            Ok(payload) => payload,
            Err(err) => {
                guard
                    .errors
                    .push(format!("failed to serialize continuation payload: {err}"));
                Value::NULL
            }
        };
        let key = guard.next_handler_key(&format!("continuation_{}", std::any::type_name::<H>()));
        if let Err(err) = guard
            .registry
            .insert_continuation(key.as_str(), H::default())
        {
            guard.errors.push(err.to_string());
        }
        let output = guard.builder.edge(format!("{name}_out"), type_spec::<O>());
        guard.builder.node(
            name,
            NodeKind::Continuation {
                key,
                payload,
                children: Vec::new(),
            },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_suspend_node<R>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
) -> TypedEdge<R>
where
    R: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let name = format!("suspend_to_{}", R::schema_name());
    let output = with_state(&state, |guard| {
        let output = guard.builder.edge(format!("{name}_out"), type_spec::<R>());
        guard.builder.node(
            name,
            NodeKind::Suspend {
                resume_type: R::schema_name(),
                payload: Value::NULL,
            },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_load_node<T, S, O, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    var_id: Option<VarId>,
    func: H,
) -> TypedEdge<O>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: 'static + Serialize + DeserializeOwned + JsonSchema,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    H: Fn(T, S) -> O + Send + Sync + 'static,
{
    add_load_node_named(
        state,
        input_edge,
        var_id,
        format!("load_{}", S::schema_name()),
        func,
    )
}

pub(super) fn add_load_node_named<T, S, O, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    var_id: Option<VarId>,
    name: String,
    func: H,
) -> TypedEdge<O>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: 'static + Serialize + DeserializeOwned + JsonSchema,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    H: Fn(T, S) -> O + Send + Sync + 'static,
{
    let Some(var_id) = var_id else {
        push_error(
            &state,
            format!("load: variable '{}' is not available", S::schema_name()),
        );
        return typed_edge(state, input_edge);
    };
    let output = with_state(&state, |guard| {
        let key = guard.next_handler_key(&name);
        let handler_name = name.clone();
        if let Err(err) = guard.registry.insert_value(key.as_str(), move |inputs| {
            let (input, state) = decode_pair::<T, S>(inputs, &handler_name)?;
            encode_one(Ok(func(input, state)), &handler_name)
        }) {
            guard.errors.push(err.to_string());
        }
        let output = guard.builder.edge(format!("{name}_out"), type_spec::<O>());
        guard.builder.node(
            name,
            NodeKind::Load { var: var_id, key },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_store_node<T, S, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    var_id: Option<VarId>,
    func: H,
) -> TypedEdge<T>
where
    T: 'static + Clone + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: 'static + Serialize + DeserializeOwned + JsonSchema,
    H: Fn(T, S) -> S + Send + Sync + 'static,
{
    add_store_node_named(
        state,
        input_edge,
        var_id,
        format!("store_{}", S::schema_name()),
        func,
    )
}

pub(super) fn add_store_node_named<T, S, H>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    var_id: Option<VarId>,
    name: String,
    func: H,
) -> TypedEdge<T>
where
    T: 'static + Clone + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    S: 'static + Serialize + DeserializeOwned + JsonSchema,
    H: Fn(T, S) -> S + Send + Sync + 'static,
{
    let Some(var_id) = var_id else {
        push_error(
            &state,
            format!("store: variable '{}' is not available", S::schema_name()),
        );
        return typed_edge(state, input_edge);
    };
    let output = with_state(&state, |guard| {
        let key = guard.next_handler_key(&name);
        let handler_name = name.clone();
        if let Err(err) = guard.registry.insert_value(key.as_str(), move |inputs| {
            let (input, state) = decode_pair::<T, S>(inputs, &handler_name)?;
            encode_one(Ok(func(input, state)), &handler_name)
        }) {
            guard.errors.push(err.to_string());
        }
        let output = guard.builder.edge(format!("{name}_out"), type_spec::<T>());
        guard.builder.node(
            name,
            NodeKind::Store { var: var_id, key },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_flow_node<I, O>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    flow: fn(Flow<I>) -> Flow<O>,
) -> TypedEdge<O>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    match compile(flow) {
        Ok(child) => add_compiled_flow_node::<I, O>(state, input_edge, &child),
        Err(err) => {
            push_error(
                &state,
                format!("flow '{} -> {}': {err}", I::schema_name(), O::schema_name()),
            );
            typed_edge(state, input_edge)
        }
    }
}

pub(super) fn add_compiled_flow_node<T, P>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    flow: &CompiledFlow<T, P>,
) -> TypedEdge<P>
where
    P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let name = format!("call_{}", P::schema_name());
    let output = with_state(&state, |guard| {
        let namespace = guard.next_handler_namespace(&name);
        let mut graph = flow.graph().clone();
        namespace_graph_handlers(&mut graph, &namespace);
        guard
            .registry
            .extend_namespaced(&namespace, flow.registry());
        let output = guard.builder.edge(format!("{name}_out"), type_spec::<P>());
        guard.builder.node(
            name,
            NodeKind::Subflow {
                graph: Box::new(graph),
            },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_each_node<I, O>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    flow: fn(Flow<I>) -> Flow<O>,
) -> TypedEdge<Vec<O>>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Vec<I>: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Vec<O>: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let child = compile(flow);
    let output = with_state(&state, |guard| {
        let child = match child {
            Ok(flow) => flow,
            Err(err) => {
                guard.errors.push(format!(
                    "each '{} -> {}': {err}",
                    I::schema_name(),
                    O::schema_name()
                ));
                return input_edge;
            }
        };
        let name = format!("each_{}_to_{}", I::schema_name(), O::schema_name());
        let namespace = guard.next_handler_namespace(&name);
        let mut graph = child.graph().clone();
        namespace_graph_handlers(&mut graph, &namespace);
        guard
            .registry
            .extend_namespaced(&namespace, child.registry());
        let output = guard
            .builder
            .edge(format!("{name}_out"), type_spec::<Vec<O>>());
        guard.builder.node(
            name,
            NodeKind::Each {
                graph: Box::new(graph),
            },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_agent_node<I, O>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    agent: Agent<O>,
) -> TypedEdge<O>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    let mut build = build_agent::<I, O>(agent);
    let output = with_state(&state, |guard| {
        if !build.errors.is_empty() {
            guard.errors.extend(build.errors.clone());
            return input_edge;
        }
        for (index, (graph, registry)) in build
            .children
            .iter_mut()
            .zip(build.registries.iter())
            .enumerate()
        {
            let namespace = guard.next_handler_namespace(&format!("agent_tool_{index}"));
            namespace_graph_handlers(graph, &namespace);
            guard.registry.extend_namespaced(&namespace, registry);
        }
        let name = format!("agent_to_{}", O::schema_name());
        let key = guard.next_handler_key(&name);
        build.payload.agent_id = key.as_str().to_owned();
        build.payload.configure_handler_key = key.as_str().to_owned();
        if build.payload.control_handler_key.is_some() {
            build.payload.control_handler_key = Some(format!("{}::control", key.as_str()));
        }
        let payload = match to_value(&build.payload) {
            Ok(payload) => payload,
            Err(err) => {
                guard
                    .errors
                    .push(format!("failed to serialize agent payload: {err}"));
                return input_edge;
            }
        };
        if let Err(err) = guard
            .registry
            .insert_continuation(key.as_str(), build.handler)
        {
            guard.errors.push(err.to_string());
            return input_edge;
        }
        let output = guard.builder.edge(format!("{name}_out"), type_spec::<O>());
        guard.builder.node(
            name,
            NodeKind::Continuation {
                key,
                payload,
                children: build.children,
            },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}

pub(super) fn add_either_node<T, A, B, Out, H, Left, Right>(
    state: Arc<Mutex<TypedBuildState>>,
    input_edge: EdgeId,
    choose: H,
    left: Left,
    right: Right,
) -> TypedEdge<Out>
where
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    A: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    B: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Out: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    H: Fn(T) -> Either<A, B> + Send + Sync + 'static,
    Left: FnOnce(Flow<A>) -> Flow<Out>,
    Right: FnOnce(Flow<B>) -> Flow<Out>,
{
    let left_flow = compile_inline_flow::<A, Out, _>("either_left", left);
    let right_flow = compile_inline_flow::<B, Out, _>("either_right", right);
    let output = with_state(&state, |guard| {
        let left_flow = match left_flow {
            Ok(flow) => flow,
            Err(err) => {
                guard.errors.push(format!("either left branch: {err}"));
                return input_edge;
            }
        };
        let right_flow = match right_flow {
            Ok(flow) => flow,
            Err(err) => {
                guard.errors.push(format!("either right branch: {err}"));
                return input_edge;
            }
        };
        let name = format!("either_to_{}", Out::schema_name());
        let key = guard.next_handler_key(&name);
        let handler_name = name.clone();
        if let Err(err) = guard.registry.insert_value(key.as_str(), move |inputs| {
            let input = decode_one::<T>(inputs, &handler_name)?;
            let choice = match choose(input) {
                Either::Left(value) => BranchChoice {
                    side: BranchSide::Left,
                    value: encode_value(value, &handler_name)?,
                },
                Either::Right(value) => BranchChoice {
                    side: BranchSide::Right,
                    value: encode_value(value, &handler_name)?,
                },
            };
            Ok(vec![to_value(choice).map_err(|err| {
                GraphError::Invalid(format!(
                    "either node '{handler_name}' failed to encode branch choice: {err}"
                ))
            })?])
        }) {
            guard.errors.push(err.to_string());
        }
        let left_namespace = guard.next_handler_namespace(&format!("{name}_left"));
        let right_namespace = guard.next_handler_namespace(&format!("{name}_right"));
        let mut left_graph = left_flow.graph().clone();
        let mut right_graph = right_flow.graph().clone();
        namespace_graph_handlers(&mut left_graph, &left_namespace);
        namespace_graph_handlers(&mut right_graph, &right_namespace);
        guard
            .registry
            .extend_namespaced(&left_namespace, left_flow.registry());
        guard
            .registry
            .extend_namespaced(&right_namespace, right_flow.registry());
        let output = guard
            .builder
            .edge(format!("{name}_out"), type_spec::<Out>());
        guard.builder.node(
            name,
            NodeKind::Either {
                key,
                left: Box::new(left_graph),
                right: Box::new(right_graph),
            },
            vec![input_edge],
            vec![output],
        );
        output
    })
    .unwrap_or(input_edge);
    typed_edge(state, output)
}
