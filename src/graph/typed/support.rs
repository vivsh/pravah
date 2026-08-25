use super::*;

impl TypedBuildState {
    pub(super) fn next_handler_key(&mut self, name: &str) -> HandlerKey {
        let key = HandlerKey::new(format!(
            "typed::{}::{}::{}",
            self.graph_name, self.handler_counter, name
        ));
        self.handler_counter = self.handler_counter.saturating_add(1);
        key
    }

    pub(super) fn next_handler_namespace(&mut self, name: &str) -> String {
        let namespace = format!("{}::{}::{name}", self.graph_name, self.handler_counter);
        self.handler_counter = self.handler_counter.saturating_add(1);
        namespace
    }
}

pub(super) fn namespace_graph_handlers(graph: &mut UntypedGraph, prefix: &str) {
    for node in &mut graph.nodes {
        match &mut node.kind {
            NodeKind::PureHandler { key }
            | NodeKind::WorkHandler { key }
            | NodeKind::Load { key, .. }
            | NodeKind::Store { key, .. } => namespace_handler_key(key, prefix),
            NodeKind::Continuation { key, children, .. } => {
                namespace_handler_key(key, prefix);
                for child in children {
                    namespace_graph_handlers(child, prefix);
                }
            }
            NodeKind::Subflow { graph } | NodeKind::Each { graph } => {
                namespace_graph_handlers(graph, prefix);
            }
            NodeKind::Either { key, left, right } => {
                namespace_handler_key(key, prefix);
                namespace_graph_handlers(left, prefix);
                namespace_graph_handlers(right, prefix);
            }
            NodeKind::Builtin { .. } | NodeKind::Suspend { .. } | NodeKind::Goto { .. } => {}
        }
    }
}

pub(super) fn namespace_handler_key(key: &mut HandlerKey, prefix: &str) {
    *key = HandlerKey::new(format!("{prefix}::{}", key.as_str()));
}

pub(super) fn with_state<R>(
    state: &Arc<Mutex<TypedBuildState>>,
    f: impl FnOnce(&mut TypedBuildState) -> R,
) -> Option<R> {
    match state.lock() {
        Ok(mut guard) => Some(f(&mut guard)),
        Err(_) => None,
    }
}

pub(super) fn push_error(state: &Arc<Mutex<TypedBuildState>>, msg: impl Into<String>) {
    let msg = msg.into();
    let _ = with_state(state, |guard| guard.errors.push(msg));
}

pub(super) fn variable_key<S>() -> VarKey
where
    S: JsonSchema,
{
    VarKey::new("rust", S::schema_name())
}

pub(super) fn compile_inline_flow<I, O, Build>(
    name: impl Into<String>,
    build: Build,
) -> Result<CompiledFlow<I, O>, GraphError>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    O: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    Build: FnOnce(Flow<I>) -> Flow<O>,
{
    build(Flow::<I>::root_named(name.into())).finish::<I>()
}

pub(super) fn type_spec<T>() -> TypeSpec
where
    T: JsonSchema,
{
    let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
    let schema = match serde_json::to_value(schema_gen.root_schema_for::<T>()) {
        Ok(schema) => schema,
        Err(err) => serde_json::Value::String(format!("schema generation failed: {err}")),
    };
    TypeSpec::new(T::schema_name(), schema)
}

pub(super) fn decode_one<T>(inputs: Vec<Value>, node: &str) -> Result<T, GraphError>
where
    T: DeserializeOwned + JsonSchema,
{
    if inputs.len() != 1 {
        return Err(GraphError::OutputArity {
            node: node.into(),
            expected: 1,
            got: inputs.len(),
        });
    }
    let mut iter = inputs.into_iter();
    let value = iter
        .next()
        .ok_or_else(|| GraphError::Invalid(format!("node '{node}' received no input value")))?;
    from_value(value).map_err(|err| {
        GraphError::Invalid(format!(
            "node '{node}' failed to decode '{}': {err}",
            T::schema_name()
        ))
    })
}

pub(super) fn decode_pair<A, B>(inputs: Vec<Value>, node: &str) -> Result<(A, B), GraphError>
where
    A: DeserializeOwned + JsonSchema,
    B: DeserializeOwned + JsonSchema,
{
    if inputs.len() != 2 {
        return Err(GraphError::OutputArity {
            node: node.into(),
            expected: 2,
            got: inputs.len(),
        });
    }
    let mut iter = inputs.into_iter();
    let first = iter
        .next()
        .ok_or_else(|| GraphError::Invalid(format!("node '{node}' received no input value")))?;
    let second = iter
        .next()
        .ok_or_else(|| GraphError::Invalid(format!("node '{node}' received no state value")))?;
    let first = from_value(first).map_err(|err| {
        GraphError::Invalid(format!(
            "node '{node}' failed to decode '{}': {err}",
            A::schema_name()
        ))
    })?;
    let second = from_value(second).map_err(|err| {
        GraphError::Invalid(format!(
            "node '{node}' failed to decode '{}': {err}",
            B::schema_name()
        ))
    })?;
    Ok((first, second))
}

pub(super) fn encode_one<T>(
    value: Result<T, GraphError>,
    node: &str,
) -> Result<Vec<Value>, GraphError>
where
    T: Serialize + JsonSchema,
{
    let value = value?;
    Ok(vec![encode_value(value, node)?])
}

pub(super) fn encode_value<T>(value: T, node: &str) -> Result<Value, GraphError>
where
    T: Serialize + JsonSchema,
{
    to_value(value).map_err(|err| {
        GraphError::Invalid(format!(
            "node '{node}' failed to encode '{}': {err}",
            T::schema_name()
        ))
    })
}
