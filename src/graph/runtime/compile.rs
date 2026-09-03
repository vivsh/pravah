use super::*;
use crate::graph::Value;

pub(super) fn compile_graph(
    graph: UntypedGraph,
    callables: &mut Vec<CompiledGraph>,
) -> Result<usize, GraphError> {
    compile_graph_at(graph, GraphPath::root(), callables)
}

fn compile_graph_at(
    graph: UntypedGraph,
    path: GraphPath,
    callables: &mut Vec<CompiledGraph>,
) -> Result<usize, GraphError> {
    let mut child_indices = vec![CompiledChildren::default(); graph.nodes.len()];
    for node in &graph.nodes {
        let slot = child_indices
            .get_mut(node.id.0)
            .ok_or(GraphError::MissingNode(node.id))?;
        match &node.kind {
            NodeKind::Subflow { graph: child } => {
                let child_path = path.child(CallSite::Subflow { node: node.id });
                slot.primary = Some(compile_graph_at((**child).clone(), child_path, callables)?);
            }
            NodeKind::Each { graph: child } => {
                let child_path = path.child(CallSite::Each { node: node.id });
                slot.primary = Some(compile_graph_at((**child).clone(), child_path, callables)?);
            }
            NodeKind::Continuation { children, .. } => {
                for (child_index, child) in children.iter().enumerate() {
                    let child_path = path.child(CallSite::Continuation {
                        node: node.id,
                        child_index,
                    });
                    slot.continuation
                        .push(compile_graph_at(child.clone(), child_path, callables)?);
                }
            }
            NodeKind::Either { left, right, .. } => {
                let left_path = path.child(CallSite::EitherLeft { node: node.id });
                let right_path = path.child(CallSite::EitherRight { node: node.id });
                slot.left = Some(compile_graph_at((**left).clone(), left_path, callables)?);
                slot.right = Some(compile_graph_at((**right).clone(), right_path, callables)?);
            }
            NodeKind::Builtin { .. }
            | NodeKind::PureHandler { .. }
            | NodeKind::WorkHandler { .. }
            | NodeKind::Suspend { .. }
            | NodeKind::Load { .. }
            | NodeKind::Store { .. }
            | NodeKind::Goto { .. } => {}
        }
    }
    let dce = dce::prepare_dce(&graph)?;
    let liveness = liveness::prepare_liveness(&graph, &dce)?;
    let nodes = compile_nodes(&graph, &child_indices, &liveness, &dce)?;
    let index = callables.len();
    let mut inheritable_by_key = HashMap::new();
    for variable in &graph.variables {
        inheritable_by_key.insert(variable.key.clone(), variable.id);
    }
    callables.push(CompiledGraph {
        path,
        graph: Arc::new(graph),
        nodes,
        instructions: Arc::clone(&dce.instructions),
        child_indices,
        inheritable_by_key,
        liveness,
    });
    Ok(index)
}

pub(super) fn compile_nodes(
    graph: &UntypedGraph,
    child_indices: &[CompiledChildren],
    liveness: &LivenessPlan,
    dce: &DcePlan,
) -> Result<Vec<CompiledNode>, GraphError> {
    let mut nodes = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let children = child_indices
            .get(node.id.0)
            .ok_or(GraphError::MissingNode(node.id))?;
        let kind = match &node.kind {
            NodeKind::Builtin { op } => CompiledNodeKind::Builtin { op: op.clone() },
            NodeKind::PureHandler { key } => CompiledNodeKind::PureHandler { key: key.clone() },
            NodeKind::WorkHandler { key } => CompiledNodeKind::WorkHandler { key: key.clone() },
            NodeKind::Load { var, key } => CompiledNodeKind::Load {
                var: *var,
                key: key.clone(),
            },
            NodeKind::Store { var, key } => CompiledNodeKind::Store {
                var: *var,
                key: key.clone(),
            },
            NodeKind::Continuation { key, payload, .. } => CompiledNodeKind::Continuation {
                key: key.clone(),
                payload: Arc::new(payload.clone()),
                children: Arc::from(children.continuation.clone().into_boxed_slice()),
            },
            NodeKind::Suspend { payload, .. } => CompiledNodeKind::Suspend {
                payload: Arc::new(payload.clone()),
            },
            NodeKind::Subflow { .. } => CompiledNodeKind::Subflow {
                child_index: children.primary.ok_or_else(|| {
                    GraphError::Invalid(format!(
                        "subflow node '{}' has no compiled child graph",
                        node.name
                    ))
                })?,
            },
            NodeKind::Either { key, .. } => CompiledNodeKind::Either {
                key: key.clone(),
                left_index: children.left.ok_or_else(|| {
                    GraphError::Invalid(format!(
                        "either node '{}' has no compiled left branch",
                        node.name
                    ))
                })?,
                right_index: children.right.ok_or_else(|| {
                    GraphError::Invalid(format!(
                        "either node '{}' has no compiled right branch",
                        node.name
                    ))
                })?,
            },
            NodeKind::Each { .. } => CompiledNodeKind::Each {
                child_index: children.primary.ok_or_else(|| {
                    GraphError::Invalid(format!(
                        "each node '{}' has no compiled child graph",
                        node.name
                    ))
                })?,
            },
            NodeKind::Goto { mark } => {
                let target = graph
                    .mark(*mark)
                    .ok_or_else(|| {
                        GraphError::Invalid(format!(
                            "goto node '{}' references missing mark {:?}",
                            node.name, mark
                        ))
                    })?
                    .target;
                CompiledNodeKind::Goto { target }
            }
        };
        let can_continue = matches!(
            kind,
            CompiledNodeKind::Continuation { .. } | CompiledNodeKind::Each { .. }
        );
        let can_suspend = matches!(
            kind,
            CompiledNodeKind::Suspend { .. } | CompiledNodeKind::Continuation { .. }
        );
        nodes.push(CompiledNode {
            id: node.id,
            name: Arc::from(node.name.as_str()),
            inputs: Arc::from(node.inputs.clone().into_boxed_slice()),
            outputs: Arc::from(node.outputs.clone().into_boxed_slice()),
            kind,
            can_continue,
            can_suspend,
            release_actions: if dce.is_active(node.id) {
                liveness::release_actions(node, liveness)?
            } else {
                Arc::from([])
            },
        });
    }
    Ok(nodes)
}

pub(super) fn new_frame(
    callables: &[CompiledGraph],
    parent_frames: &[Frame],
    graph_index: usize,
    return_target: Option<ReturnTarget>,
) -> Result<Frame, GraphError> {
    let graph = callables
        .get(graph_index)
        .ok_or_else(|| GraphError::Invalid(format!("graph index {graph_index} is invalid")))?;
    let mut variables = Vec::with_capacity(graph.graph.variables.len());
    for variable in &graph.graph.variables {
        let initialized = initial_variable_value(callables, parent_frames, variable)?;
        let value = if matches!(
            graph.liveness.variables.get(variable.id.0),
            Some(liveness::Retention::Dead)
        ) {
            None
        } else {
            initialized
        };
        variables.push(value);
    }
    let has_initialized_variable = variables.iter().any(Option::is_some);
    let initial_epoch = u64::from(has_initialized_variable);
    let variable_epochs = variables
        .iter()
        .map(|value| u64::from(value.is_some()))
        .collect();
    Ok(Frame {
        graph_index,
        values: vec![None; graph.graph.edges.len()],
        write_epoch: initial_epoch,
        edge_epochs: vec![0; graph.graph.edges.len()],
        variables,
        variable_epochs,
        checkpoints: vec![None; graph.graph.nodes.len()],
        continuation_states: vec![None; graph.graph.nodes.len()],
        continuation_inboxes: vec![Vec::new(); graph.graph.nodes.len()],
        continuation_child_queues: vec![Vec::new(); graph.graph.nodes.len()],
        node_epochs: vec![0; graph.graph.nodes.len()],
        reader_counts: graph.liveness.initial_counters()?,
        return_target,
    })
}

pub(super) fn initial_variable_value(
    callables: &[CompiledGraph],
    parent_frames: &[Frame],
    variable: &Variable,
) -> Result<Option<Value>, GraphError> {
    if variable.scope == VarScope::Inherit
        && let Some(value) = inherited_variable_value(callables, parent_frames, variable)?
    {
        return Ok(Some(value));
    }
    default_variable_value(variable)
}

pub(super) fn inherited_variable_value(
    callables: &[CompiledGraph],
    parent_frames: &[Frame],
    variable: &Variable,
) -> Result<Option<Value>, GraphError> {
    for frame in parent_frames.iter().rev() {
        let compiled = callables
            .get(frame.graph_index)
            .ok_or_else(|| GraphError::Invalid("parent frame graph index is invalid".into()))?;
        let Some(parent_var) = compiled.inheritable_by_key.get(&variable.key) else {
            continue;
        };
        let value = frame
            .variables
            .get(parent_var.0)
            .ok_or(GraphError::MissingVariable(*parent_var))?;
        if let Some(value) = value {
            validate_value(
                &variable.type_spec,
                value,
                &format!(
                    "inherited variable '{}::{}'",
                    variable.key.namespace, variable.key.type_name
                ),
            )?;
            return Ok(Some(value.clone()));
        }
    }
    Ok(None)
}

pub(super) fn default_variable_value(variable: &Variable) -> Result<Option<Value>, GraphError> {
    match &variable.init {
        VarInit::Value(value) => {
            validate_value(
                &variable.type_spec,
                value,
                &format!(
                    "initial variable '{}::{}'",
                    variable.key.namespace, variable.key.type_name
                ),
            )?;
            Ok(Some(value.clone()))
        }
        VarInit::Uninitialized => Ok(None),
    }
}

pub(super) fn validate_edge_value(
    graph: &UntypedGraph,
    edge: EdgeId,
    value: &Value,
    label: &str,
) -> Result<(), GraphError> {
    let edge_data = graph.edge(edge).ok_or(GraphError::MissingEdge(edge))?;
    let label = edge_data.label.as_ref().map_or_else(
        || format!("{label} {edge:?}"),
        |edge_label| format!("{label} edge '{edge_label}'"),
    );
    validate_value(&edge_data.type_spec, value, &label)
}

pub(super) fn suspend_payload(configured: &Value, input: &Value) -> Value {
    if configured.is_null() {
        input.clone()
    } else {
        configured.clone()
    }
}
