use super::*;
use crate::graph::Value;

pub(super) fn run_builtin(
    op: &BuiltinNode,
    inputs: Vec<Value>,
    output_count: usize,
    node_name: &str,
) -> Result<Vec<Value>, GraphError> {
    match op {
        BuiltinNode::Identity => {
            if inputs.len() != 1 || output_count != 1 {
                return Err(GraphError::Invalid(format!(
                    "identity node '{node_name}' requires one input and one output"
                )));
            }
            Ok(inputs)
        }
        BuiltinNode::FanOut => {
            if inputs.len() != 1 {
                return Err(GraphError::Invalid(format!(
                    "fan_out node '{node_name}' requires one input"
                )));
            }
            Ok((0..output_count).map(|_| inputs[0].clone()).collect())
        }
        BuiltinNode::PackTuple => Ok(vec![Value::array(inputs)]),
        BuiltinNode::UnpackTuple => {
            if inputs.len() != 1 {
                return Err(GraphError::Invalid(format!(
                    "unpack_tuple node '{node_name}' requires one input"
                )));
            }
            let Some(values) = inputs[0].as_array() else {
                return Err(GraphError::Invalid(format!(
                    "unpack_tuple node '{node_name}' input must be an array"
                )));
            };
            if values.len() != output_count {
                return Err(GraphError::OutputArity {
                    node: node_name.to_string(),
                    expected: output_count,
                    got: values.len(),
                });
            }
            Ok(values.to_vec())
        }
    }
}

pub(super) fn inputs_ready_with_new_epoch(
    frame: &Frame,
    node: &CompiledNode,
) -> Result<bool, GraphError> {
    if !inputs_ready(frame, node)? {
        return Ok(false);
    }
    let current = input_activation_epoch(frame, node)?;
    let seen = frame
        .node_epochs
        .get(node.id.0)
        .ok_or(GraphError::MissingNode(node.id))?;
    Ok(*seen < current)
}

pub(super) fn inputs_ready(frame: &Frame, node: &CompiledNode) -> Result<bool, GraphError> {
    for edge in node.inputs.iter() {
        if !edge_ready(frame, *edge)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn edge_ready(frame: &Frame, edge: EdgeId) -> Result<bool, GraphError> {
    frame
        .values
        .get(edge.0)
        .map(|value| value.is_some())
        .ok_or(GraphError::MissingEdge(edge))
}

pub(super) fn input_activation_epoch(
    frame: &Frame,
    node: &CompiledNode,
) -> Result<u64, GraphError> {
    let mut activation = 0_u64;
    for edge in node.inputs.iter().copied() {
        let epoch = frame
            .edge_epochs
            .get(edge.0)
            .copied()
            .ok_or(GraphError::MissingEdge(edge))?;
        activation = activation.max(epoch);
    }
    Ok(activation)
}

pub(super) fn remember_node_activation(
    frame: &mut Frame,
    node: &CompiledNode,
) -> Result<(), GraphError> {
    let activation = input_activation_epoch(frame, node)?;
    let slot = frame
        .node_epochs
        .get_mut(node.id.0)
        .ok_or(GraphError::MissingNode(node.id))?;
    *slot = activation;
    Ok(())
}

pub(super) fn has_continuation(frame: &Frame, node: NodeId) -> Result<bool, GraphError> {
    frame
        .checkpoints
        .get(node.0)
        .map(|value| value.is_some())
        .ok_or(GraphError::MissingNode(node))
}

pub(super) fn read_single_input(frame: &Frame, node: &CompiledNode) -> Result<Value, GraphError> {
    read_edge(frame, single_input_edge(node)?)
}

pub(super) fn read_inputs(frame: &Frame, node: &CompiledNode) -> Result<Vec<Value>, GraphError> {
    node.inputs
        .iter()
        .map(|edge| read_edge(frame, *edge))
        .collect()
}

pub(super) fn read_edge(frame: &Frame, edge: EdgeId) -> Result<Value, GraphError> {
    frame
        .values
        .get(edge.0)
        .ok_or(GraphError::MissingEdge(edge))?
        .clone()
        .ok_or(GraphError::MissingEdge(edge))
}

pub(super) fn peek_edge(frame: &Frame, edge: EdgeId) -> Result<&Value, GraphError> {
    frame
        .values
        .get(edge.0)
        .ok_or(GraphError::MissingEdge(edge))?
        .as_ref()
        .ok_or(GraphError::MissingEdge(edge))
}

pub(super) fn single_input_edge(node: &CompiledNode) -> Result<EdgeId, GraphError> {
    if node.inputs.len() != 1 {
        return Err(GraphError::Invalid(format!(
            "node '{}' expected one input",
            node.name
        )));
    }
    node.inputs
        .first()
        .copied()
        .ok_or_else(|| GraphError::Invalid(format!("node '{}' input disappeared", node.name)))
}

pub(super) fn write_edge(frame: &mut Frame, edge: EdgeId, value: Value) -> Result<(), GraphError> {
    let epoch = next_write_epoch(frame)?;
    let slot = frame
        .values
        .get_mut(edge.0)
        .ok_or(GraphError::MissingEdge(edge))?;
    *slot = Some(value);
    let edge_epoch = frame
        .edge_epochs
        .get_mut(edge.0)
        .ok_or(GraphError::MissingEdge(edge))?;
    *edge_epoch = epoch;
    Ok(())
}

pub(super) fn write_variable(
    frame: &mut Frame,
    variable: VarId,
    value: Value,
) -> Result<(), GraphError> {
    let epoch = next_write_epoch(frame)?;
    let slot = frame
        .variables
        .get_mut(variable.0)
        .ok_or(GraphError::MissingVariable(variable))?;
    *slot = Some(value);
    let variable_epoch = frame
        .variable_epochs
        .get_mut(variable.0)
        .ok_or(GraphError::MissingVariable(variable))?;
    *variable_epoch = epoch;
    Ok(())
}

pub(super) fn ensure_write_capacity(frame: &Frame, writes: usize) -> Result<(), GraphError> {
    let writes = u64::try_from(writes)
        .map_err(|_| GraphError::Invalid("write count exceeds the supported range".into()))?;
    frame
        .write_epoch
        .checked_add(writes)
        .map(|_| ())
        .ok_or_else(|| GraphError::Invalid("frame write epoch overflowed".into()))
}

fn next_write_epoch(frame: &mut Frame) -> Result<u64, GraphError> {
    frame.write_epoch = frame
        .write_epoch
        .checked_add(1)
        .ok_or_else(|| GraphError::Invalid("frame write epoch overflowed".into()))?;
    Ok(frame.write_epoch)
}

pub(super) fn take_edge(frame: &mut Frame, edge: EdgeId) -> Result<Value, GraphError> {
    frame
        .values
        .get_mut(edge.0)
        .ok_or(GraphError::MissingEdge(edge))?
        .take()
        .ok_or(GraphError::MissingEdge(edge))
}

pub(super) fn describe_waiting(
    graph: &UntypedGraph,
    instructions: &[NodeId],
    frame: &Frame,
) -> String {
    let mut items = Vec::new();
    for node_id in instructions.iter().copied() {
        let Some(node) = graph.node(node_id) else {
            continue;
        };
        let missing = node
            .inputs
            .iter()
            .filter(|edge| frame.values.get(edge.0).is_none_or(|value| value.is_none()))
            .map(|edge| {
                match graph
                    .edge(*edge)
                    .and_then(|edge_data| edge_data.label.clone())
                {
                    Some(label) => label,
                    None => format!("{edge:?}"),
                }
            })
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            items.push(format!("{} waiting on [{}]", node.name, missing.join(", ")));
        } else if frame
            .node_epochs
            .get(node.id.0)
            .is_some_and(|epoch| *epoch > 0)
        {
            items.push(format!("{} waiting on newer input generation", node.name));
        }
    }
    if items.is_empty() {
        "<no ready nodes>".to_string()
    } else {
        items.join("; ")
    }
}
