use std::collections::{HashMap, HashSet, VecDeque};

use super::error::GraphError;
use super::ids::{EdgeId, MarkId, NodeId, VarId};
use super::model::{BuiltinNode, NodeKind, UNTYPED_GRAPH_SCHEMA_VERSION, UntypedGraph};

/// Validates structural graph invariants before compilation or execution.
pub fn validate_graph_shape(graph: &UntypedGraph) -> Result<(), GraphError> {
    validate_graph_shape_inner(graph)
}

fn validate_graph_shape_inner(graph: &UntypedGraph) -> Result<(), GraphError> {
    if graph.schema_version != UNTYPED_GRAPH_SCHEMA_VERSION {
        return Err(GraphError::UnsupportedVersion {
            format: "untyped graph schema",
            got: graph.schema_version,
            expected: UNTYPED_GRAPH_SCHEMA_VERSION,
        });
    }
    let mut problems = Vec::new();

    validate_dense_edges(graph, &mut problems);
    validate_dense_vars(graph, &mut problems);
    validate_dense_marks(graph, &mut problems);
    validate_dense_nodes(graph, &mut problems);
    validate_entry_exit(graph, &mut problems);
    validate_variable_keys(graph, &mut problems);
    validate_marks(graph, &mut problems);
    validate_node_references(graph, &mut problems);
    validate_variable_writes(graph, &mut problems);
    validate_edge_metadata(graph, &mut problems);
    validate_reachability(graph, &mut problems);

    for node in &graph.nodes {
        for (label, child) in child_graphs(&node.kind) {
            if let Err(err) = validate_graph_shape_inner(child) {
                problems.push(format!("{label} node '{}': {err}", node.name));
            }
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(GraphError::GraphValidation(problems.join("; ")))
    }
}

fn validate_dense_edges(graph: &UntypedGraph, problems: &mut Vec<String>) {
    for (idx, edge) in graph.edges.iter().enumerate() {
        if edge.id != EdgeId(idx) {
            problems.push(format!(
                "edge at index {idx} has id {:?}; ids must be dense and ordered",
                edge.id
            ));
        }
    }
}

fn validate_dense_vars(graph: &UntypedGraph, problems: &mut Vec<String>) {
    for (idx, var) in graph.variables.iter().enumerate() {
        if var.id != VarId(idx) {
            problems.push(format!(
                "variable at index {idx} has id {:?}; ids must be dense and ordered",
                var.id
            ));
        }
    }
}

fn validate_dense_marks(graph: &UntypedGraph, problems: &mut Vec<String>) {
    for (idx, mark) in graph.marks.iter().enumerate() {
        if mark.id != MarkId(idx) {
            problems.push(format!(
                "mark at index {idx} has id {:?}; ids must be dense and ordered",
                mark.id
            ));
        }
    }
}

fn validate_dense_nodes(graph: &UntypedGraph, problems: &mut Vec<String>) {
    for (idx, node) in graph.nodes.iter().enumerate() {
        if node.id != NodeId(idx) {
            problems.push(format!(
                "node at index {idx} has id {:?}; ids must be dense and ordered",
                node.id
            ));
        }
    }
}

fn validate_entry_exit(graph: &UntypedGraph, problems: &mut Vec<String>) {
    if graph.edge(graph.entry).is_none() {
        problems.push(format!("entry edge {:?} does not exist", graph.entry));
    }
    if graph.edge(graph.exit).is_none() {
        problems.push(format!("exit edge {:?} does not exist", graph.exit));
    }
    if graph
        .edge(graph.entry)
        .is_some_and(|entry| entry.producer.is_some())
    {
        problems.push(format!(
            "entry edge {:?} must not have a producer",
            graph.entry
        ));
    }
}

fn validate_variable_keys(graph: &UntypedGraph, problems: &mut Vec<String>) {
    let mut seen = HashSet::new();
    for var in &graph.variables {
        if var.key.namespace.is_empty() || var.key.type_name.is_empty() {
            problems.push(format!(
                "variable {:?} has empty namespace or type_name",
                var.id
            ));
        }
        if !seen.insert((var.key.namespace.clone(), var.key.type_name.clone())) {
            problems.push(format!(
                "duplicate variable key '{}::{}'",
                var.key.namespace, var.key.type_name
            ));
        }
    }
}

fn validate_marks(graph: &UntypedGraph, problems: &mut Vec<String>) {
    for mark in &graph.marks {
        match graph.edge(mark.target) {
            Some(edge) if edge.type_spec != mark.type_spec => problems.push(format!(
                "mark {:?} target edge {:?} type '{}' does not match mark type '{}'",
                mark.id, mark.target, edge.type_spec.name, mark.type_spec.name
            )),
            Some(_) => {}
            None => problems.push(format!(
                "mark {:?} references missing target edge {:?}",
                mark.id, mark.target
            )),
        }
    }
}

fn validate_node_references(graph: &UntypedGraph, problems: &mut Vec<String>) {
    for node in &graph.nodes {
        for edge in node.inputs.iter().chain(node.outputs.iter()) {
            if graph.edge(*edge).is_none() {
                problems.push(format!(
                    "node '{}' references missing edge {:?}",
                    node.name, edge
                ));
            }
        }

        match &node.kind {
            NodeKind::Load { var, .. } | NodeKind::Store { var, .. } => {
                if graph.variable(*var).is_none() {
                    problems.push(format!(
                        "node '{}' references missing variable {:?}",
                        node.name, var
                    ));
                }
                if node.inputs.len() != 1 || node.outputs.len() != 1 {
                    problems.push(format!(
                        "node '{}' load/store must have exactly one input and one output edge",
                        node.name
                    ));
                }
            }
            NodeKind::Builtin { op } => {
                validate_builtin_arity(
                    &node.name,
                    op,
                    node.inputs.len(),
                    node.outputs.len(),
                    problems,
                );
            }
            NodeKind::PureHandler { .. } | NodeKind::WorkHandler { .. } => {
                if node.outputs.is_empty() {
                    problems.push(format!(
                        "handler node '{}' must declare at least one output edge",
                        node.name
                    ));
                }
            }
            NodeKind::Continuation { key, payload, .. } => {
                if node.outputs.is_empty() {
                    problems.push(format!(
                        "handler node '{}' must declare at least one output edge",
                        node.name
                    ));
                }
                if let Err(err) = super::agent::validate_payload_handler(payload, key.as_str()) {
                    problems.push(err.to_string());
                }
            }
            NodeKind::Suspend { .. } => {
                if node.inputs.len() != 1 || node.outputs.len() != 1 {
                    problems.push(format!(
                        "suspend node '{}' must have exactly one input and one output edge",
                        node.name
                    ));
                }
            }
            NodeKind::Subflow { .. } => {
                if node.inputs.len() != 1 || node.outputs.len() != 1 {
                    problems.push(format!(
                        "subflow node '{}' must have exactly one input and one output edge",
                        node.name
                    ));
                }
            }
            NodeKind::Either { .. } => {
                if node.inputs.len() != 1 || node.outputs.len() != 1 {
                    problems.push(format!(
                        "either node '{}' must have exactly one input and one output edge",
                        node.name
                    ));
                }
            }
            NodeKind::Each { .. } => {
                if node.inputs.len() != 1 || node.outputs.len() != 1 {
                    problems.push(format!(
                        "each node '{}' must have exactly one input and one output edge",
                        node.name
                    ));
                }
            }
            NodeKind::Goto { mark } => {
                if node.inputs.len() != 1 || !node.outputs.is_empty() {
                    problems.push(format!(
                        "goto node '{}' must have exactly one input and no output edges",
                        node.name
                    ));
                }
                let Some(mark_data) = graph.mark(*mark) else {
                    problems.push(format!(
                        "goto node '{}' references missing mark {:?}",
                        node.name, mark
                    ));
                    continue;
                };
                if let Some(input_edge) = node.inputs.first().and_then(|edge| graph.edge(*edge))
                    && input_edge.type_spec != mark_data.type_spec
                {
                    problems.push(format!(
                        "goto node '{}' input type '{}' does not match mark target type '{}'",
                        node.name, input_edge.type_spec.name, mark_data.type_spec.name
                    ));
                }
            }
        }
    }
}

fn validate_builtin_arity(
    name: &str,
    op: &BuiltinNode,
    input_count: usize,
    output_count: usize,
    problems: &mut Vec<String>,
) {
    let valid = match op {
        BuiltinNode::Identity => input_count == 1 && output_count == 1,
        BuiltinNode::FanOut => input_count == 1 && output_count >= 1,
        BuiltinNode::PackTuple => input_count >= 1 && output_count == 1,
        BuiltinNode::UnpackTuple => input_count == 1 && output_count >= 1,
    };
    if !valid {
        problems.push(format!(
            "builtin node '{name}' has invalid arity for {op:?}: {input_count} input(s), {output_count} output(s)"
        ));
    }
}

fn validate_variable_writes(graph: &UntypedGraph, problems: &mut Vec<String>) {
    let mut writes_by_var: HashMap<VarId, Vec<NodeId>> = HashMap::new();
    for node in &graph.nodes {
        if let NodeKind::Store { var, .. } = node.kind {
            writes_by_var.entry(var).or_default().push(node.id);
        }
        for (_, child) in child_graphs(&node.kind) {
            validate_variable_writes(child, problems);
        }
    }

    for (var, writers) in writes_by_var {
        if writers.len() < 2 {
            continue;
        }
        'writers: for (index, first) in writers.iter().copied().enumerate() {
            for second in writers.iter().copied().skip(index + 1) {
                if must_run_before(graph, first, second) || must_run_before(graph, second, first) {
                    continue;
                }
                problems.push(format!(
                    "variable {:?} has competing unordered stores from nodes {:?}",
                    var, writers
                ));
                break 'writers;
            }
        }
    }
}

fn validate_edge_metadata(graph: &UntypedGraph, problems: &mut Vec<String>) {
    for edge in &graph.edges {
        if edge.id == graph.entry {
            for consumer in &edge.consumers {
                if graph.node(*consumer).is_none() {
                    problems.push(format!(
                        "entry edge {:?} lists missing consumer {:?}",
                        edge.id, consumer
                    ));
                }
            }
            continue;
        }
        let has_mark = graph.marks.iter().any(|mark| mark.target == edge.id);
        if edge.producer.is_none() && !has_mark {
            problems.push(format!("non-entry edge {:?} has no producer", edge.id));
        }
        for consumer in &edge.consumers {
            match graph.node(*consumer) {
                Some(node) if node.inputs.contains(&edge.id) => {}
                Some(node) => problems.push(format!(
                    "edge {:?} lists node '{}' as consumer, but node does not list edge as input",
                    edge.id, node.name
                )),
                None => problems.push(format!(
                    "edge {:?} lists missing consumer {:?}",
                    edge.id, consumer
                )),
            }
        }
        if let Some(producer) = edge.producer {
            match graph.node(producer) {
                Some(node) if node.outputs.contains(&edge.id) => {}
                Some(node) => problems.push(format!(
                    "edge {:?} lists node '{}' as producer, but node does not list edge as output",
                    edge.id, node.name
                )),
                None => problems.push(format!(
                    "edge {:?} lists missing producer {:?}",
                    edge.id, producer
                )),
            }
        }
    }

    for node in &graph.nodes {
        for input in &node.inputs {
            if let Some(edge) = graph.edge(*input)
                && !edge.consumers.contains(&node.id)
            {
                problems.push(format!(
                    "node '{}' lists input edge {:?}, but edge does not list node as consumer",
                    node.name, input
                ));
            }
        }
        for output in &node.outputs {
            if let Some(edge) = graph.edge(*output)
                && edge.producer != Some(node.id)
            {
                problems.push(format!(
                    "node '{}' lists output edge {:?}, but edge producer is {:?}",
                    node.name, output, edge.producer
                ));
            }
        }
    }
}

fn validate_reachability(graph: &UntypedGraph, problems: &mut Vec<String>) {
    let mut reachable_edges = HashSet::new();
    let mut reachable_nodes = HashSet::new();
    let mut queue = VecDeque::new();
    reachable_edges.insert(graph.entry);
    queue.push_back(graph.entry);

    while let Some(edge_id) = queue.pop_front() {
        let Some(edge) = graph.edge(edge_id) else {
            continue;
        };
        for node_id in &edge.consumers {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            if node
                .inputs
                .iter()
                .all(|edge| reachable_edges.contains(edge))
                && reachable_nodes.insert(*node_id)
            {
                if let NodeKind::Goto { mark } = &node.kind
                    && let Some(mark) = graph.mark(*mark)
                    && reachable_edges.insert(mark.target)
                {
                    queue.push_back(mark.target);
                }
                for output in &node.outputs {
                    if reachable_edges.insert(*output) {
                        queue.push_back(*output);
                    }
                }
            }
        }
    }

    if !reachable_edges.contains(&graph.exit) {
        problems.push(format!(
            "exit edge {:?} is not reachable from entry {:?}",
            graph.exit, graph.entry
        ));
    }
    for node in &graph.nodes {
        if !reachable_nodes.contains(&node.id) {
            problems.push(format!("node '{}' is unreachable from entry", node.name));
        }
    }
}

fn must_run_before(graph: &UntypedGraph, before: NodeId, after: NodeId) -> bool {
    if before == after {
        return true;
    }
    let mut queue = VecDeque::new();
    let mut seen_nodes = HashSet::new();
    let Some(start) = graph.node(before) else {
        return false;
    };
    for output in &start.outputs {
        queue.push_back(*output);
    }
    if let NodeKind::Goto { mark } = &start.kind
        && let Some(mark) = graph.mark(*mark)
    {
        queue.push_back(mark.target);
    }

    while let Some(edge_id) = queue.pop_front() {
        let Some(edge) = graph.edge(edge_id) else {
            continue;
        };
        for consumer in &edge.consumers {
            if *consumer == after {
                return true;
            }
            if !seen_nodes.insert(*consumer) {
                continue;
            }
            let Some(node) = graph.node(*consumer) else {
                continue;
            };
            for output in &node.outputs {
                queue.push_back(*output);
            }
            if let NodeKind::Goto { mark } = &node.kind
                && let Some(mark) = graph.mark(*mark)
            {
                queue.push_back(mark.target);
            }
        }
    }
    false
}

/// Verifies every handler key in a graph can be resolved by kind.
///
/// JSON loaders and runtime constructors use this to fail before execution
/// rather than discovering missing handlers lazily.
pub fn validate_registry_keys(
    graph: &UntypedGraph,
    has_value: &dyn Fn(&str) -> bool,
    has_work: &dyn Fn(&str) -> bool,
    has_continuation: &dyn Fn(&str) -> bool,
) -> Result<(), GraphError> {
    let mut missing = Vec::new();
    for node in &graph.nodes {
        match &node.kind {
            NodeKind::PureHandler { key }
            | NodeKind::Load { key, .. }
            | NodeKind::Store { key, .. } => {
                if !has_value(key.as_str()) {
                    missing.push(format!(
                        "node '{}' missing value handler '{}'",
                        node.name,
                        key.as_str()
                    ));
                }
            }
            NodeKind::WorkHandler { key } => {
                if !has_work(key.as_str()) {
                    missing.push(format!(
                        "node '{}' missing work handler '{}'",
                        node.name,
                        key.as_str()
                    ));
                }
            }
            NodeKind::Continuation { key, .. } => {
                if !has_continuation(key.as_str()) {
                    missing.push(format!(
                        "node '{}' missing continuation handler '{}'",
                        node.name,
                        key.as_str()
                    ));
                }
                for (label, child) in child_graphs(&node.kind) {
                    if let Err(err) =
                        validate_registry_keys(child, has_value, has_work, has_continuation)
                    {
                        missing.push(format!("{label} node '{}': {err}", node.name));
                    }
                }
            }
            NodeKind::Either { key, .. } => {
                if !has_value(key.as_str()) {
                    missing.push(format!(
                        "node '{}' missing value handler '{}'",
                        node.name,
                        key.as_str()
                    ));
                }
                for (label, child) in child_graphs(&node.kind) {
                    if let Err(err) =
                        validate_registry_keys(child, has_value, has_work, has_continuation)
                    {
                        missing.push(format!("{label} node '{}': {err}", node.name));
                    }
                }
            }
            NodeKind::Subflow { .. } | NodeKind::Each { .. } => {
                for (label, child) in child_graphs(&node.kind) {
                    if let Err(err) =
                        validate_registry_keys(child, has_value, has_work, has_continuation)
                    {
                        missing.push(format!("{label} node '{}': {err}", node.name));
                    }
                }
            }
            NodeKind::Builtin { .. } | NodeKind::Suspend { .. } | NodeKind::Goto { .. } => {}
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(GraphError::Invalid(missing.join("; ")))
    }
}

fn child_graphs(kind: &NodeKind) -> Vec<(&'static str, &UntypedGraph)> {
    match kind {
        NodeKind::Subflow { graph } => vec![("subflow", graph)],
        NodeKind::Either { left, right, .. } => {
            vec![("either left", left), ("either right", right)]
        }
        NodeKind::Each { graph } => vec![("each", graph)],
        NodeKind::Continuation { children, .. } => children
            .iter()
            .map(|child| ("continuation child", child))
            .collect(),
        NodeKind::Builtin { .. }
        | NodeKind::PureHandler { .. }
        | NodeKind::WorkHandler { .. }
        | NodeKind::Suspend { .. }
        | NodeKind::Load { .. }
        | NodeKind::Store { .. } => Vec::new(),
        NodeKind::Goto { .. } => Vec::new(),
    }
}
