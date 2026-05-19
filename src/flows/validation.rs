use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::flows::{FlowGraph, NodeId, errors::BuildError, flows::FlowNode};
use crate::flows::flows::{AgentInfo, EitherInfo, ForkInfo, JoinInfo};

fn check_agent(
    key_str: &str,
    info: &AgentInfo,
    nodes: &HashMap<NodeId, FlowNode>,
    graph: &FlowGraph,
    problems: &mut Vec<String>,
) {
    if info.exit == info.id {
        problems.push(format!(
            "agent '{key_str}': exit_name equals input name — node would overwrite its own input"
        ));
    }
    if info.model.is_empty() {
        problems.push(format!("agent '{key_str}': model is empty"));
    }
    for (tool_name, (entry_id, exit_id)) in &info.tool_lookup {
        match nodes.get(entry_id) {
            None => problems.push(format!(
                "agent '{key_str}': tool '{tool_name}' has no work node registered for input slot '{}'",
                graph.interner.name_of(*entry_id),
            )),
            Some(FlowNode::Work(work)) if work.exit_name != *exit_id => problems.push(format!(
                "agent '{key_str}': tool '{tool_name}' work node output slot '{}' does not match expected exit slot '{}'",
                graph.interner.name_of(work.exit_name),
                graph.interner.name_of(*exit_id),
            )),
            _ => {}
        }
    }
}

fn check_fork(
    key_str: &str,
    info: &ForkInfo,
    nodes: &HashMap<NodeId, FlowNode>,
    graph: &FlowGraph,
    problems: &mut Vec<String>,
) {
    if info.children.len() < 2 {
        problems.push(format!(
            "fork '{key_str}': must have at least 2 children, found {}",
            info.children.len()
        ));
    }
    let mut seen_children: HashSet<NodeId> = HashSet::new();
    for &child in &info.children {
        if !seen_children.insert(child) {
            problems.push(format!("fork '{key_str}': duplicate child '{}'", graph.interner.name_of(child)));
        }
        if !nodes.contains_key(&child) {
            problems.push(format!("fork '{key_str}': child '{}' is not a registered node", graph.interner.name_of(child)));
        }
    }
}

fn check_join(
    info: &JoinInfo,
    nodes: &HashMap<NodeId, FlowNode>,
    graph: &FlowGraph,
    seen: &mut HashSet<NodeId>,
    problems: &mut Vec<String>,
) {
    if !seen.insert(info.target) {
        return;
    }
    let target_str = graph.interner.name_of(info.target);
    if info.parents.len() < 2 {
        problems.push(format!(
            "join (target '{target_str}'): must have at least 2 parents, found {}",
            info.parents.len()
        ));
    }
    let mut seen_parents: HashSet<NodeId> = HashSet::new();
    for &p in &info.parents {
        if !seen_parents.insert(p) {
            problems.push(format!("join (target '{target_str}'): duplicate parent '{}'", graph.interner.name_of(p)));
        }
        if p == info.target {
            problems.push(format!("join (target '{target_str}'): target matches parent '{}'", graph.interner.name_of(p)));
        }
        if !nodes.contains_key(&p) {
            problems.push(format!("join (target '{target_str}'): parent '{}' is not a registered node", graph.interner.name_of(p)));
        }
    }
}

fn check_either(key_str: &str, info: &EitherInfo, graph: &FlowGraph, problems: &mut Vec<String>) {
    if info.left_name == info.right_name {
        problems.push(format!(
            "either '{key_str}': both branches resolve to the same schema name '{}'",
            graph.interner.name_of(info.left_name)
        ));
    }
}

fn check_flow(key_str: &str, inner: &Arc<FlowGraph>, graph: &FlowGraph, problems: &mut Vec<String>) {
    let Some(inner_name) = inner.parent_entry else {
        problems.push(format!("flow '{key_str}': missing name or exit_name"));
        return;
    };
    let exit_str = inner.interner.name_of(inner.exit);
    let parent_entry_str = graph.interner.name_of(inner_name);
    if parent_entry_str == exit_str {
        problems.push(format!(
            "flow '{key_str}': exit_name equals input name — sub-flow output would overwrite its own input"
        ));
    }
    if graph.interner.fwd.get(exit_str).is_none() {
        problems.push(format!(
            "flow '{key_str}': output type '{exit_str}' is not registered in the parent graph"
        ));
    }
}

/// Validates node-local rules that do not depend on the graph entry point.
pub fn validate_nodes(nodes: &HashMap<NodeId, FlowNode>, graph: &FlowGraph) -> Result<(), BuildError> {
    let mut problems: Vec<String> = Vec::new();
    let mut seen_join_groups: HashSet<NodeId> = HashSet::new();
    for (key, node) in nodes {
        let key_str = graph.interner.name_of(*key);
        match node {
            FlowNode::Agent(info) => check_agent(key_str, info, nodes, graph, &mut problems),
            FlowNode::Work(info) if info.exit_name == info.name => {
                problems.push(format!("work '{key_str}': exit_name equals input name — node would overwrite its own input"));
            }
            FlowNode::Map(info) if info.exit_name == info.name => {
                problems.push(format!("map '{key_str}': exit_name equals input name — node would overwrite its own input"));
            }
            FlowNode::Suspend(info) if info.entry == info.exit => {
                problems.push(format!("suspend '{key_str}': entry equals exit — node would overwrite its own input"));
            }
            FlowNode::Fork(info) => check_fork(key_str, info, nodes, graph, &mut problems),
            FlowNode::Join(info) => check_join(info, nodes, graph, &mut seen_join_groups, &mut problems),
            FlowNode::Either(info) => check_either(key_str, info, graph, &mut problems),
            FlowNode::Flow(inner) => check_flow(key_str, inner, graph, &mut problems),
            _ => {}
        }
    }
    if problems.is_empty() { Ok(()) } else { Err(BuildError::Invalid(problems)) }
}

fn build_successors(nodes: &HashMap<NodeId, FlowNode>, graph: &FlowGraph) -> HashMap<NodeId, Vec<NodeId>> {
    nodes.iter().map(|(&key, node)| {
        let succs: Vec<NodeId> = match node {
            FlowNode::Agent(info) => {
                let mut s = vec![info.exit];
                s.extend(info.tool_lookup.values().map(|(e, _)| *e));
                s
            }
            FlowNode::Work(info) => vec![info.exit_name],
            FlowNode::Map(info) => vec![info.exit_name],
            FlowNode::Suspend(info) => vec![info.exit],
            FlowNode::Fork(info) => info.children.clone(),
            FlowNode::Join(info) => vec![info.target],
            FlowNode::Either(info) => vec![info.left_name, info.right_name],
            FlowNode::Flow(inner) => {
                let exit_str = inner.interner.name_of(inner.exit);
                graph.interner.fwd.get(exit_str).copied().map(|id| vec![id]).unwrap_or_default()
            }
        };
        (key, succs)
    }).collect()
}

fn reachable_from(entry: NodeId, successors: &HashMap<NodeId, Vec<NodeId>>, nodes: &HashMap<NodeId, FlowNode>) -> HashSet<NodeId> {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    visited.insert(entry);
    queue.push_back(entry);
    while let Some(cur) = queue.pop_front() {
        for &s in successors.get(&cur).into_iter().flatten() {
            if nodes.contains_key(&s) && visited.insert(s) {
                queue.push_back(s);
            }
        }
    }
    visited
}

fn predecessors_of(successors: &HashMap<NodeId, Vec<NodeId>>) -> HashMap<NodeId, Vec<NodeId>> {
    let mut preds: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (&key, succs) in successors {
        for &s in succs {
            preds.entry(s).or_default().push(key);
        }
    }
    preds
}

fn nodes_reaching_terminal(
    successors: &HashMap<NodeId, Vec<NodeId>>,
    predecessors: &HashMap<NodeId, Vec<NodeId>>,
    nodes: &HashMap<NodeId, FlowNode>,
) -> HashSet<NodeId> {
    let terminals: HashSet<NodeId> = successors.values()
        .flat_map(|v| v.iter().copied())
        .filter(|s| !nodes.contains_key(s))
        .collect();
    let mut can_reach: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    for &t in &terminals {
        for &p in predecessors.get(&t).into_iter().flatten() {
            if can_reach.insert(p) {
                queue.push_back(p);
            }
        }
    }
    while let Some(cur) = queue.pop_front() {
        for &p in predecessors.get(&cur).into_iter().flatten() {
            if can_reach.insert(p) {
                queue.push_back(p);
            }
        }
    }
    can_reach
}

/// Validates entry wiring and whole-graph reachability.
pub fn validate(
    nodes: &HashMap<NodeId, FlowNode>,
    entry: NodeId,
    graph: &FlowGraph,
) -> Result<(), BuildError> {
    let mut problems: Vec<String> = Vec::new();
    let entry_str = graph.interner.name_of(entry);
    if !nodes.contains_key(&entry) {
        problems.push(format!("entry '{entry_str}' is not a registered node"));
        return Err(BuildError::Invalid(problems));
    }
    let successors = build_successors(nodes, graph);
    let reachable = reachable_from(entry, &successors, nodes);
    for &key in nodes.keys() {
        if !reachable.contains(&key) {
            problems.push(format!("node '{}': unreachable from entry '{entry_str}'", graph.interner.name_of(key)));
        }
    }
    let predecessors = predecessors_of(&successors);
    let can_reach_terminal = nodes_reaching_terminal(&successors, &predecessors, nodes);
    for &key in nodes.keys() {
        if !can_reach_terminal.contains(&key) {
            problems.push(format!("node '{}': has no path to any terminal — dead end", graph.interner.name_of(key)));
        }
    }
    if problems.is_empty() { Ok(()) } else { Err(BuildError::Invalid(problems)) }
}
