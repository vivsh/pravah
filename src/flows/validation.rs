use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use super::flow::FlowGraph;
use super::nodes::{AgentInfo, EachInfo, EitherInfo, FlowNode, ForkInfo, JoinInfo};
use crate::flows::{NodeId, errors::BuildError};

fn node_outputs(node: &FlowNode) -> Vec<NodeId> {
    match node {
        FlowNode::Work(w) => vec![w.exit_name],
        FlowNode::ToolWork(w) => vec![w.exit_name],
        FlowNode::Map(m) => vec![m.exit_name],
        FlowNode::Suspend(s) => vec![s.exit],
        FlowNode::Agent(a) => {
            let mut v = vec![a.exit];
            v.extend(a.tool_lookup.values().map(|(_, e)| *e));
            v
        }
        FlowNode::Fork(f) => f.children.clone(),
        FlowNode::Join(j) => vec![j.target],
        FlowNode::Either(e) => vec![e.left_name, e.right_name],
        FlowNode::Flow(inner) => inner.parent_exit.into_iter().collect(),
        FlowNode::Each(_) => vec![],
    }
}

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
        if *exit_id == info.exit {
            continue;
        }
        match nodes.get(entry_id) {
            None => problems.push(format!(
                "agent '{key_str}': tool '{tool_name}' has no node registered for input slot '{}'",
                graph.interner.name_of(*entry_id),
            )),
            Some(node) if !node_outputs(node).contains(exit_id) => problems.push(format!(
                "agent '{key_str}': tool '{tool_name}' node at '{}' does not produce expected output slot '{}'",
                graph.interner.name_of(*entry_id),
                graph.interner.name_of(*exit_id),
            )),
            _ => {}
        }
        if nodes.contains_key(exit_id) {
            problems.push(format!(
                "agent '{key_str}': tool '{tool_name}' output slot '{}' must be terminal — \
                 it is used as an input to another node",
                graph.interner.name_of(*exit_id),
            ));
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
            problems.push(format!(
                "fork '{key_str}': duplicate child '{}'",
                graph.interner.name_of(child)
            ));
        }
        if !nodes.contains_key(&child) {
            problems.push(format!(
                "fork '{key_str}': child '{}' is not a registered node",
                graph.interner.name_of(child)
            ));
        }
    }
}

fn check_join(
    info: &JoinInfo,
    nodes: &HashMap<NodeId, FlowNode>,
    graph: &FlowGraph,
    join_groups: &mut HashMap<NodeId, Vec<NodeId>>,
    problems: &mut Vec<String>,
) {
    let target_str = graph.interner.name_of(info.target);
    let mut parents = info.parents.clone();
    parents.sort_by_key(|id| id.0);
    if let Some(existing) = join_groups.get(&info.target) {
        if existing != &parents {
            let existing_names = existing
                .iter()
                .map(|id| graph.interner.name_of(*id))
                .collect::<Vec<_>>()
                .join(", ");
            let parent_names = parents
                .iter()
                .map(|id| graph.interner.name_of(*id))
                .collect::<Vec<_>>()
                .join(", ");
            problems.push(format!(
                "join (target '{target_str}'): conflicting parent groups [{existing_names}] and [{parent_names}]"
            ));
        }
    } else {
        join_groups.insert(info.target, parents);
    }

    if info.parents.len() < 2 {
        problems.push(format!(
            "join (target '{target_str}'): must have at least 2 parents, found {}",
            info.parents.len()
        ));
    }
    let mut seen_parents: HashSet<NodeId> = HashSet::new();
    for &p in &info.parents {
        if !seen_parents.insert(p) {
            problems.push(format!(
                "join (target '{target_str}'): duplicate parent '{}'",
                graph.interner.name_of(p)
            ));
        }
        if p == info.target {
            problems.push(format!(
                "join (target '{target_str}'): target matches parent '{}'",
                graph.interner.name_of(p)
            ));
        }
        if !nodes.contains_key(&p) {
            problems.push(format!(
                "join (target '{target_str}'): parent '{}' is not a registered node",
                graph.interner.name_of(p)
            ));
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

fn check_each(key_str: &str, info: &EachInfo, graph: &FlowGraph, problems: &mut Vec<String>) {
    if info.id == info.exit {
        problems.push(format!(
            "each '{key_str}': exit equals input — node would overwrite its own input"
        ));
    }
    let exit_str = graph.interner.name_of(info.exit);
    if !graph.interner.fwd.contains_key(exit_str) {
        problems.push(format!(
            "each '{key_str}': output type '{exit_str}' is not registered in the parent graph"
        ));
    }
}

fn check_flow(
    key_str: &str,
    inner: &Arc<FlowGraph>,
    graph: &FlowGraph,
    problems: &mut Vec<String>,
) {
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
    if !graph.interner.fwd.contains_key(exit_str) {
        problems.push(format!(
            "flow '{key_str}': output type '{exit_str}' is not registered in the parent graph"
        ));
    }
}

/// Validates node-local rules that do not depend on the graph entry point.
pub fn validate_nodes(
    nodes: &HashMap<NodeId, FlowNode>,
    graph: &FlowGraph,
) -> Result<(), BuildError> {
    let mut problems: Vec<String> = Vec::new();
    let mut join_groups: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (key, node) in nodes {
        let key_str = graph.interner.name_of(*key);
        match node {
            FlowNode::Agent(info) => check_agent(key_str, info, nodes, graph, &mut problems),
            FlowNode::Work(info) if info.exit_name == info.name => {
                problems.push(format!("work '{key_str}': exit_name equals input name — node would overwrite its own input"));
            }
            FlowNode::ToolWork(info) if info.exit_name == info.name => {
                problems.push(format!("tool_work '{key_str}': exit_name equals input name — node would overwrite its own input"));
            }
            FlowNode::Map(info) if info.exit_name == info.name => {
                problems.push(format!("map '{key_str}': exit_name equals input name — node would overwrite its own input"));
            }
            FlowNode::Suspend(info) if info.entry == info.exit => {
                problems.push(format!(
                    "suspend '{key_str}': entry equals exit — node would overwrite its own input"
                ));
            }
            FlowNode::Fork(info) => check_fork(key_str, info, nodes, graph, &mut problems),
            FlowNode::Join(info) => check_join(info, nodes, graph, &mut join_groups, &mut problems),
            FlowNode::Either(info) => check_either(key_str, info, graph, &mut problems),
            FlowNode::Flow(inner) => check_flow(key_str, inner, graph, &mut problems),
            FlowNode::Each(info) => check_each(key_str, info, graph, &mut problems),
            _ => {}
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(BuildError::Invalid(problems))
    }
}

fn build_successors(
    nodes: &HashMap<NodeId, FlowNode>,
    graph: &FlowGraph,
) -> HashMap<NodeId, Vec<NodeId>> {
    nodes
        .iter()
        .map(|(&key, node)| {
            let succs: Vec<NodeId> = match node {
                FlowNode::Agent(info) => {
                    let mut s = vec![info.exit];
                    s.extend(
                        info.tool_lookup
                            .values()
                            .map(|(e, _)| *e)
                            .filter(|&e| e != info.exit),
                    );
                    s
                }
                FlowNode::Work(info) => vec![info.exit_name],
                FlowNode::ToolWork(info) => vec![info.exit_name],
                FlowNode::Map(info) => vec![info.exit_name],
                FlowNode::Suspend(info) => vec![info.exit],
                FlowNode::Fork(info) => info.children.clone(),
                FlowNode::Join(info) => vec![info.target],
                FlowNode::Either(info) => vec![info.left_name, info.right_name],
                FlowNode::Flow(inner) => {
                    let exit_str = inner.interner.name_of(inner.exit);
                    graph
                        .interner
                        .fwd
                        .get(exit_str)
                        .copied()
                        .map(|id| vec![id])
                        .unwrap_or_default()
                }
                FlowNode::Each(info) => vec![info.exit],
            };
            (key, succs)
        })
        .collect()
}

fn reachable_from(
    entry: NodeId,
    successors: &HashMap<NodeId, Vec<NodeId>>,
    nodes: &HashMap<NodeId, FlowNode>,
) -> HashSet<NodeId> {
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
    let terminals: HashSet<NodeId> = successors
        .values()
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
            problems.push(format!(
                "node '{}': unreachable from entry '{entry_str}'",
                graph.interner.name_of(key)
            ));
        }
    }
    let predecessors = predecessors_of(&successors);
    let can_reach_terminal = nodes_reaching_terminal(&successors, &predecessors, nodes);
    for &key in nodes.keys() {
        if !can_reach_terminal.contains(&key) {
            problems.push(format!(
                "node '{}': has no path to any terminal — dead end",
                graph.interner.name_of(key)
            ));
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(BuildError::Invalid(problems))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::Value;

    use super::*;
    use crate::flows::nodes::{FlowNode, JoinInfo, StateNode};

    fn join(parents: Vec<NodeId>, target: NodeId) -> FlowNode {
        FlowNode::Join(JoinInfo {
            parents,
            target,
            func: Arc::new(move |_| {
                Ok(StateNode {
                    name: "Target".into(),
                    value: Value::Null,
                })
            }),
        })
    }

    /// Join validation reports conflicting parent groups that share one target.
    #[test]
    fn validate_nodes_rejects_conflicting_join_groups_with_same_target() {
        let mut graph = FlowGraph::new();
        let a = graph.interner.intern("A");
        let b = graph.interner.intern("B");
        let c = graph.interner.intern("C");
        let d = graph.interner.intern("D");
        let target = graph.interner.intern("Target");

        graph.nodes.insert(a, join(vec![a, b], target));
        graph.nodes.insert(b, join(vec![a, b], target));
        graph.nodes.insert(c, join(vec![c, d], target));
        graph.nodes.insert(d, join(vec![c, d], target));

        let err = validate_nodes(&graph.nodes, &graph).expect_err("join groups should conflict");
        let BuildError::Invalid(problems) = err else {
            panic!("expected validation error");
        };
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("conflicting parent groups")),
            "expected conflicting join group error, got {problems:?}"
        );
    }
}
