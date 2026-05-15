use std::collections::{HashMap, HashSet, VecDeque};

use crate::flows::{FlowGraph, NodeId, errors::BuildError, flows::FlowNode};




/// Validates per-node structural rules for every node in the graph.
///
/// This function checks only local, self-contained invariants on each node. It does **not**
/// require an entry point to be known and does **not** perform reachability analysis.
/// Called by [`FlowBuilder::build`] as the first validation pass.
///
/// # Checks performed
///
/// **Agent nodes**
/// - `exit` must differ from `id` — an agent cannot write its output to the same state slot
///   it reads its input from, which would silently overwrite the input before execution.
/// - `model` must be a non-empty string.
///
/// **Work nodes**
/// - `exit_name` must differ from `name` — same self-overwrite guard as for agents.
///
/// **Fork nodes**
/// - Must declare at least 2 children; a fork with fewer children is a degenerate branch.
/// - No duplicate child `NodeId`s within the same fork.
/// - Every child `NodeId` must refer to a node that is already registered in `nodes`.
///
/// **Join nodes**
/// - Each join target group (identified by `target`) is validated only once; duplicate join
///   entries for the same target are skipped after the first.
/// - Must have at least 2 parents.
/// - No duplicate parent `NodeId`s within the same join.
/// - No parent `NodeId` may equal the join's own `target`.
/// - Every parent `NodeId` must refer to a registered node.
///
/// **Either nodes**
/// - `left_name` and `right_name` must be distinct; if both branches resolve to the same
///   schema name the routing is meaningless.
///
/// **Flow (sub-flow) nodes**
/// - Must have a `parent_entry` (i.e. the outer entry slot must be bound); a sub-flow with no
///   entry context cannot be wired into the parent graph.
/// - `parent_entry` must differ from the sub-flow's `exit`; otherwise the sub-flow output
///   would overwrite the value it was given as input.
///
/// **Tool nodes**
/// - No per-node checks; tool nodes are validated only indirectly through their owning agent.
pub fn validate_nodes(nodes: &HashMap<NodeId, FlowNode>, graph: &FlowGraph) -> Result<(), BuildError> {
    let mut problems: Vec<String> = Vec::new();
    let mut seen_join_groups: HashSet<NodeId> = HashSet::new();
    for (key, node) in nodes {
        let key_str = graph.interner.name_of(*key);
        match node {
            FlowNode::Agent(info) => {
                if info.exit == info.id {
                    problems.push(format!(
                        "agent '{}': exit_name equals input name — node would overwrite its own input",
                        key_str
                    ));
                }
                if info.model.is_empty() {
                    problems.push(format!("agent '{}': model is empty", key_str));
                }
            }
            FlowNode::Work(info) => {
                if info.exit_name == info.name {
                    problems.push(format!(
                        "work '{}': exit_name equals input name — node would overwrite its own input",
                        key_str
                    ));
                }
            }
            FlowNode::Map(info) => {
                if info.exit_name == info.name {
                    problems.push(format!(
                        "map '{}': exit_name equals input name — node would overwrite its own input",
                        key_str
                    ));
                }
            }
            FlowNode::Suspend(info) => {
                if info.entry == info.exit {
                    problems.push(format!(
                        "suspend '{}': entry equals exit — node would overwrite its own input",
                        key_str
                    ));
                }
            }
            FlowNode::Fork(info) => {
                if info.children.len() < 2 {
                    problems.push(format!(
                        "fork '{}': must have at least 2 children, found {}",
                        key_str,
                        info.children.len()
                    ));
                }
                let mut seen_children: HashSet<NodeId> = HashSet::new();
                for &child in &info.children {
                    if !seen_children.insert(child) {
                        problems.push(format!(
                            "fork '{}': duplicate child '{}'",
                            key_str,
                            graph.interner.name_of(child)
                        ));
                    }
                    if !nodes.contains_key(&child) {
                        problems.push(format!(
                            "fork '{}': child '{}' is not a registered node",
                            key_str,
                            graph.interner.name_of(child)
                        ));
                    }
                }
            }
            FlowNode::Join(info) => {
                let group_key = info.target;
                if !seen_join_groups.insert(group_key) {
                    continue;
                }
                if info.parents.len() < 2 {
                    problems.push(format!(
                        "join (target '{}'): must have at least 2 parents, found {}",
                        graph.interner.name_of(info.target),
                        info.parents.len()
                    ));
                }
                let mut seen_parents: HashSet<NodeId> = HashSet::new();
                for &p in &info.parents {
                    if !seen_parents.insert(p) {
                        problems.push(format!(
                            "join (target '{}'): duplicate parent '{}'",
                            graph.interner.name_of(info.target),
                            graph.interner.name_of(p)
                        ));
                    }
                    if p == info.target {
                        problems.push(format!(
                            "join (target '{}'): target matches parent '{}'",
                            graph.interner.name_of(info.target),
                            graph.interner.name_of(p)
                        ));
                    }
                    if !nodes.contains_key(&p) {
                        problems.push(format!(
                            "join (target '{}'): parent '{}' is not a registered node",
                            graph.interner.name_of(info.target),
                            graph.interner.name_of(p)
                        ));
                    }
                }
            }
            FlowNode::Either(info) => {
                if info.left_name == info.right_name {
                    problems.push(format!(
                        "either '{}': both branches resolve to the same schema name '{}'",
                        key_str,
                        graph.interner.name_of(info.left_name)
                    ));
                }
            }
            FlowNode::Flow(inner) => {
                let (inner_name, _inner_exit) = match inner.parent_entry {
                    Some(n) => (n, inner.exit),
                    None => {
                        problems.push(format!("flow '{}': missing name or exit_name", key_str));
                        continue;
                    }
                };
                let exit_str = inner.interner.name_of(inner.exit);
                let parent_entry_str = graph.interner.name_of(inner_name);
                if parent_entry_str == exit_str {
                    problems.push(format!(
                        "flow '{}': exit_name equals input name — sub-flow output would overwrite its own input",
                        key_str
                    ));
                }
                if graph.interner.fwd.get(exit_str).is_none() {
                    problems.push(format!(
                        "flow '{}': output type '{}' is not registered in the parent graph",
                        key_str, exit_str
                    ));
                }
            }
            FlowNode::AgentTool(info) => {
                if info.exit == info.id {
                    problems.push(format!(
                        "agent_tool '{}': exit equals entry — would overwrite input",
                        key_str
                    ));
                }
            }
            FlowNode::Tool(_) | FlowNode::FlowTool { .. } => {}
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(BuildError::Invalid(problems))
    }
}

/// Validates entry registration and graph-level reachability.
///
/// Called by [`FlowGraph::with_entry`] after an entry point has been designated. This is the
/// second validation pass and assumes [`validate_nodes`] has already succeeded.
///
/// # Checks performed
///
/// **Entry registration**
/// - The `entry` `NodeId` must refer to a node that is registered in `nodes`. If the entry is
///   not found, reachability checks are skipped entirely (they would be meaningless without a
///   valid starting point).
///
/// **Forward reachability** (skipped when entry is invalid)
/// - Every non-`Tool` node must be reachable from `entry` by following successor edges:
///   - `Agent` → `exit`
///   - `Work` → `exit_name`
///   - `Fork` → each child
///   - `Join` → `target`
///   - `Either` → `left_name`, `right_name`
///   - `Flow` → the outer-interner translation of the inner graph's `exit`
/// - A node that cannot be reached from `entry` is flagged as unreachable. `Tool` nodes are
///   excluded because they are driven by agents, not by the state-transition graph.
///
/// **Backward reachability / liveness** (skipped when entry is invalid)
/// - Terminal state IDs are identified as successor IDs that are **not** themselves registered
///   nodes (i.e. they are pure output slots, not executable nodes).
/// - Every non-`Tool` node must have at least one path leading to a terminal state ID. A node
///   with no path to any terminal is a dead end — execution that reaches it can never produce
///   a final value.
pub fn validate(
    nodes: &HashMap<NodeId, FlowNode>,
    entry: NodeId,
    graph: &FlowGraph,
) -> Result<(), BuildError> {
    let mut problems: Vec<String> = Vec::new();
    let entry_str = graph.interner.name_of(entry);

    // --- Entry check ---
    if !nodes.contains_key(&entry) {
        problems.push(format!("entry '{}' is not a registered node", entry_str));
    }

    // --- Reachability (only when entry is valid) ---
    if nodes.contains_key(&entry) {
        let successors: HashMap<NodeId, Vec<NodeId>> = nodes
            .iter()
            .filter(|(_, node)| !matches!(node, FlowNode::Tool(_) | FlowNode::AgentTool(_) | FlowNode::FlowTool { .. }))
            .map(|(&key, node)| {
                let succs: Vec<NodeId> = match node {
                    FlowNode::Agent(info) => vec![info.exit],
                    FlowNode::Work(info) => vec![info.exit_name],
                    FlowNode::Map(info) => vec![info.exit_name],
                    FlowNode::Suspend(info) => vec![info.exit],
                    FlowNode::Fork(info) => info.children.clone(),
                    FlowNode::Join(info) => vec![info.target],
                    FlowNode::Either(info) => vec![info.left_name, info.right_name],
                    FlowNode::Flow(inner) => {
                        // inner.exit is a NodeId in the inner graph's interner.
                        // Resolve the string, then look up in the outer (graph) interner.
                        let exit_str = inner.interner.name_of(inner.exit);
                        graph
                            .interner
                            .fwd
                            .get(exit_str)
                            .copied()
                            .map(|id| vec![id])
                            .unwrap_or_default()
                    }
                    FlowNode::Tool(_) | FlowNode::AgentTool(_) | FlowNode::FlowTool { .. } => vec![],
                };
                (key, succs)
            })
            .collect();

        let mut reachable: HashSet<NodeId> = HashSet::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        reachable.insert(entry);
        queue.push_back(entry);
        while let Some(cur) = queue.pop_front() {
            if let Some(succs) = successors.get(&cur) {
                for &s in succs {
                    if nodes.contains_key(&s) && reachable.insert(s) {
                        queue.push_back(s);
                    }
                }
            }
        }
        for &key in nodes.keys() {
            if matches!(nodes[&key], FlowNode::Tool(_) | FlowNode::AgentTool(_) | FlowNode::FlowTool { .. }) {
                continue;
            }
            if !reachable.contains(&key) {
                problems.push(format!(
                    "node '{}': unreachable from entry '{}'",
                    graph.interner.name_of(key),
                    entry_str
                ));
            }
        }

        let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (&key, succs) in &successors {
            for &s in succs {
                predecessors.entry(s).or_default().push(key);
            }
        }

        let terminals: HashSet<NodeId> = successors
            .values()
            .flat_map(|v| v.iter().copied())
            .filter(|s| !nodes.contains_key(s))
            .collect();

        let mut can_reach_terminal: HashSet<NodeId> = HashSet::new();
        let mut queue2: VecDeque<NodeId> = VecDeque::new();
        for &t in &terminals {
            if let Some(preds) = predecessors.get(&t) {
                for &p in preds {
                    if can_reach_terminal.insert(p) {
                        queue2.push_back(p);
                    }
                }
            }
        }
        while let Some(cur) = queue2.pop_front() {
            if let Some(preds) = predecessors.get(&cur) {
                for &p in preds {
                    if can_reach_terminal.insert(p) {
                        queue2.push_back(p);
                    }
                }
            }
        }
        for &key in nodes.keys() {
            if matches!(nodes[&key], FlowNode::Tool(_) | FlowNode::AgentTool(_) | FlowNode::FlowTool { .. }) {
                continue;
            }
            if !can_reach_terminal.contains(&key) {
                problems.push(format!(
                    "node '{}': has no path to any terminal — dead end",
                    graph.interner.name_of(key)
                ));
            }
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(BuildError::Invalid(problems))
    }
}


