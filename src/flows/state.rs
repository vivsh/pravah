use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::clients::ToolCall;

/// Mutable portion of the flow engine
#[derive(Serialize, Deserialize, Clone)]
pub struct FlowState {
    states: IndexMap<String, Value>,
    agent_initialized: bool,
    pending_calls: Option<Vec<ToolCall>>,
    suspension: Option<String>,
}

impl FlowState {
    pub fn new() -> Self {
        Self {
            states: IndexMap::new(),
            agent_initialized: false,
            pending_calls: None,
            suspension: None,
        }
    }

    pub fn suspension(&self) -> Option<&String> {
        self.suspension.as_ref()
    }

    pub fn clear_suspension(&mut self) {
        self.suspension = None;
        self.pending_calls = None;
    }

    pub fn suspend(&mut self, node_id: &str, calls: Option<Vec<ToolCall>>) {
        self.suspension = Some(node_id.to_string());
        self.pending_calls = calls;
    }

    pub fn contains_state(&self, state_name: &str) -> bool {
        self.states.contains_key(state_name)
    }

    pub fn agent_started(&mut self) -> bool {
        let old = self.agent_initialized;
        self.agent_initialized = true;
        old
    }

    pub fn agent_stopped(&mut self) -> bool {
        let old = self.agent_initialized;
        self.agent_initialized = false;
        old
    }

    pub fn set_state(&mut self, new_node: &str, value: Value, old_node: Option<&str>) {
        self.states.insert(new_node.to_string(), value);
        if let Some(old_node) = old_node {
            self.states.shift_remove(old_node);
        }
    }

    pub fn remove_state(&mut self, state_name: &str) {
        self.states.shift_remove(state_name);
    }

    pub fn get_state(&self, node_id: &str) -> Option<&Value> {
        self.states.get(node_id)
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn get_index(&self, i: usize) -> Option<(&String, &Value)> {
        self.states.get_index(i)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.states.keys()
    }

    pub fn take_pending_calls(&mut self) -> Option<Vec<ToolCall>> {
        self.pending_calls.take()
    }
}
