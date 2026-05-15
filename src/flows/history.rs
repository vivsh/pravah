use crate::clients::{ClientError, Message, Role, TokenUsage, ToolCall};

/// Conversation history for a flow, with optional sliding-window eviction and token accounting.
///
/// Usage is auto-extracted from pushed [`Message`]s that carry a `usage` field.
/// The first `pinned` messages are never evicted (default: 1, to preserve the
/// initial task/seed User message).
#[derive(Debug, Clone, Default)]
pub struct FlowHistory {
    messages: Vec<Message>,
    max_turns: Option<usize>,
    pinned: usize,
    last_usage: Option<TokenUsage>,
    total_input: Option<u32>,
    total_output: Option<u32>,
}

impl FlowHistory {
    pub fn new(max_turns: Option<usize>) -> Self {
        Self {
            max_turns,
            pinned: 1,
            ..Default::default()
        }
    }

    pub fn with_pinned(max_turns: Option<usize>, pinned: usize) -> Self {
        Self {
            max_turns,
            pinned,
            ..Default::default()
        }
    }

    /// Appends `msg`, auto-extracts its token usage, then evicts if over the turn limit.
    pub fn push(&mut self, msg: Message) {
        if let Some(u) = msg.usage {
            self.last_usage = Some(u);
            self.total_input = add_opt(self.total_input, u.input);
            self.total_output = add_opt(self.total_output, u.output);
        }
        self.messages.push(msg);
        self.evict_if_needed();
    }

    pub fn extend(&mut self, msgs: impl IntoIterator<Item = Message>) {
        for msg in msgs {
            self.push(msg);
        }
    }

    pub fn as_slice(&self) -> &[Message] {
        &self.messages
    }

    pub fn last_role(&self) -> Option<&Role> {
        self.messages.last().map(|m| &m.role)
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Number of complete assistant turns after the pinned region.
    pub fn turn_count(&self) -> usize {
        let start = self.pinned.min(self.messages.len());
        self.messages[start..]
            .iter()
            .filter(|m| matches!(m.role, Role::Assistant | Role::AssistantToolCalls { .. }))
            .count()
    }

    pub fn validate(&self) -> Result<(), ClientError> {
        if matches!(self.last_role(), Some(Role::AssistantToolCalls { .. })) {
            return Err(ClientError::Validation(
                "history ends with assistant tool calls without tool results".into(),
            ));
        }
        Ok(())
    }

    pub fn last_usage(&self) -> Option<TokenUsage> {
        self.last_usage
    }

    pub fn total_input(&self) -> Option<u32> {
        self.total_input
    }

    pub fn total_output(&self) -> Option<u32> {
        self.total_output
    }

    /// Cumulative input + output tokens, or `None` if either is missing.
    pub fn total_usage(&self) -> Option<u32> {
        match (self.total_input, self.total_output) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        }
    }

    /// Returns a filtered slice of messages visible to the given session.
    ///
    /// Untagged messages (`session_id` is empty) are always included — they represent
    /// shared context such as the initial seed message. Passing an empty `session_id`
    /// returns all messages.
    pub fn for_session(&self, session_id: &str) -> Vec<Message> {
        if session_id.is_empty() {
            return self.messages.clone();
        }
        self.messages
            .iter()
            .filter(|m| m.session_id.is_empty() || m.session_id == session_id)
            .cloned()
            .collect()
    }

    fn first_turn_end_exclusive(&self) -> Option<usize> {
        // Scan forward from `pinned` to find the first assistant turn.
        // This tolerates histories loaded via `with_history` where the message
        // at index `pinned` may not be an assistant turn.
        let start = (self.pinned..self.messages.len()).find(|&i| {
            matches!(
                self.messages[i].role,
                Role::Assistant | Role::AssistantToolCalls { .. }
            )
        })?;
        match &self.messages[start].role {
            Role::AssistantToolCalls { .. } => {
                let mut end = start + 1;
                while end < self.messages.len()
                    && matches!(self.messages[end].role, Role::Tool { .. })
                {
                    end += 1;
                }
                Some(end)
            }
            Role::Assistant => Some(start + 1),
            _ => unreachable!(),
        }
    }

    fn evict_if_needed(&mut self) {
        let max = match self.max_turns {
            Some(m) => m,
            None => return,
        };
        while self.turn_count() > max {
            match self.first_turn_end_exclusive() {
                Some(end) => {
                    self.messages.drain(self.pinned..end);
                }
                None => break,
            }
        }
    }
}

fn add_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input: Some(input),
            output: Some(output),
        }
    }

    fn msg_with_usage(u: TokenUsage) -> Message {
        Message {
            role: Role::Assistant,
            content: "hi".into(),
            usage: Some(u),
            agent_id: String::new(),
            session_id: String::new(),
        }
    }

    fn atc_msg(calls: Vec<ToolCall>) -> Message {
        Message {
            role: Role::AssistantToolCalls { calls },
            content: String::new(),
            usage: None,
            agent_id: String::new(),
            session_id: String::new(),
        }
    }

    fn tool_msg(call_id: &str) -> Message {
        Message {
            role: Role::Tool {
                call_id: call_id.into(),
            },
            content: "ok".into(),
            usage: None,
            agent_id: String::new(),
            session_id: String::new(),
        }
    }

    fn dummy_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "f".into(),
            args: serde_json::json!({}),
            thought_signatures: None,
        }
    }

    /// `push` records the most recently seen token usage.
    #[test]
    fn push_records_last_usage() {
        let mut h = FlowHistory::new(None);
        h.push(msg_with_usage(usage(10, 5)));
        let u = h.last_usage().unwrap();
        assert_eq!(u.input, Some(10));
        assert_eq!(u.output, Some(5));
    }

    /// `push` accumulates total input and output tokens across all messages.
    #[test]
    fn push_accumulates_totals() {
        let mut h = FlowHistory::new(None);
        h.push(msg_with_usage(usage(10, 5)));
        h.push(msg_with_usage(usage(20, 8)));
        assert_eq!(h.total_input(), Some(30));
        assert_eq!(h.total_output(), Some(13));
        assert_eq!(h.total_usage(), Some(43));
    }

    /// `turn_count` counts only assistant turns after the pinned region.
    #[test]
    fn turn_count_counts_assistant_turns() {
        let mut h = FlowHistory::new(None);
        h.push(Message::user("seed"));
        h.push(atc_msg(vec![dummy_call("1")]));
        h.push(tool_msg("1"));
        assert_eq!(h.turn_count(), 1);
        h.push(atc_msg(vec![dummy_call("2")]));
        h.push(tool_msg("2"));
        assert_eq!(h.turn_count(), 2);
    }

    /// Sliding window evicts the oldest non-pinned turn when limit is exceeded.
    #[test]
    fn sliding_evicts_oldest_turn() {
        let mut h = FlowHistory::new(Some(1));
        h.push(Message::user("seed"));
        h.push(atc_msg(vec![dummy_call("1")]));
        h.push(tool_msg("1"));
        h.push(atc_msg(vec![dummy_call("2")]));
        h.push(tool_msg("2"));
        assert_eq!(h.turn_count(), 1);
        assert!(matches!(h.as_slice()[0].role, Role::User));
    }

    /// Pinned messages are never evicted even when the window is exceeded.
    #[test]
    fn pinned_messages_survive_eviction() {
        let mut h = FlowHistory::with_pinned(Some(1), 2);
        h.push(Message::user("seed"));
        h.push(Message::user("ctx"));
        h.push(atc_msg(vec![dummy_call("1")]));
        h.push(tool_msg("1"));
        h.push(atc_msg(vec![dummy_call("2")]));
        h.push(tool_msg("2"));
        assert_eq!(h.as_slice().len(), 4);
        assert!(matches!(h.as_slice()[0].role, Role::User));
        assert!(matches!(h.as_slice()[1].role, Role::User));
    }

    /// `validate` returns an error when history ends with unresolved tool calls.
    #[test]
    fn validate_rejects_dangling_tool_calls() {
        let mut h = FlowHistory::new(None);
        h.push(Message::user("seed"));
        h.push(atc_msg(vec![dummy_call("1")]));
        assert!(matches!(h.validate(), Err(ClientError::Validation(_))));
    }

    /// `last_role` returns the role of the most recently pushed message.
    #[test]
    fn last_role_returns_correct_role() {
        let mut h = FlowHistory::new(None);
        assert!(h.last_role().is_none());
        h.push(Message::user("hi"));
        assert!(matches!(h.last_role(), Some(Role::User)));
    }

    /// `total_usage` returns `None` when either input or output is missing.
    #[test]
    fn total_usage_requires_both_values() {
        let mut h = FlowHistory::new(None);
        h.push(Message {
            role: Role::Assistant,
            content: "x".into(),
            usage: Some(TokenUsage {
                input: Some(5),
                output: None,
            }),
            agent_id: String::new(),
            session_id: String::new(),
        });
        assert_eq!(h.total_input(), Some(5));
        assert_eq!(h.total_output(), None);
        assert_eq!(h.total_usage(), None);
    }
}
