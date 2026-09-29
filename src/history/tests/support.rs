use crate::clients::{Message, Role, TokenUsage, ToolCall};
use crate::history::MessageHistory;

/// Creates complete token usage for history accounting tests.
pub(super) fn usage(input: u32, output: u32) -> TokenUsage {
    {
        let mut usage = TokenUsage::default();
        usage.input = Some(input);
        usage.output = Some(output);
        usage
    }
}

/// Appends one assistant message with optional usage.
pub(super) fn push_assistant(
    history: &mut MessageHistory,
    session: &str,
    usage: Option<TokenUsage>,
) {
    let mut message = Message::assistant("hi");
    if let Some(usage) = usage {
        message = message.with_usage(usage);
    }
    history.push(session, "agent", message);
}

/// Appends one assistant proposal containing tool calls.
pub(super) fn push_tool_calls(history: &mut MessageHistory, session: &str, calls: Vec<ToolCall>) {
    history.push(
        session,
        "agent",
        Message::new(Role::AssistantToolCalls { calls }, String::new()),
    );
}

/// Appends one tool result for a prior proposal.
pub(super) fn push_tool(history: &mut MessageHistory, session: &str, call_id: &str) {
    history.push(session, "agent", Message::tool_output(call_id.into(), "ok"));
}

/// Appends one user message.
pub(super) fn push_user(history: &mut MessageHistory, session: &str, content: &str) {
    history.push(session, "agent", Message::user(content));
}

/// Creates a deterministic tool call fixture.
pub(super) fn tool_call(id: &str) -> ToolCall {
    ToolCall::new(id.into(), "f".into(), serde_json::json!({}))
}
