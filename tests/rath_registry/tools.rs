use super::*;
use pravah::clients::ToolChoice;
use pravah::testing::mock_tool_call;
use pravah::{ChatBuilder, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Serialize, Deserialize, JsonSchema)]
#[schemars(rename = "echo")]
struct Echo {
    text: String,
}

async fn echo(input: Echo, _: Context) -> Result<String, pravah::tools::ToolError> {
    Ok(input.text)
}

fn tools(root: Toolset) -> Toolset {
    root.tool(echo)
}

fn definition() -> ChatBuilder<String, String> {
    Chat::builder::<String, String>()
        .model("openai:///recorded-model")
        .tools(tools)
        .tool_budget::<Echo>(1)
        .turn_budget(1)
        .max_output_tokens(128)
}

struct ToolFactory {
    script: ScriptedFactory,
    calls: Arc<AtomicUsize>,
}

impl ProviderFactory for ToolFactory {
    /// Both dispatches retain the output cap while conclusion removes only the domain tools.
    async fn llm(&self, url: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        let dispatch = self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(options.max_output_tokens, Some(128));
        match dispatch {
            0 => {
                assert_eq!(
                    options
                        .tools
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>(),
                    ["echo"]
                );
                assert!(matches!(options.tool_choice, ToolChoice::Auto));
            }
            1 | 2 => {
                assert!(options.tools.is_empty());
                assert!(matches!(options.tool_choice, ToolChoice::Disabled));
            }
            _ => {
                return Err(ClientError::new(
                    pravah::clients::ErrorKind::Validation,
                    "unexpected dispatch",
                ));
            }
        }
        self.script.llm(url, options).await
    }
}

/// Recorded built-in replies use real tool execution and the ordinary forced-conclusion boundary.
#[tokio::test]
async fn injected_tools_and_budgets_restore_without_production_dispatch() -> Result<(), TestError> {
    let script = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "call-1",
            "echo",
            json!({"text": "evidence"}),
        )])
        .then_output(json!("answer"));
    let calls = Arc::new(AtomicUsize::new(0));
    let ctx =
        Context::default().with_providers(ProviderRegistry::with_builtin_factory(ToolFactory {
            script: script.clone(),
            calls: calls.clone(),
        }));
    let mut chat = definition().build(ctx)?;
    assert_eq!(
        chat.send_with_key("question", "message-1").await?.output,
        "answer"
    );
    // Conclusion preparation and generation construct separate clients; only two execute.
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let recorded = script.calls();
    assert_eq!(recorded.len(), 2);
    let (_, messages) = recorded
        .last()
        .ok_or(TestError::Missing("conclusion request"))?;
    assert!(messages.iter().any(
        |m| matches!(&m.role, pravah::clients::Role::Tool {call_id} if call_id == "call-1")
            && m.content.contains("evidence")
    ));
    assert!(messages.iter().any(|m| m.content.contains("FINAL TURN")));
    let snapshot = chat.snapshot()?;
    let before = serde_json::to_value(&snapshot)?;
    let restored = definition().restore::<()>(snapshot, Context::default())?;
    assert_eq!(before, serde_json::to_value(restored.snapshot()?)?);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    Ok(())
}
