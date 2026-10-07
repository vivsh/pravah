use super::*;
use pravah::{Toolset, tools::ToolError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Search {
    query: String,
}

fn tools(root: Toolset) -> Toolset {
    root.tool(search)
}
async fn search(input: Search, _: Context) -> Result<String, ToolError> {
    Ok(input.query)
}

struct ToolFactory;
struct ToolModel {
    model: ModelUrl,
    options: ClientOptions,
}
impl ProviderFactory for ToolFactory {
    async fn llm(&self, model: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        Ok(Client::from_backend(ToolModel {
            model: model.clone(),
            options,
        }))
    }
}
impl LlmBackend for ToolModel {
    fn model_url(&self) -> &ModelUrl {
        &self.model
    }
    fn options(&self) -> &ClientOptions {
        &self.options
    }
    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let output = if messages
            .last()
            .is_some_and(|m| matches!(m.role, pravah::clients::Role::Tool { .. }))
        {
            ClientOutput::Output(serde_json::json!("answer"))
        } else {
            ClientOutput::ToolCalls {
                text: None,
                calls: vec![pravah::testing::mock_tool_call(
                    "search-call",
                    "search",
                    serde_json::json!({"query":"question"}),
                )],
            }
        };
        Ok(ClientResponse::new(Provider::OpenAi, output))
    }
}

/// Measures a complete two-generation turn with a tool, persistence and protected-exchange compaction.
pub(super) fn run(rt: &tokio::runtime::Runtime) -> Result<(), GraphError> {
    let context = Context::default().with_providers(pravah::testing::providers(ToolFactory)?);
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .instructions("Search once, then answer.")
        .tools(tools)
        .store(pravah::history::NoopHistoryStore)
        .compactor(BoundHistory)
        .build(context)?;
    super::cases::measure(
        rt,
        "chat/tool_compaction_persistence",
        &mut chat,
        || "question".to_owned(),
        false,
    )
}
