use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::*;
use pravah::clients::{Client, ClientError, ClientOptions, ModelUrl, ProviderFactory, ToolChoice};
use pravah::testing::mock_tool_call;
use pravah::tools::ToolError;
use pravah::{AgentDecision, AgentInterventionPoint, AgentLoop, Flow, Toolset, compile};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Lookup {
    query: String,
}

async fn lookup(input: Lookup, _ctx: Context) -> Result<String, ToolError> {
    Ok(input.query)
}

fn toolset(tools: Toolset) -> Toolset {
    tools.tool(lookup)
}

fn researcher(root: Agent<Question>) -> Agent<Answer> {
    root.tools(toolset).control(control).configure(configure)
}

fn workflow(root: Flow<Question>) -> Flow<Answer> {
    root.agent(researcher)
}

/// Supplies runtime memory and provider settings alongside a single ordinary model turn.
async fn configure(question: Question, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test",
        "Answer briefly.",
        Message::user(question.text).with_key("research:42"),
    )
    .memory("known preference")
    .provider_config(serde_json::json!({"setting": true}))
    .turn_budget(1))
}

/// Supplies guidance at both model boundaries without changing the hard turn budget.
async fn control(loop_: AgentLoop<Question>, _ctx: Context) -> Result<AgentDecision, GraphError> {
    Ok(match loop_.point() {
        AgentInterventionPoint::BeforeModel => {
            AgentDecision::redirect().guidance("look up evidence")
        }
        AgentInterventionPoint::AfterTools => {
            AgentDecision::redirect().guidance("finish using evidence")
        }
        AgentInterventionPoint::BeforeTools => AgentDecision::continue_(),
    })
}

struct OverrideFactory(ScriptedFactory);

impl ProviderFactory for OverrideFactory {
    async fn llm(
        &self,
        model: &ModelUrl,
        mut options: ClientOptions,
    ) -> Result<Client, ClientError> {
        options.temperature = Some(0.25);
        self.0.llm(model, options).await
    }
}

#[derive(Clone, Default)]
struct ObserveTools {
    calls: Arc<AtomicUsize>,
    reject_tools: bool,
}

impl Compactor for ObserveTools {
    type Error = std::convert::Infallible;

    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(request.committed().is_empty());
        assert_eq!(
            request.protected()[0].message.key.as_deref(),
            Some("research:42")
        );
        assert!(
            request.protected()[1..]
                .iter()
                .all(|entry| entry.message.key.is_none())
        );
        verify_options(request.options());
        if call == 0 {
            assert_eq!(request.options().tools.len(), 1);
            assert_eq!(request.protected().len(), 1);
            assert!(
                request.framework_messages()[0]
                    .content
                    .contains("look up evidence")
            );
        } else {
            assert_eq!(request.protected().len(), 4);
            assert!(matches!(
                request.protected()[1].message.role,
                Role::AssistantToolCalls { .. }
            ));
            assert!(request.options().tools.is_empty());
            assert!(matches!(
                request.options().tool_choice,
                ToolChoice::Disabled
            ));
            assert_eq!(request.framework_messages().len(), 2);
            assert!(
                request.framework_messages()[0]
                    .content
                    .contains("finish using evidence")
            );
            assert!(matches!(request.framework_messages()[1].role, Role::User));
        }
        Ok(CompactionResult {
            evict_indices: if call > 0 && self.reject_tools {
                vec![0]
            } else {
                Vec::new()
            },
            summary: None,
        })
    }
}

/// Verifies that the policy sees resolved memory and the actual factory-adjusted client options.
fn verify_options(options: &ClientOptions) {
    assert_eq!(options.temperature, Some(0.25));
    assert_eq!(
        options.provider_config,
        Some(serde_json::json!({"setting":true}))
    );
    assert!(
        options
            .preamble
            .as_deref()
            .is_some_and(|p| p.contains("<memory>\nknown preference"))
    );
    assert!(matches!(
        options.response_format,
        pravah::clients::ResponseFormat::JsonSchema { .. }
    ));
}

fn scripted_tools() -> ScriptedFactory {
    ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("a", "lookup", serde_json::json!({"query":"first"})),
            mock_tool_call("b", "lookup", serde_json::json!({"query":"second"})),
        ])
        .then_output(serde_json::json!({"text":"finished"}))
}

/// Preparation sees factory overrides, effective tools and exact controller/conclusion guidance.
#[tokio::test]
async fn request_view_matches_tool_loop_and_conclusion() -> Result<(), GraphError> {
    let factory = scripted_tools();
    let policy = ObserveTools::default();
    let mut chat = Chat::new(
        researcher,
        Context::default().with_providers(pravah::testing::providers(OverrideFactory(
            factory.clone(),
        ))?),
    )?
    .with_compactor(policy.clone());
    chat.send(Question {
        text: "research".into(),
    })
    .await?;
    assert_eq!(factory.calls().len(), 2);
    assert_eq!(policy.calls.load(Ordering::SeqCst), 2);
    let calls = factory.calls();
    assert_eq!(calls[1].1.len(), 6);
    assert!(matches!(calls[1].1[2].role, Role::Tool { .. }));
    assert!(matches!(calls[1].1[3].role, Role::Tool { .. }));
    assert!(calls[1].1[4].content.contains("finish using evidence"));
    Ok(())
}

/// The runtime rejects eviction of an active tool exchange atomically before redispatch.
#[tokio::test]
async fn tool_loop_rejects_protected_eviction() -> Result<(), GraphError> {
    let factory = scripted_tools();
    let flow = compile(workflow)?;
    let executor = flow
        .prepared()
        .executor(
            Context::default().with_providers(pravah::testing::providers(OverrideFactory(
                factory.clone(),
            ))?),
        )
        .with_compactor(ObserveTools {
            reject_tools: true,
            ..ObserveTools::default()
        });
    let mut runtime = flow.start(
        Question {
            text: "research".into(),
        },
        uuid::Uuid::nil(),
    )?;
    for _ in 0..100 {
        let before = serde_json::to_value(runtime.snapshot()?).expect("snapshot");
        match host::step(&mut runtime, &executor).await {
            Err(GraphError::HistoryCompactionValidation { .. }) => {
                assert_eq!(
                    serde_json::to_value(runtime.snapshot()?).expect("snapshot"),
                    before
                );
                assert_eq!(factory.calls().len(), 1);
                return Ok(());
            }
            Err(error) => return Err(error),
            Ok(_) => {}
        }
    }
    panic!("expected protected-entry rejection")
}
