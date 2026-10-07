use super::*;
use pravah::graph::{NodeKind, PreparedGraph, to_value};
use pravah::tools::ToolError;
use pravah::{Agent, AgentConfig, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Batch {
    values: Vec<String>,
}

async fn echo(input: Batch, _: Context) -> Result<Batch, ToolError> {
    Ok(input)
}

fn tools(tools: Toolset) -> Toolset {
    tools.tool(echo)
}

async fn configure(input: String, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///test",
        "test",
        Message::user(input),
    ))
}

fn agent(root: Agent<String>) -> Agent<String> {
    root.tools(tools).configure(configure)
}

fn flow(root: Flow<String>) -> Flow<String> {
    root.agent(agent)
}

/// Prepares the authored tool child directly so no model or agent work enters allocation counts.
fn tool_graph() -> Result<PreparedGraph, GraphError> {
    let workflow = compile(flow)?;
    let graph = workflow
        .graph()
        .nodes
        .iter()
        .find_map(|node| match &node.kind {
            NodeKind::Continuation { children, .. } => children.first(),
            _ => None,
        })
        .ok_or_else(|| GraphError::Invalid("missing tool child".into()))?;
    PreparedGraph::new(graph.clone(), workflow.registry().clone())
}

/// Constructing a tool request shares its input and costs the same for small and large batches.
#[test]
fn tool_request_creation_does_not_reencode_input() -> Result<(), GraphError> {
    let graph = tool_graph()?;
    let mut allocations = Vec::new();
    for size in [1, 1000] {
        let input = to_value(Batch {
            values: vec!["value".repeat(100); size],
        })
        .map_err(codec)?;
        let mut runtime = graph.start(input.clone(), Uuid::from_u128(1))?;
        let mut result = Ok(Step::Continue);
        let measured = allocation_counter::measure(|| {
            result = runtime.next();
        });
        let Step::Agent(fetch) = result? else {
            return Err(GraphError::Invalid("missing tool AgentRequest".into()));
        };
        assert_eq!(fetch.kind(), "tool");
        allocations.push((measured.count_total, measured.bytes_total));
    }
    assert_eq!(allocations[0], allocations[1]);
    Ok(())
}

/// The accepted tool response keeps its shared result while the continuation completes.
#[tokio::test]
async fn tool_response_delivery_does_not_reencode_output() -> Result<(), GraphError> {
    let graph = tool_graph()?;
    let input = to_value(Batch {
        values: vec!["data".repeat(100); 1000],
    })
    .map_err(codec)?;
    let mut runtime = graph.start(input, Uuid::from_u128(1))?;
    let fetch = next_agent(&mut runtime)?;
    let response = graph.executor(Context::default()).execute(&fetch).await;
    let retained = response
        .outcome()
        .map_err(|_| codec("tool failure"))?
        .clone();
    runtime.resume_agent(response)?;
    loop {
        match runtime.next()? {
            Step::Continue => {}
            Step::Done(output) => {
                assert_eq!(output, retained);
                let expected = retained
                    .get("value")
                    .and_then(|value| value.get("values"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| GraphError::Invalid("missing result array".into()))?;
                let actual = output
                    .get("value")
                    .and_then(|value| value.get("values"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| GraphError::Invalid("missing output array".into()))?;
                assert!(std::ptr::eq(expected, actual));
                break;
            }
            _ => return Err(GraphError::Invalid("unexpected tool boundary".into())),
        }
    }
    Ok(())
}
