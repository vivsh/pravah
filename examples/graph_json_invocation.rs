//! Stateless JSON invocation of a trusted graph through completion.
//!
//! This example is deterministic and requires no external services.

use pravah::graph::{Flow, GraphError, JSON_WIRE_VERSION, JsonInvoker, JsonRequest, JsonResponse};
use pravah::{Context, FlowConf};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), GraphError> {
    let flow = Flow::<i64>::root()
        .map_named("prepare", |value| value + 1)
        .suspend::<i64>()
        .finish::<i64>()?;
    let (graph, registry) = flow.into_parts();
    let invoker = JsonInvoker::new(graph, registry)?;
    let ctx = Context::new(FlowConf::default());

    let JsonResponse::Continue { snapshot, .. } = invoker
        .invoke(
            JsonRequest::Start {
                version: JSON_WIRE_VERSION,
                input: json!(40),
            },
            ctx.clone(),
        )
        .await?
    else {
        return Err(GraphError::Invalid(
            "JSON start did not advance exactly one step".into(),
        ));
    };
    let JsonResponse::Suspend { snapshot, .. } = invoker
        .invoke(
            JsonRequest::Next {
                version: JSON_WIRE_VERSION,
                snapshot,
            },
            ctx.clone(),
        )
        .await?
    else {
        return Err(GraphError::Invalid(
            "JSON next did not reach suspension".into(),
        ));
    };
    let response = invoker
        .invoke(
            JsonRequest::Resume {
                version: JSON_WIRE_VERSION,
                snapshot,
                input: json!(42),
            },
            ctx,
        )
        .await?;
    let JsonResponse::Done { output, .. } = response else {
        return Err(GraphError::Invalid("JSON resume did not complete".into()));
    };
    println!("{output}");
    Ok(())
}
