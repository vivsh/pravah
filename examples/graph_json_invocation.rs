//! Drive a trusted workflow through stateless JSON operations.
//!
//! No services are required. The next request always carries the returned snapshot.

mod support;

use pravah::graph::{Flow, JSON_WIRE_VERSION as VERSION, JsonInvoker, JsonRequest, JsonResponse};
use support::ExampleError;

/// Passes each returned snapshot to the next operation, including approval delivery.
fn main() -> Result<(), ExampleError> {
    let workflow = Flow::<String>::root()
        .suspend::<bool>()
        .finish::<String>()?;
    let (graph, handlers) = workflow.into_parts();
    let invoker = JsonInvoker::new(graph, handlers)?;
    let mut request = JsonRequest::Start {
        version: VERSION,
        execution_id: uuid::Uuid::now_v7(),
        input: serde_json::json!("Publish the report?"),
    };

    loop {
        request = match invoker.invoke(request)? {
            JsonResponse::Continue { snapshot, .. } => JsonRequest::Next {
                version: VERSION,
                snapshot,
            },
            JsonResponse::Suspend {
                snapshot, payload, ..
            } => {
                println!("Approval request: {payload}");
                JsonRequest::Resume {
                    version: VERSION,
                    snapshot,
                    input: serde_json::json!(true),
                }
            }
            JsonResponse::Done { output, .. } => {
                println!("Approved: {output}");
                return Ok(());
            }
            JsonResponse::Agent { .. } => {
                return Err(ExampleError::from("approval requested an external effect"));
            }
        };
    }
}
