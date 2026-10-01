# Driving external requests

A Fetch pauses execution with a buffered, serializable request. The host performs the operation and delivers its response or failure. The VM itself steps synchronously.

```rust
use pravah::{FetchError, FetchRequest, FetchResponse, Flow, GraphError, Step, compile};
use uuid::Uuid;

fn request(input: Flow<FetchRequest>) -> Flow<Result<FetchResponse, FetchError>> {
    input.fetch()
}

fn demonstrate_delivery(execution_id: Uuid) -> Result<(), GraphError> {
    let workflow = compile(request)?;
    let mut execution = workflow.start(
        FetchRequest::new("GET", "https://example.com/status"),
        execution_id,
    )?;
    let fetch = loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Fetch(fetch) => break fetch,
            Step::Suspend(_) | Step::Done(_) => {
                return Err(GraphError::FetchValidation("unexpected boundary".into()));
            }
        }
    };
    let checkpoint = execution.snapshot()?;
    // Persist checkpoint before executing fetch externally.
    let mut execution = workflow.restore(checkpoint)?;
    // A synthetic response illustrates delivery; this example makes no HTTP call.
    execution.resume_fetch(fetch.id(), Ok(FetchResponse::new(200)))?;
    // Call next() to process the accepted outcome and continue the flow.
    Ok(())
}
```

Choose one UUID for each independent execution and persist it with that execution.
See [the runnable Fetch example](../examples/graph_fetch.rs) for a complete
response and failure round trip without network access.

For a worker that has a persisted Fetch but no graph, construct an executor once
with `FetchExecutor::new(ctx)` and call `executor.execute(&fetch).await`. This
supports HTTP and Rath requests; register application schemes on the executor
when needed. Persistence and compaction are not Fetch operations; use a caller-owned
`HistoryManager` as described in the [history guide](history.md).

Agent and tool hooks need their matching Rust handlers. Install an
`Arc<HandlerRegistry>` with `.with_registry(registry)`, or obtain an executor
from `workflow.prepared().executor(ctx)` before releasing the workflow. The
executor can then run independently of the graph or VM. Missing handlers and
unknown schemes fail explicitly. The executor owns execution dependencies, never history or maintenance state.

To select a Rath LLM work lane before execution, decode
`pravah::graph::fetch::rath::RathRequest::from_fetch_request(fetch.request())`
and inspect its borrowed
`provider()` and full `model()` URL. The provider is the URL's logical scheme,
not necessarily the physical provider: for example, `claude:` is a Rath alias
for Anthropic. If a custom factory overrides its destination, keep the host's
lane mapping in agreement with that factory. Rath Fetch does not represent image
or video work.

HTTP error statuses are responses, not transport failures. Deliver a transport or execution failure with `Err(FetchError::new(code, message))`. `FetchError::from_execution_error(&error)` explicitly converts an executor error to portable diagnostic data. Inspect original typed Rath errors before conversion if the application needs their Rust source chain.

Acceptance and execution are separate: `resume_fetch` validates and records delivery; `next` processes it. An accepted outcome survives a later processing error and must not be dispatched again. Invalid or duplicate delivery is rejected. There is only one pending external request per execution.

Malformed Rath request envelopes are rejected before becoming pending requests, and checked again on restoration. Responses are validated before acceptance. With no compactor or budget-conclusion reminder, client construction happens at generation; a construction failure leaves that generation request pending. Compactors still receive the provider's effective request options, which requires client construction during preparation as well as generation.

## Durable host responsibilities

1. Persist the pending snapshot before dispatch.
2. Use the Fetch UUID as an idempotency key where the external service supports it.
3. Durably record the outcome before delivering it to the VM.
4. Accept the recorded outcome and persist the advanced snapshot with a fenced or compare-and-swap update.
5. On recovery, check for a recorded outcome before dispatching the request again.

A crash between an external effect and completion recording remains uncertain. The host must use idempotency, reconciliation or a deliberate duplicate-effect policy. Pravah does not provide exactly-once effects, automatic retries, queues or leases.

Requests and snapshots may contain credentials, headers, messages and other sensitive data. Secure them as application data. Runtime dependencies are not serialized; reinstall them when restoring. A pending Rath request freezes its logical inputs, not credentials, factory code or provider behavior.
