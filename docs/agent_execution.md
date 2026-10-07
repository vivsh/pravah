# Agent workers and durable delivery

The synchronous VM emits `Step::Agent(request)` when an agent needs asynchronous
work. An `AgentRequest` contains the operation's durable inputs; an `AgentExecutor`
holds only runtime dependencies. A worker does not need access to the parent VM.

```rust
let executor = workflow.prepared().executor(ctx)
    .with_store(store)
    .with_compactor(compactor);
let mut execution = workflow.start(input, execution_id)?
    .with_history(HistoryPolicy { persist: true, load: true, compact: true })?;

loop {
    match execution.next()? {
        Step::Continue => {}
        Step::Agent(request) => {
            save_snapshot(execution.snapshot()?).await?;
            let response = executor.execute(&request).await;
            save_completion(&response).await?;
            execution.resume_agent(response)?;
        }
        Step::Suspend(payload) => {
            save_snapshot(execution.snapshot()?).await?;
            present(payload);
            break;
        }
        Step::Done(output) => {
            complete(workflow.decode_output(output)?);
            break;
        }
    }
}
```

`execute` returns an `AgentResponse`, including portable failure diagnostics and
acknowledgements of any stages that succeeded before failure. It does not retry
or fall back to another provider. `resume_agent(response)` validates delivery and
commits those acknowledgements; `next()` subsequently processes the outcome.
An accepted outcome survives processing errors and restoration without being
dispatched again. Invalid identities, duplicate deliveries and malformed
acknowledgements are rejected without mutation.

## Independent workers and routing

Retain `workflow.prepared().executor(ctx)` after dropping the workflow, or use
`AgentExecutor::new(ctx).with_registry(registry)` with a compatible shared
`Arc<HandlerRegistry>`. Reinstall stores and compactors on each worker as required.
Generation can execute without a registry; configuration, controller and tool
callbacks need their registered Rust implementations. Missing dependencies fail
explicitly. No callback code is reconstructed from serialized requests.

Use borrowed `request.id()`, `kind()`, `model()`, `provider()`, `handler()` and
`conversation_key()` for task routing. Model and provider are available for
generation, not necessarily for configuration or tools. The provider is the
model URL's logical scheme, not the physical destination of a custom factory.
Keep host lane mappings consistent with that factory. Requests cover LLM agent
work; ordinary HTTP, media and application tasks use typed suspension instead.

Configuration and resource resolution may share one operation with keyed history
loading. Generation persists accepted original rows, prepares working history
when enabled, and calls the model. Final messages are flushed before completion
when persistence is enabled. Workers never save an unaccepted model proposal.
See [history usage](history.md) for policy and store contracts.

## Durable host responsibilities

1. Persist the pending snapshot before dispatch.
2. Use the request UUID for external deduplication where supported.
3. Durably record the complete response before delivering it.
4. Persist the advanced snapshot with a fenced or compare-and-swap update.
5. On recovery, deliver recorded completion before considering redispatch.

There is one outstanding request per runtime. A crash after an external effect
but before completion recording remains uncertain: use idempotency, reconciliation
or an explicit duplicate-effect policy. Pravah provides neither exactly-once
effects nor a queue, lease store or automatic retry.

Snapshots contain working history and accepted outcomes, not live dependencies.
Requests freeze logical inputs, not credentials, factory code or provider revisions.
Supplied data and portable diagnostics may be sensitive; storage and access policy
belong to the application. Default request/error formatting omits payloads.

For a complete synchronous non-agent example, see
[external task suspension](../examples/graph_external_task.rs). Async `Chat::send`
and the manual Chat interface use these same delivery rules.
