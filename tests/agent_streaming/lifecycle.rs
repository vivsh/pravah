use super::chat::builder;
use super::*;
use futures::Future;
use std::sync::atomic::Ordering;

/// Awaited callbacks prevent polling the next event; cancellation changes no VM history or checkpoint.
#[tokio::test]
async fn callback_backpressure_is_operation_local() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let flow = compile(workflow)?;
    let executor = flow
        .prepared()
        .executor(context(Mode::Complete, stats.clone())?);
    let mut runtime = flow.start("question".into(), Uuid::nil())?;
    let request = generation(&mut runtime, &executor).await?;
    let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    let mut seen = 0;
    let mut execution = Box::pin(executor.execute_stream(&request, |_, _| {
        seen += 1;
        std::future::pending::<()>()
    }));
    let result =
        futures::future::poll_fn(|cx| std::task::Poll::Ready(execution.as_mut().poll(cx))).await;
    assert!(result.is_pending());
    assert_eq!(stats.polls.load(Ordering::SeqCst), 1);
    drop(execution);
    assert_eq!(seen, 1);
    assert_eq!(
        before,
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?
    );
    assert_eq!(
        runtime.pending_agent().map(AgentRequest::id),
        Some(request.id())
    );
    Ok(())
}

/// A cancelled stream leaves a restorable pending request, not a resumable transport or accepted preview.
#[tokio::test]
async fn cancelled_chat_restores_for_explicit_worker_execution() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let mut chat = builder().build(context(Mode::Complete, stats.clone())?)?;
    let mut send = Box::pin(chat.send_stream("question", |_, _| std::future::pending::<()>()));
    assert!(
        futures::future::poll_fn(|cx| std::task::Poll::Ready(send.as_mut().poll(cx)))
            .await
            .is_pending()
    );
    drop(send);
    assert_eq!(chat.snapshot()?.history().entries().len(), 1);
    assert!(chat.pending_agent().is_some());
    assert!(matches!(
        chat.send_stream("new", |_, _| std::future::ready(())).await,
        Err(GraphError::ChatNotReady { .. })
    ));
    let encoded = serde_json::to_vec(&chat.snapshot()?).map_err(codec)?;
    let mut restored = builder().restore::<()>(
        serde_json::from_slice(&encoded).map_err(codec)?,
        context(Mode::Complete, stats.clone())?,
    )?;
    finish_restored_chat(&mut restored).await?;
    assert_eq!(stats.starts.load(Ordering::SeqCst), 2);
    Ok(())
}

/// Explicitly delivers the saved request before stepping a restored pending chat.
async fn finish_restored_chat(chat: &mut pravah::Chat<String, String>) -> Result<(), GraphError> {
    let pending = chat
        .pending_agent()
        .cloned()
        .ok_or_else(|| GraphError::Invalid("restored chat lost its pending generation".into()))?;
    let response = chat
        .executor()
        .execute_stream(&pending, |_, _| std::future::ready(()))
        .await;
    chat.resume_agent(response)?;
    for _ in 0..100 {
        match chat.next()? {
            pravah::ChatStep::Continue => {}
            pravah::ChatStep::Agent(request) => {
                let response = chat
                    .executor()
                    .execute_stream(&request, |_, _| std::future::ready(()))
                    .await;
                chat.resume_agent(response)?;
            }
            pravah::ChatStep::Done(turn) => {
                assert_eq!(turn.output, "authoritative answer");
                return Ok(());
            }
            _ => return Err(GraphError::Invalid("unexpected suspension".into())),
        }
    }
    Err(GraphError::Invalid("restored chat did not finish".into()))
}

/// Accepted streamed output survives JSON and CBOR restore without another model generation.
#[tokio::test]
async fn accepted_completion_round_trips_without_redispatch() -> Result<(), GraphError> {
    for cbor in [false, true] {
        let stats = Arc::new(Stats::default());
        let flow = compile(workflow)?;
        let executor = flow
            .prepared()
            .executor(context(Mode::Complete, stats.clone())?);
        let mut runtime = flow.start("question".into(), Uuid::nil())?;
        let request = generation(&mut runtime, &executor).await?;
        runtime.resume_agent(
            executor
                .execute_stream(&request, |_, _| std::future::ready(()))
                .await,
        )?;
        assert!(runtime.pending_agent().is_none());
        let mut bytes = Vec::new();
        let snapshot = if cbor {
            ciborium::into_writer(&runtime.snapshot()?, &mut bytes).map_err(codec)?;
            ciborium::from_reader(bytes.as_slice()).map_err(codec)?
        } else {
            bytes = serde_json::to_vec(&runtime.snapshot()?).map_err(codec)?;
            serde_json::from_slice(&bytes).map_err(codec)?
        };
        let mut restored = flow.restore(snapshot)?;
        complete(&mut restored, &flow)?;
        assert_eq!(stats.starts.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

/// Selection validation rejects a streaming submission before accepting any input or calling a client.
#[tokio::test]
async fn invalid_selection_leaves_chat_unchanged() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let mut chat = builder().build(context(Mode::Complete, stats.clone())?)?;
    let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    let input = pravah::ChatRequest::from("question".to_owned()).tools(["undeclared"]);
    assert!(matches!(
        chat.send_stream(input, |_, _| std::future::ready(())).await,
        Err(GraphError::ChatRequestValidation { .. })
    ));
    assert_eq!(
        before,
        serde_json::to_value(chat.snapshot()?).map_err(codec)?
    );
    assert_eq!(stats.starts.load(Ordering::SeqCst), 0);
    assert_eq!(
        chat.send_stream("corrected", |_, _| std::future::ready(()))
            .await?
            .output,
        "authoritative answer"
    );
    Ok(())
}
