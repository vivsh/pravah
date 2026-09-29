use super::*;
use pravah::{CompactionRequest, CompactionResult, Compactor};

struct InvalidPolicy(u8);

impl Compactor for InvalidPolicy {
    type Error = GraphError;

    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        match self.0 {
            0 => Err(GraphError::Invalid("intentional policy failure".into())),
            1 => Ok(CompactionResult {
                evict_indices: vec![usize::MAX],
                summary: None,
            }),
            _ => Ok(CompactionResult {
                evict_indices: vec![request.committed().len()],
                summary: Some("invalid".into()),
            }),
        }
    }
}

/// Policy errors, invalid indices and pending-input eviction leave preparation retryable and exact.
#[tokio::test]
async fn failed_preparation_preserves_history_and_pending_request() -> Result<(), GraphError> {
    for mode in 0..3 {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut chat = builder()
            .compactor(InvalidPolicy(mode))
            .build(context(&calls))?;
        chat.submit_with_key("protected", "key")?;
        let fetch = preparation::pending(&mut chat).await?;
        let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
        for _ in 0..2 {
            assert!(chat.executor().execute(&fetch).await.is_err());
            assert_eq!(chat.pending_fetch().map(Fetch::id), Some(fetch.id()));
            assert_eq!(
                serde_json::to_value(chat.snapshot()?).map_err(codec)?,
                before
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }
    }
    Ok(())
}
