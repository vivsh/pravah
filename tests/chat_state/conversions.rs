use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static ENCODED: AtomicUsize = AtomicUsize::new(0);
static DECODED: AtomicUsize = AtomicUsize::new(0);

#[derive(JsonSchema)]
struct Measured(Vec<String>);

impl Serialize for Measured {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ENCODED.fetch_add(1, Ordering::SeqCst);
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Measured {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        DECODED.fetch_add(1, Ordering::SeqCst);
        Vec::<String>::deserialize(deserializer).map(Self)
    }
}

/// VM execution and snapshot encoding never re-encode or decode the typed application state.
#[tokio::test]
async fn state_conversion_occurs_only_at_explicit_boundaries() -> Result<(), TestError> {
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let state = Measured(vec!["large".repeat(1000); 128]);
    let mut chat = Chat::with_state(assistant, state, context(script)?)?;
    assert_eq!(ENCODED.load(Ordering::SeqCst), 1);
    assert_eq!(DECODED.load(Ordering::SeqCst), 0);
    chat.send("one").await?;
    chat.send("two").await?;
    let _copies = roundtrips(&chat.snapshot()?)?;
    assert_eq!(ENCODED.load(Ordering::SeqCst), 1);
    assert_eq!(DECODED.load(Ordering::SeqCst), 0);
    let value = chat.get()?;
    assert_eq!(DECODED.load(Ordering::SeqCst), 1);
    chat.set(value)?;
    assert_eq!(ENCODED.load(Ordering::SeqCst), 2);
    let _restored = Chat::<String, String, Measured>::from_snapshot(
        assistant,
        chat.snapshot()?,
        Context::default(),
    )?;
    assert_eq!(DECODED.load(Ordering::SeqCst), 2);
    Ok(())
}

/// Snapshot capture allocates the same metadata regardless of the retained composite payload size.
#[tokio::test]
async fn snapshot_capture_shares_large_payloads() -> Result<(), TestError> {
    let small = Chat::with_state(assistant, vec!["small".to_owned()], Context::default())?;
    let large = Chat::with_state(
        assistant,
        vec!["large".repeat(1000); 128],
        Context::default(),
    )?;
    let mut small_snapshot = None;
    let small_allocations = allocation_counter::measure(|| {
        small_snapshot = Some(small.snapshot());
    });
    let mut large_snapshot = None;
    let large_allocations = allocation_counter::measure(|| {
        large_snapshot = Some(large.snapshot());
    });
    small_snapshot.ok_or(TestError::Missing("small snapshot"))??;
    large_snapshot.ok_or(TestError::Missing("large snapshot"))??;
    assert_eq!(small_allocations.count_total, large_allocations.count_total);
    assert_eq!(small_allocations.bytes_total, large_allocations.bytes_total);
    Ok(())
}
