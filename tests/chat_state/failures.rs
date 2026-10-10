use super::*;

#[derive(Deserialize, JsonSchema)]
struct FallibleState(u32);

impl Serialize for FallibleState {
    fn serialize<T: serde::Serializer>(&self, serializer: T) -> Result<T::Ok, T::Error> {
        match self.0 {
            99 => Err(serde::ser::Error::custom("state preparation failed")),
            98 => serializer.serialize_str("wrong shape"),
            value => serializer.serialize_u32(value),
        }
    }
}

/// Failed encoding and shape validation preserve the entire snapshot, including epochs.
#[tokio::test]
async fn rejected_set_is_atomic() -> Result<(), TestError> {
    let mut chat = Chat::with_state(assistant, FallibleState(1), Context::default())?;
    let before = serde_json::to_value(chat.snapshot()?)?;
    assert!(matches!(
        chat.set(FallibleState(99)),
        Err(GraphError::ValueConversion { .. })
    ));
    assert_eq!(before, serde_json::to_value(chat.snapshot()?)?);
    assert!(matches!(
        chat.set(FallibleState(98)),
        Err(GraphError::Schema { .. })
    ));
    assert_eq!(before, serde_json::to_value(chat.snapshot()?)?);
    assert_eq!(chat.get()?.0, 1);
    Ok(())
}

/// Application writes are independent of exhausted VM epochs and never alter frame metadata.
#[tokio::test]
async fn application_state_does_not_consume_frame_epochs() -> Result<(), TestError> {
    let chat = Chat::with_state(assistant, initial_state(), Context::default())?;
    let mut snapshot = serde_json::to_value(chat.snapshot()?)?;
    *snapshot
        .pointer_mut("/state/frames/0/write_epoch")
        .ok_or(TestError::Missing("write epoch"))? = json!(u64::MAX);
    let mut chat = Chat::<String, String, Session>::from_snapshot(
        assistant,
        serde_json::from_value(snapshot)?,
        Context::default(),
    )?;
    let before = serde_json::to_value(chat.snapshot()?)?;
    chat.set(Session {
        visits: 1,
        ..initial_state()
    })?;
    let after = serde_json::to_value(chat.snapshot()?)?;
    assert_eq!(
        before.pointer("/state/frames"),
        after.pointer("/state/frames")
    );
    assert_eq!(chat.get()?.visits, 1);
    Ok(())
}
