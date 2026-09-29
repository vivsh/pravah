use super::*;
use serde::{Deserializer, Serializer};

/// Wrapping a non-Clone input performs no serialization or allocation.
#[test]
fn construction_only_moves_input() -> Result<(), TestError> {
    let input = Question {
        topic: "owned".into(),
        depth: 1,
    };
    let pointer = input.topic.as_ptr();
    let mut request = None;
    let allocations = allocation_counter::measure(|| request = Some(ChatRequest::from(input)));
    let request = request.ok_or_else(|| GraphError::Invalid("request missing".into()))?;
    assert_eq!(allocations.count_total, 0);
    assert_eq!(request.input().topic.as_ptr(), pointer);
    assert!(request.memory_text().is_none());
    assert!(request.selected_tools().is_none());
    assert!(request.selected_resources().is_none());
    Ok(())
}

/// String builders render JSON strings, not guessed plaintext or Message objects.
#[tokio::test]
async fn string_rendering_is_json() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("ok"));
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .build(Context::default().with_providers(pravah::testing::providers(factory)?))?;
    chat.send("hello\nworld").await?;
    assert_eq!(
        chat.snapshot()?.history().entries()[0].message.content,
        r#""hello\nworld""#
    );
    Ok(())
}

#[derive(JsonSchema)]
#[schemars(with = "String")]
struct FailingInput {
    fail: bool,
}

impl Serialize for FailingInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.fail {
            Err(serde::ser::Error::custom(
                "application serialization failure",
            ))
        } else {
            serializer.serialize_str("accepted")
        }
    }
}

impl<'de> Deserialize<'de> for FailingInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?;
        Ok(Self { fail: true })
    }
}

/// Entry serialization errors are atomic; activation rendering errors never call the model.
#[tokio::test]
async fn serialization_failure_boundaries() -> Result<(), TestError> {
    let factory = ScriptedFactory::new();
    let mut chat = Chat::builder::<FailingInput, String>()
        .model("test:///test")
        .build(Context::default().with_providers(pravah::testing::providers(factory.clone())?))?;
    let before = serde_json::to_value(chat.snapshot()?)?;
    assert!(chat.send(FailingInput { fail: true }).await.is_err());
    assert_eq!(serde_json::to_value(chat.snapshot()?)?, before);
    assert!(chat.send(FailingInput { fail: false }).await.is_err());
    assert!(factory.calls().is_empty());
    assert!(chat.snapshot()?.history().entries().is_empty());
    assert!(matches!(
        chat.send(FailingInput { fail: false }).await,
        Err(GraphError::ChatNotReady { .. })
    ));
    Ok(())
}
