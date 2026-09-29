//! Standalone async tool functions lowered to externally executed continuations.

use super::*;
use crate::graph::{DynFetchHandler, Fetch, FetchBody, FetchResponse};

type ToolFunction =
    dyn Fn(Value, Context) -> BoxFuture<'static, Result<Value, GraphError>> + Send + Sync;

pub(crate) struct FunctionTool {
    call: Box<ToolFunction>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct ToolHook {
    pub version: u32,
    pub handler: String,
    pub input: Value,
}

impl FunctionTool {
    pub(super) fn new<I, O, Fut>(func: fn(I, Context) -> Fut) -> Self
    where
        I: DeserializeOwned + Send + 'static,
        O: Serialize + 'static,
        Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
    {
        Self {
            call: Box::new(move |input, context| {
                async move {
                    let input = from_value(input).map_err(|error| GraphError::ValueConversion {
                        target: "tool input".into(),
                        reason: error.to_string(),
                    })?;
                    let envelope = match func(input, context).await {
                        Ok(output) => EdgeToolResult::Success {
                            value: serde_json::to_value(output).map_err(|error| {
                                GraphError::ValueConversion {
                                    target: "tool output".into(),
                                    reason: error.to_string(),
                                }
                            })?,
                        },
                        Err(error) if !error.is_fatal() => EdgeToolResult::Error {
                            value: error.to_json(""),
                        },
                        Err(error) => return Err(GraphError::Invalid(error.to_string())),
                    };
                    effects::encode(envelope)
                }
                .boxed()
            }),
        }
    }
}

impl ContinuationHandler for FunctionTool {
    fn start<'a>(
        &'a self,
        payload: &'a Value,
        _state: Option<Value>,
        inputs: Vec<Value>,
        _ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let handler = payload
            .get("tool_handler_key")
            .and_then(Value::as_str)
            .ok_or_else(|| GraphError::GraphValidation("missing tool handler identity".into()))?;
        let hook = ToolHook {
            version: 1,
            handler: handler.into(),
            input: single_input(inputs, "tool")?,
        };
        Ok(ContinuationTransition {
            checkpoint: Some(Value::from(1_u32)),
            fetch: Some(hook.into_request()?),
            ..Default::default()
        })
    }
    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        checkpoint: Value,
        event: ContinuationEvent,
        _ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        if checkpoint.as_u64() != Some(1) {
            return Err(GraphError::SnapshotValidation(
                "unsupported tool checkpoint".into(),
            ));
        }
        let ContinuationEvent::Fetch { outcome, .. } = event else {
            return Err(GraphError::Invalid(
                "tool has no accepted external outcome".into(),
            ));
        };
        let response = effects::success(outcome)?;
        let Some(FetchBody::Value(output)) = response.body_ref() else {
            return Err(GraphError::FetchValidation(
                "expected structured tool output".into(),
            ));
        };
        Ok(ContinuationTransition {
            outputs: vec![output.clone()],
            ..Default::default()
        })
    }
}

impl DynFetchHandler for FunctionTool {
    fn execute<'a>(
        &'a self,
        fetch: &'a Fetch,
        context: Context,
    ) -> BoxFuture<'a, Result<FetchResponse, GraphError>> {
        async move {
            let hook = ToolHook::from_request(fetch.request())?;
            effects::check_protocol(hook.version)?;
            let output = (self.call)(hook.input, context).await?;
            Ok(FetchResponse::new(200).body(FetchBody::Value(output)))
        }
        .boxed()
    }
}
