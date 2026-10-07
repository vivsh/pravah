//! Standalone async tool functions lowered to externally executed continuations.

use super::*;
use crate::graph::agent_request::AgentOperation;
use crate::graph::{AgentRequest, DynAgentHandler, HandlerKey};

type ToolFunction =
    dyn Fn(Value, Context) -> BoxFuture<'static, Result<Value, GraphError>> + Send + Sync;

pub(crate) struct FunctionTool {
    call: Box<ToolFunction>,
    json_contract: Option<Arc<JsonToolContract>>,
}

impl FunctionTool {
    /// Registers typed execution with the existing tagged success/error encoding.
    pub(super) fn new<I, O, Fut>(func: fn(I, Context) -> Fut) -> Self
    where
        I: DeserializeOwned + Send + 'static,
        O: Serialize + 'static,
        Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
    {
        Self {
            json_contract: None,
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
    /// Registers exact JSON execution without introducing a second worker path.
    pub(super) fn json<H, Fut>(contract: Arc<JsonToolContract>, handler: H) -> Self
    where
        H: Fn(JsonValue, Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<JsonValue, ToolError>> + Send + 'static,
    {
        let handler = Arc::new(handler);
        let validation = Arc::clone(&contract);
        Self {
            json_contract: Some(contract),
            call: Box::new(move |input, context| {
                let handler = Arc::clone(&handler);
                let validation = Arc::clone(&validation);
                async move {
                    let input = from_value(input).map_err(|error| GraphError::ValueConversion {
                        target: "JSON tool input".into(),
                        reason: error.to_string(),
                    })?;
                    let envelope = match handler(input, context).await {
                        Ok(value) => {
                            validation
                                .validate(&value, true)
                                .map_err(GraphError::Invalid)?;
                            EdgeToolResult::Success { value }
                        }
                        Err(error) if !error.is_fatal() => EdgeToolResult::Error {
                            value: error.to_json(validation.name()),
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
    fn validate_payload(&self, payload: &Value) -> Result<(), GraphError> {
        match (&self.json_contract, payload.get("json_contract")) {
            (None, None) => Ok(()),
            (Some(contract), Some(authored)) if &contract.definition == authored => Ok(()),
            _ => Err(GraphError::GraphValidation(
                "JSON tool handler contract mismatch".into(),
            )),
        }
    }

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
        let input = single_input(inputs, "tool")?;
        if let Some(contract) = &self.json_contract {
            contract.validate_input(&input)?;
        }
        let operation = AgentOperation::Tool {
            handler: HandlerKey::new(handler),
            input,
        };
        Ok(ContinuationTransition {
            checkpoint: Some(Value::from(1_u32)),
            agent: Some(AgentRequest::new(Uuid::nil(), operation)),
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
        let ContinuationEvent::Agent { response, .. } = event else {
            return Err(GraphError::Invalid(
                "tool has no accepted external outcome".into(),
            ));
        };
        let output = effects::success(response)?;
        if let Some(contract) = &self.json_contract {
            contract.validate_envelope(&output)?;
        }
        Ok(ContinuationTransition {
            outputs: vec![output],
            ..Default::default()
        })
    }
}

impl DynAgentHandler for FunctionTool {
    fn execute<'a>(
        &'a self,
        request: &'a AgentRequest,
        context: Context,
    ) -> BoxFuture<'a, Result<Value, GraphError>> {
        async move {
            let AgentOperation::Tool { input, .. } = request.operation.as_ref() else {
                return Err(GraphError::AgentRequestValidation(
                    "wrong tool operation".into(),
                ));
            };
            if let Some(contract) = &self.json_contract {
                contract.validate_input(input)?;
            }
            (self.call)(input.clone(), context).await
        }
        .boxed()
    }
}
