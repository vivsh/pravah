use super::*;
use crate::graph::chat::ChatRequest;

impl<T> Agent<T> {
    /// Unwraps Chat's durable submission without changing application callback types.
    pub(crate) fn for_chat(mut self) -> Self {
        if let Some(configure) = self.definition.configure.as_mut() {
            let original = Arc::clone(&configure.call);
            configure.call = Arc::new(move |value, data, execution_id, ctx| {
                let original = Arc::clone(&original);
                async move {
                    let request = ChatRequest::decode(&value)
                        .map_err(|e| GraphError::AgentConfigValidation(e.to_string()))?;
                    let mut config = original(request.input, data, execution_id, ctx).await?;
                    if config.key.is_none() {
                        config.key = Some(execution_id.to_string());
                    }
                    if let Some(memory) = request.memory {
                        config.memory = Some(memory);
                    }
                    if let Some(tools) = request.tools {
                        config = config.tool_filter(crate::graph::ToolFilter::only(tools));
                    }
                    if let Some(resources) = request.resources {
                        config.resources = resources;
                    }
                    if let Some(key) = request.key {
                        config.message.key = Some(key);
                    }
                    Ok(config)
                }
                .boxed()
            });
        }
        self.unwrap_chat_controller();
        self
    }

    /// Keeps controller observations typed over the invocation input, not the Chat envelope.
    fn unwrap_chat_controller(&mut self) {
        if let Some(controller) = self.definition.controller.as_mut() {
            let original = Arc::clone(&controller.call);
            controller.call = Arc::new(move |mut data, ctx| {
                let input = ChatRequest::decode(&data.input);
                match input {
                    Ok(request) => {
                        data.input = request.input;
                        original(data, ctx)
                    }
                    Err(error) => async move {
                        Err(GraphError::AgentControl {
                            agent: data.agent_id,
                            reason: error.to_string(),
                        })
                    }
                    .boxed(),
                }
            });
        }
    }
}
