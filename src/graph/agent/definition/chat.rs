use super::*;
use crate::graph::chat::ChatSubmission;

impl<T> Agent<T> {
    /// Unwraps Chat's durable submission without changing application callback types.
    pub(crate) fn for_chat(mut self) -> Self {
        if let Some(configure) = self.definition.configure.as_mut() {
            let original = Arc::clone(&configure.call);
            configure.call = Arc::new(move |value, data, ctx| {
                let original = Arc::clone(&original);
                async move {
                    let request = ChatSubmission::decode(&value)
                        .map_err(|e| GraphError::AgentConfigValidation(e.to_string()))?;
                    let mut config = original(request.input, data, ctx).await?;
                    if let Some(key) = request.key {
                        config.message.key = Some(key);
                    }
                    Ok(config)
                }
                .boxed()
            });
        }
        if let Some(controller) = self.definition.controller.as_mut() {
            let original = Arc::clone(&controller.call);
            controller.call = Arc::new(move |mut data, ctx| {
                let input = ChatSubmission::decode(&data.input);
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
        self
    }
}
