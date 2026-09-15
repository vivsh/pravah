use super::*;
use crate::graph::value::to_value;
use serde::{Deserialize, Serialize};

/// Immutable, schema-described configure input owned by the authored graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ConfigurationData {
    pub(crate) value: Value,
    pub(crate) schema: serde_json::Value,
}

pub(super) fn validate_absent(data: Option<&ConfigurationData>) -> Result<(), GraphError> {
    if data.is_some() {
        return Err(GraphError::GraphValidation(
            "configure function does not accept data".into(),
        ));
    }
    Ok(())
}

/// Checks the registered data type and full schema before a graph can execute.
fn validate_data<D: DeserializeOwned + JsonSchema>(
    data: Option<&ConfigurationData>,
) -> Result<(), GraphError> {
    let validate = || -> Result<(), String> {
        let data = data.ok_or("configure data is missing")?;
        let expected = serde_json::to_value(schemars::schema_for!(D)).map_err(|e| e.to_string())?;
        if data.schema != expected {
            return Err("configure data schema differs from handler".into());
        }
        let validator = jsonschema::validator_for(&data.schema).map_err(|e| e.to_string())?;
        let json = serde_json::to_value(&data.value).map_err(|e| e.to_string())?;
        validator.validate(&json).map_err(|e| e.to_string())?;
        from_value::<D>(data.value.clone()).map_err(|e| e.to_string())?;
        Ok(())
    };
    validate().map_err(GraphError::GraphValidation)
}

impl<T> Agent<T> {
    /// Registers definition data and its typed activation function without capturing the data.
    pub(crate) fn configure_with<O, D, Fut, E>(
        mut self,
        data: D,
        configure: fn(T, D, Context) -> Fut,
    ) -> Agent<O>
    where
        T: 'static + DeserializeOwned + JsonSchema + Send + Sync,
        O: 'static + DeserializeOwned + JsonSchema + Send + Sync,
        D: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        Fut: Future<Output = Result<AgentConfig, E>> + Send + 'static,
        E: Error + Send + Sync + 'static,
    {
        if self.definition.configure.is_some() {
            self.definition
                .errors
                .push("agent configure may only be declared once".into());
        } else {
            match encode_data(data) {
                Ok(data) => self.definition.configuration = Some(data),
                Err(error) => self.definition.errors.push(error.to_string()),
            }
            self.definition.configure = Some(data_configurator::<T, O, D, Fut, E>(configure));
        }
        Agent {
            definition: self.definition,
            _marker: PhantomData,
        }
    }
}

/// Keeps the callable independent of settings ownership; activation receives graph-owned data.
fn data_configurator<T, O, D, Fut, E>(configure: fn(T, D, Context) -> Fut) -> AgentConfigurator
where
    T: 'static + DeserializeOwned + Send,
    O: JsonSchema,
    D: 'static + DeserializeOwned + JsonSchema + Send,
    Fut: Future<Output = Result<AgentConfig, E>> + Send + 'static,
    E: Error + Send + Sync + 'static,
{
    AgentConfigurator {
        validate_data: validate_data::<D>,
        call: Arc::new(move |input, data, ctx| {
            async move {
                let input = from_value(input)
                    .map_err(|e| GraphError::AgentConfigValidation(e.to_string()))?;
                let data = data.ok_or_else(|| {
                    GraphError::AgentConfigValidation("configure data is missing".into())
                })?;
                let data = from_value(data)
                    .map_err(|e| GraphError::AgentConfigValidation(e.to_string()))?;
                configure(input, data, ctx)
                    .await
                    .map_err(|e| GraphError::AgentConfiguration {
                        agent: O::schema_name().into_owned(),
                        reason: e.to_string(),
                    })
            }
            .boxed()
        }),
    }
}

fn encode_data<D: Serialize + JsonSchema>(data: D) -> Result<ConfigurationData, String> {
    Ok(ConfigurationData {
        value: to_value(data).map_err(|e| e.to_string())?,
        schema: serde_json::to_value(schemars::schema_for!(D)).map_err(|e| e.to_string())?,
    })
}
