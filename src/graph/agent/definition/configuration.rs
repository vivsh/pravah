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

/// Immutable registered type contract; it never owns an invocation's settings value.
pub(super) struct ConfigurationValidator {
    schema: serde_json::Value,
    validator: jsonschema::Validator,
    decode: fn(&Value) -> Result<(), GraphError>,
}

impl ConfigurationValidator {
    fn new<D: DeserializeOwned>(schema: serde_json::Value) -> Result<Self, String> {
        let validator = jsonschema::validator_for(&schema).map_err(|error| error.to_string())?;
        Ok(Self {
            schema,
            validator,
            decode: validate_decode::<D>,
        })
    }

    /// Validates incoming definition data against the fixed registered contract without recompiling.
    pub(super) fn validate(&self, data: Option<&ConfigurationData>) -> Result<(), GraphError> {
        let data =
            data.ok_or_else(|| GraphError::GraphValidation("configure data is missing".into()))?;
        if data.schema != self.schema {
            return Err(GraphError::GraphValidation(
                "configure data schema differs from handler".into(),
            ));
        }
        let json = serde_json::to_value(&data.value)
            .map_err(|error| GraphError::GraphValidation(error.to_string()))?;
        self.validator
            .validate(&json)
            .map_err(|error| GraphError::GraphValidation(error.to_string()))?;
        (self.decode)(&data.value)
    }
}

fn validate_decode<D: DeserializeOwned>(value: &Value) -> Result<(), GraphError> {
    from_value::<D>(value.clone())
        .map(|_| ())
        .map_err(|error| GraphError::GraphValidation(error.to_string()))
}

impl<T> Agent<T> {
    /// Registers graph-owned settings and invocation instructions owned by the callable.
    /// Only instructions may change on restore; activation checkpoints the resolved value.
    pub(crate) fn configure_with<O, D, Fut, E>(
        mut self,
        data: D,
        instructions: String,
        configure: fn(T, D, String, Context) -> Fut,
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
            match prepare_data(data) {
                Ok((data, contract)) => {
                    self.definition.configuration = Some(data);
                    self.definition.configure = Some(data_configurator::<T, O, D, Fut, E>(
                        instructions,
                        configure,
                        contract,
                    ));
                }
                Err(error) => self.definition.errors.push(error.to_string()),
            }
        }
        Agent {
            definition: self.definition,
            _marker: PhantomData,
        }
    }
}

/// Owns only replaceable instructions; every other setting comes from the graph payload.
fn data_configurator<T, O, D, Fut, E>(
    instructions: String,
    configure: fn(T, D, String, Context) -> Fut,
    contract: Arc<ConfigurationValidator>,
) -> AgentConfigurator
where
    T: 'static + DeserializeOwned + Send,
    O: JsonSchema,
    D: 'static + DeserializeOwned + JsonSchema + Send,
    Fut: Future<Output = Result<AgentConfig, E>> + Send + 'static,
    E: Error + Send + Sync + 'static,
{
    AgentConfigurator {
        data: Some(contract),
        call: Arc::new(move |input, data, ctx| {
            let instructions = instructions.clone();
            async move {
                let input = from_value(input)
                    .map_err(|e| GraphError::AgentConfigValidation(e.to_string()))?;
                let data = data.ok_or_else(|| {
                    GraphError::AgentConfigValidation("configure data is missing".into())
                })?;
                let data = from_value(data)
                    .map_err(|e| GraphError::AgentConfigValidation(e.to_string()))?;
                configure(input, data, instructions, ctx)
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

/// Serializes graph-owned settings and prepares the independent registered type contract once.
fn prepare_data<D: Serialize + DeserializeOwned + JsonSchema>(
    data: D,
) -> Result<(ConfigurationData, Arc<ConfigurationValidator>), String> {
    let schema = serde_json::to_value(schemars::schema_for!(D)).map_err(|e| e.to_string())?;
    let contract = Arc::new(ConfigurationValidator::new::<D>(schema.clone())?);
    Ok((
        ConfigurationData {
            value: to_value(data).map_err(|e| e.to_string())?,
            schema,
        },
        contract,
    ))
}
