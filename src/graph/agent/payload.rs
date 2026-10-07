use super::definition::ConfigurationData;
use super::{
    AgentHandler, AgentToolPayload, GraphError, JSON_PAYLOAD_VERSION, PAYLOAD_VERSION, Value,
    from_value,
};

/// Operation-local execution metadata, never retained in a handler or checkpoint.
/// Names and configure data borrow the immutable graph; provider metadata is decoded locally.
pub(super) struct AgentPayloadView<'a> {
    pub(super) raw: &'a Value,
    pub(super) agent_id: &'a str,
    pub(super) output_type_name: &'a str,
    pub(super) configuration: Option<&'a Value>,
    output_schema: &'a Value,
    pub(super) tools: Vec<AgentToolPayload>,
}

impl<'a> AgentPayloadView<'a> {
    /// Reads execution fields from a prepared payload without rebuilding its input/data schemas.
    /// Full definition decoding and configure-data validation remain preparation responsibilities.
    pub(super) fn read(payload: &'a Value) -> Result<Self, GraphError> {
        let agent_id = validate_identity(payload)?;
        let configuration = match payload.get("configuration") {
            None => None,
            Some(value) if value.is_null() => None,
            Some(value) => Some(field(value, "value")?),
        };
        Ok(Self {
            raw: payload,
            agent_id,
            output_type_name: text_field(payload, "output_type_name")?,
            configuration,
            output_schema: field(payload, "output_schema")?,
            tools: decode_field(payload, "tools")?,
        })
    }

    /// Borrows the authored schema for sharing with the durable generation request.
    pub(super) fn output_schema(&self) -> &Value {
        self.output_schema
    }
}

impl AgentHandler {
    /// Validates a definition boundary once while retaining the same borrowed execution view.
    /// Definition schemas are JSON values; only configure data has a registered schema contract.
    pub(super) fn validated_payload<'a>(
        &self,
        payload: &'a Value,
    ) -> Result<AgentPayloadView<'a>, GraphError> {
        let view = AgentPayloadView::read(payload).map_err(|error| {
            GraphError::GraphValidation(format!("invalid registered agent payload: {error}"))
        })?;
        field(payload, "input_schema")?;
        super::json_tool::validate_runtime_contracts(&view.tools, &self.tools, payload)?;
        let data = configuration_data(payload)?;
        self.configure.validate_data(data.as_ref())?;
        let control = payload
            .get("control_handler_key")
            .filter(|value| !value.is_null());
        match (control, self.controller.is_some()) {
            (Some(key), false) => {
                return Err(GraphError::MissingHandler(
                    key.as_str()
                        .ok_or_else(|| {
                            GraphError::GraphValidation("invalid controller key".into())
                        })?
                        .into(),
                ));
            }
            (None, true) => {
                return Err(GraphError::GraphValidation(
                    "registered agent controller is absent from its payload".into(),
                ));
            }
            _ => {}
        }
        Ok(view)
    }
}

/// Reads the settings contract without recursively reconstructing the already shared value.
fn configuration_data(payload: &Value) -> Result<Option<ConfigurationData>, GraphError> {
    let Some(data) = payload
        .get("configuration")
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    Ok(Some(ConfigurationData {
        value: field(data, "value")?.clone(),
        schema: decode_field(data, "schema")?,
    }))
}

/// Keeps version and handler-identity rejection explicit even for direct handler invocations.
pub(super) fn validate_identity(payload: &Value) -> Result<&str, GraphError> {
    let version = decode_field::<u32>(payload, "version")?;
    if version != PAYLOAD_VERSION && version != JSON_PAYLOAD_VERSION {
        return Err(GraphError::UnsupportedVersion {
            format: "agent payload",
            got: version,
            expected: PAYLOAD_VERSION,
        });
    }
    let agent_id = text_field(payload, "agent_id")?;
    if agent_id.is_empty() || text_field(payload, "configure_handler_key")? != agent_id {
        return Err(GraphError::AgentConfigValidation(
            "agent configure handler identity is missing or inconsistent".into(),
        ));
    }
    if let Some(control) = payload
        .get("control_handler_key")
        .filter(|value| !value.is_null())
    {
        let valid = control
            .as_str()
            .and_then(|key| key.strip_suffix("::control"))
            == Some(agent_id);
        if !valid {
            return Err(GraphError::AgentConfigValidation(
                "agent control handler identity is inconsistent".into(),
            ));
        }
    }
    Ok(agent_id)
}

#[cfg(test)]
mod tests;

fn field<'a>(payload: &'a Value, name: &str) -> Result<&'a Value, GraphError> {
    payload
        .get(name)
        .ok_or_else(|| GraphError::GraphValidation(format!("missing agent payload field '{name}'")))
}

fn decode_field<T: serde::de::DeserializeOwned>(
    payload: &Value,
    name: &str,
) -> Result<T, GraphError> {
    from_value(field(payload, name)?.clone()).map_err(|error| {
        GraphError::GraphValidation(format!("invalid agent payload field '{name}': {error}"))
    })
}

fn text_field<'a>(payload: &'a Value, name: &str) -> Result<&'a str, GraphError> {
    field(payload, name)?.as_str().ok_or_else(|| {
        GraphError::GraphValidation(format!("agent payload field '{name}' must be a string"))
    })
}
