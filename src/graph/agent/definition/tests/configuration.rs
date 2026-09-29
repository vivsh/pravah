use super::super::*;
use crate::clients::Message;
use crate::graph::to_value;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Settings {
    model: String,
}

async fn configure(
    input: String,
    settings: Settings,
    instructions: String,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        settings.model,
        instructions,
        Message::user(input),
    ))
}

/// Preparation rejects missing, schema-mismatched and undecodable configure data.
#[test]
fn registered_data_contract_is_checked() -> Result<(), GraphError> {
    let agent: Agent<String> = Agent::root().configure_with(
        Settings {
            model: "openai:///test".into(),
        },
        "instructions".into(),
        configure,
    );
    let definition = agent.definition;
    let handler = definition
        .configure
        .ok_or_else(|| GraphError::Invalid("missing test handler".into()))?;
    let mut data = definition
        .configuration
        .ok_or_else(|| GraphError::Invalid("missing test data".into()))?;
    handler.validate_data(Some(&data))?;
    assert!(handler.validate_data(None).is_err());
    let original = data.schema.clone();
    data.schema = serde_json::json!({"type":"string"});
    assert!(handler.validate_data(Some(&data)).is_err());
    data.schema = original;
    data.value = to_value(serde_json::json!({"model": false}))
        .map_err(|e| GraphError::Invalid(e.to_string()))?;
    assert!(handler.validate_data(Some(&data)).is_err());
    Ok(())
}

/// Callback-free settings never become an allowed extra input for ordinary configure handlers.
#[test]
fn ordinary_configuration_rejects_definition_data() -> Result<(), GraphError> {
    let agent: Agent<String> = Agent::root().configure_with(
        Settings {
            model: "openai:///test".into(),
        },
        "instructions".into(),
        configure,
    );
    assert!(configuration::validate_absent(agent.definition.configuration.as_ref()).is_err());
    configuration::validate_absent(None)
}
