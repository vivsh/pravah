//! Application-only state boundaries, independent of VM frames and external effects.

use super::*;
use crate::graph::model::TypeSpec;
use crate::graph::typed::type_spec;
use crate::graph::value::from_value;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

impl Runtime {
    /// Decodes the execution's application state, including after graph completion.
    ///
    /// Fails if no state was initialized, the requested schema differs, or decoding fails.
    /// Conversion may allocate; the runtime retains no typed copy.
    pub fn get_state<S: DeserializeOwned + JsonSchema>(&self) -> Result<S, GraphError> {
        let (spec, value) = self.application_state::<S>()?;
        from_value(value.clone()).map_err(|error| GraphError::ValueConversion {
            target: format!("application state '{}'", spec.name),
            reason: error.to_string(),
        })
    }

    /// Replaces application-only state between synchronous steps without changing VM epochs.
    ///
    /// Outstanding and accepted-but-unprocessed agent operations reject replacement.
    /// Missing state, type, conversion and schema failures leave execution unchanged.
    /// This never changes issued requests, graph variables, history or model context.
    pub fn set_state<S: Serialize + JsonSchema>(&mut self, state: S) -> Result<(), GraphError> {
        self.require_application_state_write()?;
        let (spec, _) = self.application_state::<S>()?;
        let value = encode_state(state)?;
        validate_state_value(spec, &value)?;
        let (_, current) = self
            .state
            .application_state
            .as_mut()
            .ok_or_else(missing_state)?;
        *current = value;
        Ok(())
    }

    /// Installs the sole application value before an execution is exposed to its caller.
    pub(crate) fn initialize_application_state<S: Serialize + JsonSchema>(
        &mut self,
        state: S,
    ) -> Result<(), GraphError> {
        if self.state.application_state.is_some() {
            return Err(GraphError::ApplicationState(
                "state is already initialized".into(),
            ));
        }
        let spec = type_spec::<S>();
        let value = encode_state(state)?;
        validate_state_value(&spec, &value)?;
        self.state.application_state = Some((spec, value));
        Ok(())
    }

    /// Checks fixed type metadata without retaining a second schema or decoded value.
    fn application_state<S: JsonSchema>(&self) -> Result<(&TypeSpec, &Value), GraphError> {
        let (spec, value) = self
            .state
            .application_state
            .as_ref()
            .ok_or_else(missing_state)?;
        if *spec != type_spec::<S>() {
            return Err(GraphError::ApplicationState(format!(
                "expected '{}', requested '{}'",
                spec.name,
                S::schema_name()
            )));
        }
        Ok((spec, value))
    }

    /// Derives mutation eligibility from existing pending requests and accepted inputs.
    fn require_application_state_write(&self) -> Result<(), GraphError> {
        let accepted = self.state.frames.iter().any(|frame| {
            frame
                .continuation_inboxes
                .iter()
                .flatten()
                .any(|input| matches!(input, ContinuationInput::Agent { .. }))
        });
        if matches!(self.state.waiting, Some(Waiting::Agent { .. })) || accepted {
            return Err(GraphError::ApplicationStateBusy);
        }
        Ok(())
    }
}

/// Validates persisted state even when the completed execution has no frames.
pub(super) fn validate_snapshot_application_state(state: &State) -> Result<(), GraphError> {
    if let Some((spec, value)) = &state.application_state {
        validate_state_value(spec, value)?;
    }
    Ok(())
}

/// Full schema checks are operation-local and run only at state entry/restore boundaries.
fn validate_state_value(spec: &TypeSpec, value: &Value) -> Result<(), GraphError> {
    if spec.name.trim().is_empty() {
        return Err(GraphError::ApplicationState(
            "state type name is empty".into(),
        ));
    }
    validate_value(spec, value, "application state")?;
    let validator = jsonschema::validator_for(&spec.schema)
        .map_err(|error| GraphError::ApplicationState(format!("invalid schema: {error}")))?;
    let json = serde_json::to_value(value).map_err(|error| GraphError::ValueConversion {
        target: "application state validation".into(),
        reason: error.to_string(),
    })?;
    if !validator.is_valid(&json) {
        return Err(GraphError::ApplicationState(
            "value does not satisfy its schema".into(),
        ));
    }
    Ok(())
}

fn missing_state() -> GraphError {
    GraphError::ApplicationState("no application state was initialized".into())
}

fn encode_state<S: Serialize>(state: S) -> Result<Value, GraphError> {
    to_value(state).map_err(|error| GraphError::ValueConversion {
        target: "application state".into(),
        reason: error.to_string(),
    })
}

#[cfg(test)]
#[path = "tests/application_state.rs"]
mod tests;
