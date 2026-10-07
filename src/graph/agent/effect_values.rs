//! Concrete durable envelopes that retain shared VM values instead of recoding them.

use super::effects::*;
use super::*;
use std::borrow::Cow;

fn invalid() -> GraphError {
    GraphError::SnapshotValidation("invalid agent boundary value".into())
}

fn object<const N: usize>(fields: [(&'static str, Value); N]) -> Result<Value, GraphError> {
    Value::from_shared_object(
        fields
            .into_iter()
            .map(|(key, value)| (Cow::Borrowed(key), value))
            .collect(),
    )
    .map_err(|_| invalid())
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, GraphError> {
    value.get(name).ok_or_else(invalid)
}

pub(super) fn read_field<T: DeserializeOwned>(value: &Value, name: &str) -> Result<T, GraphError> {
    from_value(field(value, name)?.clone()).map_err(|_| invalid())
}

fn optional_value(value: &Value, name: &str) -> Option<Value> {
    value.get(name).filter(|value| !value.is_null()).cloned()
}

impl AgentEffectCheckpoint {
    /// Only the envelope is new; input and nested checkpoints keep their shared ownership.
    pub(super) fn into_value(self) -> Result<Value, GraphError> {
        match self {
            Self::Configure { version, input } => object([
                ("effect", "configure".into()),
                ("version", version.into()),
                ("input", input),
            ]),
            Self::Control {
                version,
                checkpoint,
            } => boundary("control", version, checkpoint),
            Self::Flush {
                version,
                transition,
            } => object([
                ("effect", "flush".into()),
                ("version", version.into()),
                ("transition", transition),
            ]),
            Self::Generate {
                version,
                checkpoint,
            } => boundary("generate", version, checkpoint),
        }
    }

    /// Preserves opaque children while enforcing the same closed envelope as Serde.
    pub(super) fn from_value(value: &Value) -> Result<Self, GraphError> {
        let kind = field(value, "effect")?.as_str().ok_or_else(invalid)?;
        let version = read_field(value, "version")?;
        let allowed = match kind {
            "configure" => &["effect", "version", "input"][..],
            "control" | "generate" => &["effect", "version", "checkpoint"][..],
            "flush" => &["effect", "version", "transition"][..],
            _ => return Err(invalid()),
        };
        if value
            .object_entries()
            .ok_or_else(invalid)?
            .any(|(key, _)| !allowed.contains(&key))
        {
            return Err(invalid());
        }
        Self::read_kind(kind, version, value)
    }

    /// Reads the selected effect fields without decoding its nested checkpoint prematurely.
    fn read_kind(kind: &str, version: u32, value: &Value) -> Result<Self, GraphError> {
        Ok(match kind {
            "configure" => Self::Configure {
                version,
                input: field(value, "input")?.clone(),
            },
            "control" => Self::Control {
                version,
                checkpoint: field(value, "checkpoint")?.clone(),
            },
            "flush" => Self::Flush {
                version,
                transition: field(value, "transition")?.clone(),
            },
            "generate" => Self::Generate {
                version,
                checkpoint: field(value, "checkpoint")?.clone(),
            },
            _ => return Err(invalid()),
        })
    }
}

fn boundary(kind: &str, version: u32, checkpoint: Value) -> Result<Value, GraphError> {
    object([
        ("effect", kind.into()),
        ("version", version.into()),
        ("checkpoint", checkpoint),
    ])
}

impl EdgeAgentCheckpoint {
    /// Encodes mutable metadata once, without recursively copying invocation or controller values.
    pub(super) fn into_value(self) -> Result<Value, GraphError> {
        let mut fields = vec![
            ("version", self.version.into()),
            ("phase", encode(self.phase)?),
            ("session_id", self.session_id.into()),
            ("input", self.input),
            ("resolved", self.resolved),
            ("selected_tools", encode(self.selected_tools)?),
            ("guidance", encode(self.guidance)?),
            ("metrics", encode(self.metrics)?),
            ("control_state", self.control_state.unwrap_or(Value::NULL)),
        ];
        if let Some(budget) = self.budget {
            fields.push(("budget", encode(budget)?));
        }
        Value::from_shared_object(
            fields
                .into_iter()
                .map(|(key, value)| (Cow::Borrowed(key), value))
                .collect(),
        )
        .map_err(|_| invalid())
    }

    /// Decodes domain metadata and borrows the shared input subtrees for this operation.
    pub(super) fn from_value(value: &Value) -> Result<Self, GraphError> {
        Ok(Self {
            version: read_field(value, "version")?,
            phase: read_field(value, "phase")?,
            session_id: read_field(value, "session_id")?,
            input: field(value, "input")?.clone(),
            resolved: field(value, "resolved")?.clone(),
            selected_tools: read_field(value, "selected_tools")?,
            budget: optional_value(value, "budget")
                .map(from_value)
                .transpose()
                .map_err(|_| invalid())?,
            guidance: optional_value(value, "guidance")
                .map(from_value)
                .transpose()
                .map_err(|_| invalid())?,
            metrics: read_field(value, "metrics")?,
            control_state: optional_value(value, "control_state"),
        })
    }
}

/// Keeps user-owned controller input/state shared while encoding the observation metadata.
pub(super) fn observation_value(data: AgentLoopData) -> Result<Value, GraphError> {
    object([
        ("input", data.input),
        ("point", encode(data.point)?),
        ("agent_id", data.agent_id.into()),
        ("session_id", data.session_id.into()),
        ("configured_tools", encode(data.configured_tools)?),
        ("active_tools", encode(data.active_tools)?),
        ("history", encode(data.history)?),
        ("proposal", encode(data.proposal)?),
        ("results", encode(data.results)?),
        ("metrics", encode(data.metrics)?),
        ("budget", encode(data.budget)?),
        ("control_state", data.control_state.unwrap_or(Value::NULL)),
    ])
}

/// Reconstructs the callback's owned view without recursively cloning its opaque values.
pub(super) fn read_observation(value: &Value) -> Result<AgentLoopData, GraphError> {
    Ok(AgentLoopData {
        input: field(value, "input")?.clone(),
        point: read_field(value, "point")?,
        agent_id: read_field(value, "agent_id")?,
        session_id: read_field(value, "session_id")?,
        configured_tools: read_field(value, "configured_tools")?,
        active_tools: read_field(value, "active_tools")?,
        history: read_field(value, "history")?,
        proposal: read_field(value, "proposal")?,
        results: read_field(value, "results")?,
        metrics: read_field(value, "metrics")?,
        budget: read_field(value, "budget")?,
        control_state: optional_value(value, "control_state"),
    })
}

#[cfg(test)]
#[path = "tests/effect_values.rs"]
mod tests;
