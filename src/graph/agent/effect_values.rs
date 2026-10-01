//! Concrete durable envelopes that retain shared VM values instead of recoding them.

use super::effects::*;
use super::function_tool::ToolHook;
use super::*;
use crate::graph::{FetchBody, FetchRequest, FetchResponse};
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

/// Borrows a required field from a structured boundary body, rejecting malformed envelopes.
pub(super) fn body_field<'a>(
    body: Option<&'a FetchBody>,
    name: &str,
) -> Result<&'a Value, GraphError> {
    let Some(FetchBody::Value(value)) = body else {
        return Err(invalid());
    };
    field(value, name)
}

/// Serializes borrowed history once; the request owns its encoded values after this operation.
pub(super) fn preparation_request(
    checkpoint: &EdgeAgentCheckpoint,
    request: Value,
    guidance: Vec<Message>,
    budget_conclusion: bool,
    ctx: ContinuationContext<'_>,
) -> Result<FetchRequest, GraphError> {
    let entries = ctx
        .history()
        .session_entries(&checkpoint.session_id)
        .into_iter()
        .map(|entry| encode(&entry.message))
        .collect::<Result<Vec<_>, _>>()?;
    let request =
        crate::graph::fetch::rath::RathRequest::replace_messages(&request, Value::array(entries))?;
    let Some(FetchBody::Value(source)) = request.body_ref() else {
        return Err(invalid());
    };
    let body = object([
        ("version", 1_u32.into()),
        ("request", source.clone()),
        ("guidance", encode(guidance)?),
        ("budget_conclusion", budget_conclusion.into()),
    ])?;
    Ok(FetchRequest::new("POST", "pravah://agent-prepare").body(FetchBody::Value(body)))
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
            Self::Prepare {
                version,
                checkpoint,
            } => boundary("prepare", version, checkpoint),
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
            "control" | "prepare" | "generate" => &["effect", "version", "checkpoint"][..],
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
            "prepare" => Self::Prepare {
                version,
                checkpoint: field(value, "checkpoint")?.clone(),
            },
            "generate" => Self::Generate {
                version,
                checkpoint: field(value, "checkpoint")?.clone(),
            },
            _ => return Err(invalid()),
        })
    }
}

impl Prepared {
    pub(super) fn into_response(self) -> Result<FetchResponse, GraphError> {
        Ok(FetchResponse::new(200).body(FetchBody::Value(object([
            ("version", self.version.into()),
            ("generation", self.generation),
        ])?)))
    }

    pub(super) fn from_response(response: &FetchResponse) -> Result<Self, GraphError> {
        let Some(FetchBody::Value(value)) = response.body_ref() else {
            return Err(invalid());
        };
        Ok(Self {
            version: read_field(value, "version")?,
            generation: field(value, "generation")?.clone(),
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

impl AgentHook {
    /// Builds an operation-local request that shares its authored payload and invocation input.
    pub(super) fn into_request(self) -> Result<FetchRequest, GraphError> {
        let operation = match self.operation {
            AgentHookOperation::Configure {
                input,
                execution_id,
            } => object([(
                "Configure",
                object([("input", input), ("execution_id", encode(execution_id)?)])?,
            )])?,
            AgentHookOperation::Control { observation } => object([(
                "Control",
                object([("observation", observation_value(observation)?)])?,
            )])?,
        };
        Ok(
            FetchRequest::new("POST", "pravah://agent").body(FetchBody::Value(object([
                ("version", self.version.into()),
                ("handler", self.handler.into()),
                ("payload", self.payload),
                ("operation", operation),
            ])?)),
        )
    }

    /// Reads only the selected callback envelope; authored metadata remains shared.
    pub(super) fn from_request(request: &FetchRequest) -> Result<Self, GraphError> {
        let Some(FetchBody::Value(value)) = request.body_ref() else {
            return Err(invalid());
        };
        let operation = field(value, "operation")?;
        let mut variants = operation.object_entries().ok_or_else(invalid)?;
        let (kind, content) = variants.next().ok_or_else(invalid)?;
        if variants.next().is_some() {
            return Err(invalid());
        }
        let operation = match kind {
            "Configure" => AgentHookOperation::Configure {
                input: field(content, "input")?.clone(),
                execution_id: read_field(content, "execution_id")?,
            },
            "Control" => AgentHookOperation::Control {
                observation: read_observation(field(content, "observation")?)?,
            },
            _ => return Err(invalid()),
        };
        Ok(Self {
            version: read_field(value, "version")?,
            handler: read_field(value, "handler")?,
            payload: field(value, "payload")?.clone(),
            operation,
        })
    }
}

impl ToolHook {
    /// Moves the existing tool input into its durable envelope without traversing it.
    pub(super) fn into_request(self) -> Result<FetchRequest, GraphError> {
        Ok(
            FetchRequest::new("POST", "pravah://tool").body(FetchBody::Value(object([
                ("version", self.version.into()),
                ("handler", self.handler.into()),
                ("input", self.input),
            ])?)),
        )
    }

    /// Checks routing fields while sharing the opaque input with the caller's request.
    pub(super) fn from_request(request: &FetchRequest) -> Result<Self, GraphError> {
        let Some(FetchBody::Value(value)) = request.body_ref() else {
            return Err(invalid());
        };
        Ok(Self {
            version: read_field(value, "version")?,
            handler: read_field(value, "handler")?,
            input: field(value, "input")?.clone(),
        })
    }
}

/// Keeps user-owned controller input/state shared while encoding the observation metadata.
fn observation_value(data: AgentLoopData) -> Result<Value, GraphError> {
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
fn read_observation(value: &Value) -> Result<AgentLoopData, GraphError> {
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
