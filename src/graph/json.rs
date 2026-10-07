use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent_request::{AgentRequest, AgentResponse};
use uuid::Uuid;

use super::error::GraphError;
use super::model::{TypeSpec, UntypedGraph};
use super::registry::HandlerRegistry;
use super::runtime::{PreparedGraph, Runtime, Snapshot};
use super::state::Step;
use super::value::to_value;

/// Current JSON invocation request and response version.
pub const JSON_WIRE_VERSION: u32 = 11;

/// One external command for a trusted graph-backed workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation")]
pub enum JsonRequest {
    /// Accepts one recorded external outcome without advancing execution.
    ResumeAgent {
        version: u32,
        snapshot: Snapshot,
        response: AgentResponse,
    },
    /// Starts a new workflow and advances it by one VM step.
    Start {
        version: u32,
        input: Value,
        execution_id: Uuid,
    },
    /// Restores a snapshot and advances it by one VM step.
    Next { version: u32, snapshot: Snapshot },
    /// Restores a suspended snapshot and supplies its external input.
    Resume {
        version: u32,
        snapshot: Snapshot,
        input: Value,
    },
}

impl JsonRequest {
    fn version(&self) -> u32 {
        match self {
            Self::ResumeAgent { version, .. }
            | Self::Start { version, .. }
            | Self::Next { version, .. }
            | Self::Resume { version, .. } => *version,
        }
    }
}

/// Result of exactly one JSON-driven VM operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum JsonResponse {
    /// One durable external operation is awaiting outcome delivery.
    Agent {
        version: u32,
        request: AgentRequest,
        snapshot: Snapshot,
    },
    /// The workflow advanced and can be stepped again.
    Continue { version: u32, snapshot: Snapshot },
    /// The workflow requires an external value of the named type.
    Suspend {
        version: u32,
        payload: Value,
        resume_type: String,
        snapshot: Snapshot,
    },
    /// The root frame completed with a JSON output value.
    Done {
        version: u32,
        output: Value,
        snapshot: Snapshot,
    },
}

/// Stateless JSON facade bound to one trusted graph and handler registry.
///
/// Applications own transport, authentication, snapshot storage, and retries.
/// The invoker never accepts graphs or executable handlers from callers.
#[derive(Clone)]
pub struct JsonInvoker {
    prepared: PreparedGraph,
}

impl JsonInvoker {
    /// Binds a validated graph to the only handlers external requests may use.
    pub fn new(graph: UntypedGraph, registry: HandlerRegistry) -> Result<Self, GraphError> {
        let prepared = PreparedGraph::new(graph, registry)?;
        Ok(Self { prepared })
    }

    /// Decodes and executes one request, returning an encoded response.
    pub fn invoke_str(&self, request: &str) -> Result<String, GraphError> {
        let request = serde_json::from_str(request).map_err(|err| GraphError::JsonDecode {
            target: "invocation request".into(),
            reason: err.to_string(),
        })?;
        let response = self.invoke(request)?;
        serde_json::to_string(&response).map_err(|err| GraphError::JsonEncode {
            target: "invocation response".into(),
            reason: err.to_string(),
        })
    }

    /// Executes exactly one start, next, or resume operation.
    pub fn invoke(&self, request: JsonRequest) -> Result<JsonResponse, GraphError> {
        validate_wire_version(request.version())?;
        let (mut runtime, step) = match request {
            JsonRequest::ResumeAgent {
                snapshot, response, ..
            } => {
                let mut runtime = self.restore(snapshot)?;
                runtime.resume_agent(response)?;
                (runtime, Step::Continue)
            }
            JsonRequest::Start {
                input,
                execution_id,
                ..
            } => {
                let graph = self.prepared.graph();
                let entry = graph.edge(graph.entry).ok_or_else(|| {
                    GraphError::GraphValidation("trusted graph entry edge is missing".into())
                })?;
                validate_external_value(&entry.type_spec, &input, "start input")?;
                let input = boundary_to_runtime(input, "start input")?;
                let mut runtime = self.prepared.start(input, execution_id)?;
                let step = runtime.next()?;
                (runtime, step)
            }
            JsonRequest::Next { snapshot, .. } => {
                let mut runtime = self.restore(snapshot)?;
                let step = runtime.next()?;
                (runtime, step)
            }
            JsonRequest::Resume {
                snapshot, input, ..
            } => {
                let mut runtime = self.restore(snapshot)?;
                validate_external_value(runtime.suspension_type_spec()?, &input, "resume input")?;
                let input = boundary_to_runtime(input, "resume input")?;
                runtime.resume_value(input)?;
                (runtime, Step::Continue)
            }
        };
        response_from_step(&mut runtime, step)
    }

    fn restore(&self, snapshot: Snapshot) -> Result<Runtime, GraphError> {
        self.prepared.restore(snapshot)
    }
}

fn response_from_step(runtime: &mut Runtime, step: Step) -> Result<JsonResponse, GraphError> {
    let suspension_type = runtime
        .suspension()
        .map(|suspension| suspension.resume_type().to_owned());
    let snapshot = runtime.snapshot()?;
    Ok(match step {
        Step::Agent(request) => JsonResponse::Agent {
            version: JSON_WIRE_VERSION,
            request,
            snapshot,
        },
        Step::Continue => JsonResponse::Continue {
            version: JSON_WIRE_VERSION,
            snapshot,
        },
        Step::Suspend(payload) => JsonResponse::Suspend {
            version: JSON_WIRE_VERSION,
            payload: runtime_to_boundary(payload, "suspend payload")?,
            resume_type: suspension_type.ok_or_else(|| {
                GraphError::SnapshotValidation("suspend step has no suspension state".into())
            })?,
            snapshot,
        },
        Step::Done(output) => JsonResponse::Done {
            version: JSON_WIRE_VERSION,
            output: runtime_to_boundary(output, "workflow output")?,
            snapshot,
        },
    })
}

fn boundary_to_runtime(value: Value, target: &str) -> Result<super::Value, GraphError> {
    to_value(value).map_err(|err| GraphError::ValueConversion {
        target: target.into(),
        reason: err.to_string(),
    })
}

fn runtime_to_boundary(value: super::Value, target: &str) -> Result<Value, GraphError> {
    serde_json::to_value(value).map_err(|err| GraphError::JsonEncode {
        target: target.into(),
        reason: err.to_string(),
    })
}

fn validate_wire_version(version: u32) -> Result<(), GraphError> {
    if version == JSON_WIRE_VERSION {
        Ok(())
    } else {
        Err(GraphError::UnsupportedVersion {
            format: "JSON invocation wire",
            got: version,
            expected: JSON_WIRE_VERSION,
        })
    }
}

fn validate_external_value(
    type_spec: &TypeSpec,
    value: &Value,
    label: &str,
) -> Result<(), GraphError> {
    let validator = jsonschema::validator_for(&type_spec.schema).map_err(|err| {
        GraphError::GraphValidation(format!(
            "schema '{}' cannot be compiled: {err}",
            type_spec.name
        ))
    })?;
    validator.validate(value).map_err(|err| GraphError::Schema {
        label: label.into(),
        expected: type_spec.name.clone(),
        value: err.to_string(),
    })
}

#[cfg(test)]
#[path = "tests/json.rs"]
mod tests;
