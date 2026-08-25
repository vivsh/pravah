use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Context;

use super::error::GraphError;
use super::model::{TypeSpec, UntypedGraph};
use super::registry::{HandlerRegistry, RuntimeServices};
use super::runtime::{PreparedGraph, Runtime, Snapshot};
use super::state::Step;
use super::value::to_value;

/// Current JSON invocation request and response version.
pub const JSON_WIRE_VERSION: u32 = 4;

/// One external command for a trusted graph-backed workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation")]
pub enum JsonRequest {
    /// Starts a new workflow and advances it by one VM step.
    Start { version: u32, input: Value },
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
            Self::Start { version, .. }
            | Self::Next { version, .. }
            | Self::Resume { version, .. } => *version,
        }
    }
}

/// Result of exactly one JSON-driven VM operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum JsonResponse {
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
    services: RuntimeServices,
}

impl JsonInvoker {
    /// Binds a validated graph to the only handlers external requests may use.
    pub fn new(graph: UntypedGraph, registry: HandlerRegistry) -> Result<Self, GraphError> {
        let prepared = PreparedGraph::new(graph, registry)?;
        Ok(Self {
            prepared,
            services: RuntimeServices::new(),
        })
    }

    /// Configures runtime-owned provider, memory, compaction, and history services.
    pub fn with_services(mut self, services: RuntimeServices) -> Self {
        self.services = services;
        self
    }

    /// Decodes and executes one request, returning an encoded response.
    pub async fn invoke_str(&self, request: &str, ctx: Context) -> Result<String, GraphError> {
        let request = serde_json::from_str(request).map_err(|err| GraphError::JsonDecode {
            target: "invocation request".into(),
            reason: err.to_string(),
        })?;
        let response = self.invoke(request, ctx).await?;
        serde_json::to_string(&response).map_err(|err| GraphError::JsonEncode {
            target: "invocation response".into(),
            reason: err.to_string(),
        })
    }

    /// Executes exactly one start, next, or resume operation.
    pub async fn invoke(
        &self,
        request: JsonRequest,
        ctx: Context,
    ) -> Result<JsonResponse, GraphError> {
        validate_wire_version(request.version())?;
        let (mut runtime, step) = match request {
            JsonRequest::Start { input, .. } => {
                let graph = self.prepared.graph();
                let entry = graph.edge(graph.entry).ok_or_else(|| {
                    GraphError::GraphValidation("trusted graph entry edge is missing".into())
                })?;
                validate_external_value(&entry.type_spec, &input, "start input")?;
                let input = boundary_to_runtime(input, "start input")?;
                let mut runtime = self
                    .prepared
                    .start(input)?
                    .with_runtime_services(self.services.clone());
                let step = runtime.next(ctx).await?;
                (runtime, step)
            }
            JsonRequest::Next { snapshot, .. } => {
                let mut runtime = self.restore(snapshot)?;
                let step = runtime.next(ctx).await?;
                (runtime, step)
            }
            JsonRequest::Resume {
                snapshot, input, ..
            } => {
                let mut runtime = self.restore(snapshot)?;
                validate_external_value(runtime.suspension_type_spec()?, &input, "resume input")?;
                let input = boundary_to_runtime(input, "resume input")?;
                let step = runtime.resume(input, ctx).await?;
                (runtime, step)
            }
        };
        response_from_step(&mut runtime, step)
    }

    fn restore(&self, snapshot: Snapshot) -> Result<Runtime, GraphError> {
        self.prepared
            .restore(snapshot)
            .map(|runtime| runtime.with_runtime_services(self.services.clone()))
    }
}

fn response_from_step(runtime: &mut Runtime, step: Step) -> Result<JsonResponse, GraphError> {
    let suspension_type = runtime
        .suspension()
        .map(|suspension| suspension.resume_type.clone());
    let snapshot = runtime.snapshot()?;
    Ok(match step {
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
mod tests {
    use serde_json::json;

    use crate::{Context, FlowConf};

    use super::*;
    use crate::graph::{BuiltinNode, HandlerKey, NodeKind, TypeSpec, UntypedGraphBuilder};

    fn ctx() -> Context {
        Context::new(FlowConf::default())
    }

    fn suspended_graph() -> UntypedGraph {
        let mut builder = UntypedGraphBuilder::new("json_suspend");
        let input = builder.edge("input", TypeSpec::new("Number", json!({"type": "number"})));
        let waiting = builder.edge(
            "waiting",
            TypeSpec::new("Number", json!({"type": "number"})),
        );
        let output = builder.edge("output", TypeSpec::new("Number", json!({"type": "number"})));
        builder.set_entry(input).set_exit(output);
        builder.node(
            "prepare",
            NodeKind::Builtin {
                op: BuiltinNode::Identity,
            },
            vec![input],
            vec![waiting],
        );
        builder.node(
            "approve",
            NodeKind::Suspend {
                resume_type: "Number".into(),
                payload: to_value(json!({"prompt": "replacement number"}))
                    .expect("payload should enter runtime domain"),
            },
            vec![waiting],
            vec![output],
        );
        builder.build().expect("JSON test graph should build")
    }

    /// Verifies stateless JSON requests preserve one-step execution and resume state.
    #[tokio::test]
    async fn json_invocation_runs_start_next_resume_done() {
        let invoker = JsonInvoker::new(suspended_graph(), HandlerRegistry::new())
            .expect("invoker should build");
        let first = invoker
            .invoke(
                JsonRequest::Start {
                    version: JSON_WIRE_VERSION,
                    input: json!(1),
                },
                ctx(),
            )
            .await
            .expect("start should advance");
        let JsonResponse::Continue { snapshot, .. } = first else {
            panic!("start should execute only the prepare node");
        };
        let second = invoker
            .invoke(
                JsonRequest::Next {
                    version: JSON_WIRE_VERSION,
                    snapshot,
                },
                ctx(),
            )
            .await
            .expect("next should suspend");
        let JsonResponse::Suspend {
            snapshot,
            resume_type,
            payload,
            ..
        } = second
        else {
            panic!("next should reach suspension");
        };
        assert_eq!(resume_type, "Number");
        assert_eq!(payload, json!({"prompt": "replacement number"}));

        let done = invoker
            .invoke(
                JsonRequest::Resume {
                    version: JSON_WIRE_VERSION,
                    snapshot,
                    input: json!(7),
                },
                ctx(),
            )
            .await
            .expect("resume should finish");
        assert!(matches!(done, JsonResponse::Done { output, .. } if output == json!(7)));
    }

    /// Verifies callers cannot substitute a different graph through a snapshot.
    #[tokio::test]
    async fn json_invocation_rejects_snapshot_graph_substitution() {
        let invoker = JsonInvoker::new(suspended_graph(), HandlerRegistry::new())
            .expect("invoker should build");
        let JsonResponse::Continue { mut snapshot, .. } = invoker
            .invoke(
                JsonRequest::Start {
                    version: JSON_WIRE_VERSION,
                    input: json!(1),
                },
                ctx(),
            )
            .await
            .expect("start should advance")
        else {
            panic!("start should continue");
        };
        let mut substituted = suspended_graph();
        substituted.name = "substituted".into();
        snapshot.graph_fingerprint = PreparedGraph::new(substituted, HandlerRegistry::new())
            .expect("substituted graph should prepare")
            .fingerprint();

        let err = invoker
            .invoke(
                JsonRequest::Next {
                    version: JSON_WIRE_VERSION,
                    snapshot,
                },
                ctx(),
            )
            .await
            .expect_err("substituted graph should fail");
        assert!(matches!(err, GraphError::GraphMismatch { .. }));
    }

    /// Verifies the JSON boundary performs full schema validation before execution.
    #[tokio::test]
    async fn json_invocation_rejects_invalid_external_value() {
        let invoker = JsonInvoker::new(suspended_graph(), HandlerRegistry::new())
            .expect("invoker should build");
        let err = invoker
            .invoke(
                JsonRequest::Start {
                    version: JSON_WIRE_VERSION,
                    input: json!("not a number"),
                },
                ctx(),
            )
            .await
            .expect_err("invalid input should fail");
        assert!(matches!(err, GraphError::Schema { .. }));
    }

    /// Verifies malformed JSON and unsupported wire versions fail explicitly.
    #[tokio::test]
    async fn json_invocation_rejects_bad_wire_input() {
        let invoker = JsonInvoker::new(suspended_graph(), HandlerRegistry::new())
            .expect("invoker should build");
        let malformed = invoker
            .invoke_str("{", ctx())
            .await
            .expect_err("malformed JSON should fail");
        assert!(matches!(malformed, GraphError::JsonDecode { .. }));

        let version = invoker
            .invoke(
                JsonRequest::Start {
                    version: JSON_WIRE_VERSION + 1,
                    input: json!(1),
                },
                ctx(),
            )
            .await
            .expect_err("unsupported version should fail");
        assert!(matches!(version, GraphError::UnsupportedVersion { .. }));

        let obsolete = invoker
            .invoke(
                JsonRequest::Start {
                    version: 3,
                    input: json!(1),
                },
                ctx(),
            )
            .await
            .expect_err("obsolete wire version should fail");
        assert!(matches!(obsolete, GraphError::UnsupportedVersion { .. }));
    }

    /// Verifies trusted graphs cannot reference handlers absent from the host registry.
    #[test]
    fn json_invoker_rejects_missing_handler() {
        let mut builder = UntypedGraphBuilder::new("missing_handler");
        let input = builder.edge("input", TypeSpec::new("Number", json!({"type": "number"})));
        let output = builder.edge("output", TypeSpec::new("Number", json!({"type": "number"})));
        builder.set_entry(input).set_exit(output);
        builder.node(
            "missing",
            NodeKind::PureHandler {
                key: HandlerKey::new("missing"),
            },
            vec![input],
            vec![output],
        );
        let graph = builder.build().expect("graph shape should be valid");

        assert!(JsonInvoker::new(graph, HandlerRegistry::new()).is_err());
    }
}
