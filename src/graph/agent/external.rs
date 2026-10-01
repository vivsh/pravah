//! Asynchronous agent hooks, exclusively invoked by FetchExecutor outside the VM.

use super::effects::*;
use super::*;
use crate::graph::{DynFetchHandler, Fetch, FetchResponse};

impl DynFetchHandler for AgentHandler {
    fn execute<'a>(
        &'a self,
        fetch: &'a Fetch,
        context: Context,
    ) -> BoxFuture<'a, Result<FetchResponse, GraphError>> {
        async move {
            let hook = AgentHook::from_request(fetch.request())?;
            check_agent_hook_version(hook.version)?;
            let payload = self.validated_payload(&hook.payload)?;
            if hook.handler != payload.agent_id {
                return Err(GraphError::FetchValidation(
                    "agent hook identity mismatch".into(),
                ));
            }
            match hook.operation {
                AgentHookOperation::Configure {
                    input,
                    execution_id,
                } => {
                    let config = self
                        .configure
                        .configure(input, payload.configuration, execution_id, context.clone())
                        .await?;
                    response(resolve_agent_config(&payload, config, &context).await?)
                }
                AgentHookOperation::Control { observation } => {
                    let configured = observation
                        .configured_tools
                        .iter()
                        .map(|tool| tool.name().to_owned())
                        .collect::<Vec<_>>();
                    let controller = self
                        .controller
                        .as_ref()
                        .ok_or_else(|| GraphError::FetchValidation("missing controller".into()))?;
                    let decision = controller.control(observation, context).await?;
                    response(resolve_decision(decision, &payload, &configured)?)
                }
            }
        }
        .boxed()
    }
}

/// Resolves predicates to portable ordered names; no executable closure crosses the boundary.
fn resolve_decision(
    decision: AgentDecision,
    payload: &AgentPayloadView<'_>,
    configured: &[String],
) -> Result<ResolvedDecision, GraphError> {
    super::intervention::validate_decision(&decision)?;
    let kind = match decision.kind {
        AgentDecisionKind::Continue => DecisionKind::Continue,
        AgentDecisionKind::Conclude(text) => DecisionKind::Conclude(text),
        AgentDecisionKind::Suspend(value) => DecisionKind::Suspend(value),
        AgentDecisionKind::Abort(reason) => DecisionKind::Abort(reason),
        AgentDecisionKind::Redirect(directive) => {
            let tools = if let Some(filter) = directive.tool_filter {
                filter
                    .validate_names(configured.iter().map(String::as_str))
                    .map_err(GraphError::AgentControlValidation)?;
                Some(
                    payload
                        .tools
                        .iter()
                        .filter(|tool| configured.contains(&tool.name) && filter.allows(tool))
                        .map(|tool| tool.name.clone())
                        .collect(),
                )
            } else {
                directive.tool_names
            };
            DecisionKind::Redirect {
                guidance: directive.guidance,
                tools,
            }
        }
    };
    Ok(ResolvedDecision {
        kind,
        state: decision.state,
    })
}

/// Routes explicit framework operations without invoking a VM or retaining execution state.
pub(crate) async fn execute_hook(
    fetch: &Fetch,
    context: &Context,
    registry: &HandlerRegistry,
) -> Result<FetchResponse, GraphError> {
    if fetch.request().method() != "POST" || !fetch.request().headers().is_empty() {
        return Err(GraphError::FetchValidation(
            "invalid framework hook envelope".into(),
        ));
    }
    match fetch.request().url() {
        "pravah://tool" | "pravah://agent" => {
            let key = hook_handler(fetch)?;
            let handler = registry
                .fetch(&crate::graph::HandlerKey::new(key))
                .ok_or_else(|| {
                    GraphError::FetchValidation("missing external hook handler".into())
                })?;
            handler.execute(fetch, context.clone()).await
        }
        "pravah://agent-prepare" => prepare(fetch, context).await,
        _ => Err(GraphError::FetchValidation("unknown framework hook".into())),
    }
}

/// Borrows only routing fields; the selected handler validates and decodes its complete input.
fn hook_handler(fetch: &Fetch) -> Result<&str, GraphError> {
    let Some(crate::graph::FetchBody::Value(body)) = fetch.request().body_ref() else {
        return Err(GraphError::FetchValidation(
            "expected structured hook payload".into(),
        ));
    };
    let version = body
        .get("version")
        .and_then(Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| GraphError::FetchValidation("missing hook version".into()))?;
    if fetch.request().url() == "pravah://agent" {
        check_agent_hook_version(version)?;
    } else {
        check_protocol(version)?;
    }
    body.get("handler")
        .and_then(Value::as_str)
        .ok_or_else(|| GraphError::FetchValidation("missing hook handler".into()))
}

/// Computes one candidate and materializes its attachments without mutating committed history.
async fn prepare(fetch: &Fetch, context: &Context) -> Result<FetchResponse, GraphError> {
    let mut preparation: PreparationRequest = decode(fetch.request().body_ref())?;
    check_protocol(preparation.version)?;
    let mut guidance = std::mem::take(&mut preparation.guidance);
    if preparation.budget_conclusion {
        let client = preparation_client(&preparation, context).await?;
        guidance.push(Message::user(crate::clients::conclusion_message(
            &client.provider(),
        )));
    }
    let messages =
        super::preparation::messages(&preparation, fetch.request(), guidance, context).await?;
    let source = super::effect_values::body_field(fetch.request().body_ref(), "request")?;
    let generation = crate::graph::fetch::rath::RathRequest::replace_messages(source, messages)?;
    Prepared {
        version: 1,
        generation: generation.into_value()?,
    }
    .into_response()
}

/// Constructs a client only when preparation needs provider-effective configuration or guidance.
async fn preparation_client(
    preparation: &PreparationRequest,
    context: &Context,
) -> Result<crate::clients::Client, GraphError> {
    context
        .providers()
        .llm(
            preparation.request.model(),
            preparation.request.options().clone(),
        )
        .await
        .map_err(|source| GraphError::AgentClient {
            operation: crate::graph::AgentClientOperation::Create,
            source,
        })
}
