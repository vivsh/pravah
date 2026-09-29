//! Asynchronous agent hooks, exclusively invoked by FetchExecutor outside the VM.

use super::effects::*;
use super::*;
use crate::graph::{DynFetchHandler, Fetch, FetchResponse, RuntimeServices};
use crate::history::{
    CompactionRequest, CompactionResult, protected_start, validate_message_groups,
};

impl DynFetchHandler for AgentHandler {
    fn execute<'a>(
        &'a self,
        fetch: &'a Fetch,
        context: Context,
    ) -> BoxFuture<'a, Result<FetchResponse, GraphError>> {
        async move {
            let hook = AgentHook::from_request(fetch.request())?;
            check_protocol(hook.version)?;
            let payload = self.validated_payload(&hook.payload)?;
            if hook.handler != payload.agent_id {
                return Err(GraphError::FetchValidation(
                    "agent hook identity mismatch".into(),
                ));
            }
            match hook.operation {
                AgentHookOperation::Configure { input } => {
                    let config = self
                        .configure
                        .configure(input, payload.configuration, context.clone())
                        .await?;
                    let (resolved, message, budget) =
                        resolve_agent_config(&payload, config, &context).await?;
                    response(Configured {
                        resolved,
                        message,
                        budget,
                    })
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
    services: &RuntimeServices,
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
        "pravah://prepare" => prepare(fetch, context, services).await,
        "pravah://history" => {
            let request: RecordRequest = decode(fetch.request().body_ref())?;
            check_protocol(request.version)?;
            for entry in request.entries {
                services
                    .store()
                    .record_dyn(&entry)
                    .await
                    .map_err(|error| GraphError::HistoryPersistence(error.to_string()))?;
            }
            response(())
        }
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
    check_protocol(version)?;
    body.get("handler")
        .and_then(Value::as_str)
        .ok_or_else(|| GraphError::FetchValidation("missing hook handler".into()))
}

/// Computes one candidate and materializes its attachments without mutating committed history.
async fn prepare(
    fetch: &Fetch,
    context: &Context,
    services: &RuntimeServices,
) -> Result<FetchResponse, GraphError> {
    let mut preparation: PreparationRequest = decode(fetch.request().body_ref())?;
    check_protocol(preparation.version)?;
    let mut guidance = std::mem::take(&mut preparation.guidance);
    let decision = prepare_policy(&preparation, &mut guidance, context, services).await?;
    let messages = super::preparation::messages(
        &preparation,
        fetch.request(),
        decision.clone(),
        guidance,
        context,
    )
    .await?;
    let source = super::effect_values::body_field(fetch.request().body_ref(), "request")?;
    let generation = crate::graph::fetch::rath::RathRequest::replace_messages(source, messages)?;
    Prepared {
        version: 1,
        decision,
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

/// Resolves policy inputs locally; without a policy or reminder the client is created at generation.
async fn prepare_policy(
    preparation: &PreparationRequest,
    guidance: &mut Vec<Message>,
    context: &Context,
    services: &RuntimeServices,
) -> Result<CompactionResult, GraphError> {
    let compactor = services.compactor();
    if compactor.is_none() && !preparation.budget_conclusion {
        return Ok(CompactionResult::default());
    }
    let client = preparation_client(preparation, context).await?;
    if preparation.budget_conclusion {
        guidance.push(Message::user(crate::clients::conclusion_message(
            &client.provider(),
        )));
    }
    let Some(compactor) = compactor else {
        return Ok(CompactionResult::default());
    };
    let entries = preparation.entries.iter().collect::<Vec<_>>();
    validate_message_groups(entries.iter().map(|entry| &entry.message)).map_err(|reason| {
        GraphError::HistoryCompactionValidation {
            session_id: preparation.session.clone(),
            reason,
        }
    })?;
    let (committed, protected) = entries.split_at(protected_start(&entries));
    compactor
        .compact_dyn(
            CompactionRequest {
                session_id: &preparation.session,
                model: preparation.request.model(),
                options: client.options(),
                framework_messages: guidance,
                committed,
                protected,
            },
            context.clone(),
        )
        .await
}
