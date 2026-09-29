use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use rmcp::model::{
    ErrorData, ListResourceTemplatesResult, ListResourcesResult, PaginatedRequestParams,
    ReadResourceResponse, ReadResourceResult, Resource, ResourceTemplate, ServerCapabilities,
    ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{RoleServer, ServerHandler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::clients::Message;
use crate::graph::tests::host;
use crate::graph::{Agent, AgentConfig, Flow, Step, compile};
use crate::testing::ScriptedFactory;
use std::sync::Arc;

#[derive(Clone, Default)]
struct ResourceServer;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResourceInput {
    question: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct ResourceOutput {
    answer: String,
}

async fn configure_resource_agent(
    input: ResourceInput,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test-model",
        "Use the selected resource.",
        Message::user(input.question),
    )
    .resources([McpResourceRef::new("docs", "docs://a-first")]))
}

fn resource_agent(root: Agent<ResourceInput>) -> Agent<ResourceOutput> {
    root.configure(configure_resource_agent)
}

fn resource_flow(root: Flow<ResourceInput>) -> Flow<ResourceOutput> {
    root.agent(resource_agent)
}

impl ServerHandler for ResourceServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_resources().build())
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult::with_all_items(vec![
            Resource::new("docs://z-last", "last"),
            Resource::new("docs://a-first", "first"),
        ]))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        Ok(ListResourceTemplatesResult::with_all_items(vec![
            ResourceTemplate::new("docs://guide/{name}", "guide"),
        ]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        let contents = if request.uri == "docs://blob" {
            ResourceContents::blob("AA==", request.uri)
        } else {
            ResourceContents::text(format!("text for {}", request.uri), request.uri)
        };
        Ok(ReadResourceResult::new(vec![contents]).into())
    }
}

async fn require_test_credentials(request: Request<Body>, next: Next) -> Response {
    let bearer = request
        .headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok());
    let tenant = request
        .headers()
        .get("x-tenant")
        .and_then(|value| value.to_str().ok());
    if bearer == Some("Bearer test-token") && tenant == Some("tenant-1") {
        next.run(request).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

/// Starts a credential-checking local Streamable HTTP MCP service.
async fn spawn_resource_server() -> (String, CancellationToken) {
    let config = StreamableHttpServerConfig::default()
        .with_json_response(true)
        .with_cancellation_token(CancellationToken::new());
    let cancellation = config.cancellation_token.clone();
    let service: StreamableHttpService<ResourceServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(ResourceServer), Default::default(), config);
    let router = axum::Router::new()
        .nest_service("/mcp", service)
        .layer(middleware::from_fn(require_test_credentials));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test MCP listener should bind");
    let address = listener
        .local_addr()
        .expect("test MCP listener should have an address");
    let shutdown = cancellation.clone();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await;
    });
    (format!("http://{address}/mcp"), cancellation)
}

fn test_context(url: String) -> Context {
    Context::default().with_mcp_server(
        McpServer::new("docs", url)
            .bearer_token("test-token")
            .header("x-tenant", "tenant-1"),
    )
}

/// Verifies URI-template arguments are encoded deterministically.
#[test]
fn template_arguments_expand_into_a_stable_uri() {
    let mut arguments = BTreeMap::new();
    arguments.insert("name".into(), "a/b".into());
    let resource = McpResourceRef::template("docs", "docs://{name}", arguments);

    assert_eq!(expand_uri(&resource).unwrap(), "docs://a%2Fb");
}

/// Verifies unresolved resource-template arguments fail before transport use.
#[test]
fn unresolved_template_arguments_are_rejected() {
    let resource = McpResourceRef::new("docs", "docs://{name}");

    assert!(matches!(
        expand_uri(&resource),
        Err(McpError::InvalidConfiguration(_))
    ));
}

/// Verifies selected text contents preserve server order and blob data is rejected.
#[test]
fn resource_contents_accept_only_ordered_text() {
    let text = text_contents(vec![
        ResourceContents::text("first", "docs://one"),
        ResourceContents::text("second", "docs://two"),
    ])
    .unwrap();
    let blob = text_contents(vec![ResourceContents::blob("AA==", "docs://blob")]);

    assert_eq!(text, "first\nsecond");
    assert!(matches!(blob, Err(McpError::UnsupportedContent(_))));
}

/// Verifies server diagnostics disclose neither bearer tokens nor header values.
#[test]
fn server_debug_redacts_runtime_credentials() {
    let server = McpServer::new("docs", "https://example.test/mcp")
        .bearer_token("secret-token")
        .header("x-api-key", "secret-key");
    let debug = format!("{server:?}");

    assert!(!debug.contains("secret-token"));
    assert!(!debug.contains("secret-key"));
    assert!(debug.contains("x-api-key"));
}

/// Verifies catalogs, templates, credentials, and text reads over Streamable HTTP.
#[tokio::test]
async fn streamable_http_catalog_and_resource_resolution_work() -> Result<(), crate::GraphError> {
    let (url, cancellation) = spawn_resource_server().await;
    let ctx = test_context(url);
    let catalog = ctx.mcp_resources("docs").await.unwrap();
    let uris = catalog.iter().map(McpResourceInfo::uri).collect::<Vec<_>>();
    let mut arguments = BTreeMap::new();
    arguments.insert("name".into(), "a/b".into());
    let refs = vec![McpResourceRef::template(
        "docs",
        "docs://guide/{name}",
        arguments,
    )];
    let resolved = resolve_resources(&ctx, &refs).await.unwrap();

    assert_eq!(
        uris,
        vec!["docs://a-first", "docs://guide/{name}", "docs://z-last"]
    );
    assert_eq!(resolved[0].uri, "docs://guide/a%2Fb");
    assert_eq!(resolved[0].text, "text for docs://guide/a%2Fb");
    cancellation.cancel();
    Ok(())
}

/// Verifies missing credentials fail and blob resources remain unsupported.
#[tokio::test]
async fn streamable_http_rejects_unauthorized_and_blob_resources() -> Result<(), crate::GraphError>
{
    let (url, cancellation) = spawn_resource_server().await;
    let unauthorized = Context::default()
        .with_mcp_server(McpServer::new("docs", url.clone()))
        .mcp_resources("docs")
        .await;
    let ctx = test_context(url);
    let blob = resolve_resources(&ctx, &[McpResourceRef::new("docs", "docs://blob")]).await;

    assert!(matches!(unauthorized, Err(McpError::Transport(_))));
    assert!(matches!(blob, Err(GraphError::McpResource(_))));
    cancellation.cancel();
    Ok(())
}

/// Verifies restored agents use checkpointed resource text without network access.
#[tokio::test]
async fn restored_agent_does_not_reread_mcp_resources() -> Result<(), GraphError> {
    let (url, cancellation) = spawn_resource_server().await;
    let flow = compile(resource_flow).unwrap();
    let executor =
        crate::graph::FetchExecutor::new(test_context(url), Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(
            ResourceInput {
                question: "question".into(),
            },
            uuid::Uuid::nil(),
        )
        .unwrap();
    while runtime.snapshot()?.history().entries().is_empty() {
        host::step(&mut runtime, &executor).await?;
    }
    let snapshot = runtime.snapshot().unwrap();
    let encoded = serde_json::to_string(&snapshot).unwrap();
    assert!(encoded.contains("text for docs://a-first"));
    cancellation.cancel();

    let factory =
        ScriptedFactory::new().then_output(serde_json::json!({ "answer": "from checkpoint" }));
    let executor = crate::graph::FetchExecutor::new(
        Context::default().with_providers(crate::testing::providers(factory)?),
        Arc::new(flow.registry().clone()),
    );
    let mut restored = flow.restore(snapshot).unwrap();
    let step = host::finish(&mut restored, &executor).await?;
    let Step::Done(value) = step else {
        panic!("restored agent should complete");
    };
    assert_eq!(
        flow.decode_output(value).unwrap(),
        ResourceOutput {
            answer: "from checkpoint".into()
        }
    );
    Ok(())
}
