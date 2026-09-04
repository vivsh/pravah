use std::collections::{BTreeMap, HashMap};
use std::fmt;

use http::{HeaderName, HeaderValue};
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use rmcp::ServiceExt;
use rmcp::model::{ReadResourceRequestParams, ResourceContents};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use thiserror::Error;

use crate::Context;

use super::agent::ResolvedResource;
use super::{GraphError, McpResourceRef};

/// Runtime-only Streamable HTTP MCP server configuration.
#[derive(Clone)]
pub struct McpServer {
    id: String,
    url: String,
    bearer_token: Option<String>,
    headers: BTreeMap<String, String>,
}

impl fmt::Debug for McpServer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpServer")
            .field("id", &self.id)
            .field("url", &self.url)
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "<redacted>"),
            )
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl McpServer {
    /// Creates a named Streamable HTTP server registration.
    pub fn new(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            bearer_token: None,
            headers: BTreeMap::new(),
        }
    }

    /// Adds the bearer token sent to this server at runtime.
    pub fn bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Adds one custom HTTP header sent to this server at runtime.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Returns the application-defined server identifier.
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// Resource or resource-template metadata returned by an MCP server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpResourceInfo {
    uri: String,
    name: String,
    title: Option<String>,
    description: Option<String>,
    template: bool,
}

impl McpResourceInfo {
    /// Returns the resource URI or URI template.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Returns the programmatic resource name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the optional display title.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Returns the optional resource description.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Reports whether the URI requires template arguments.
    pub fn is_template(&self) -> bool {
        self.template
    }
}

/// Failure while configuring or using an MCP resource server.
#[derive(Debug, Error)]
pub enum McpError {
    /// No server exists under the requested identifier.
    #[error("MCP server '{0}' is not configured")]
    MissingServer(String),
    /// A configured URL or header is invalid.
    #[error("invalid MCP server configuration: {0}")]
    InvalidConfiguration(String),
    /// The Streamable HTTP session could not be started or used.
    #[error("MCP transport failed: {0}")]
    Transport(String),
    /// The selected resource content is unsupported.
    #[error("unsupported MCP resource content: {0}")]
    UnsupportedContent(String),
}

pub(crate) async fn list_resources(
    ctx: &Context,
    server_id: &str,
) -> Result<Vec<McpResourceInfo>, McpError> {
    let server = ctx
        .mcp_server(server_id)
        .ok_or_else(|| McpError::MissingServer(server_id.into()))?;
    let client = connect(server).await?;
    let resources = client
        .list_all_resources()
        .await
        .map_err(|err| McpError::Transport(err.to_string()))?;
    let templates = client
        .list_all_resource_templates()
        .await
        .map_err(|err| McpError::Transport(err.to_string()))?;
    let mut result = resources
        .into_iter()
        .map(|resource| McpResourceInfo {
            uri: resource.uri,
            name: resource.name,
            title: resource.title,
            description: resource.description,
            template: false,
        })
        .collect::<Vec<_>>();
    result.extend(templates.into_iter().map(|resource| McpResourceInfo {
        uri: resource.uri_template,
        name: resource.name,
        title: resource.title,
        description: resource.description,
        template: true,
    }));
    result.sort_by(|left, right| left.uri.cmp(&right.uri));
    Ok(result)
}

pub(crate) async fn resolve_resources(
    ctx: &Context,
    resources: &[McpResourceRef],
) -> Result<Vec<ResolvedResource>, GraphError> {
    let mut resolved = Vec::with_capacity(resources.len());
    for resource in resources {
        resolved.push(resolve_resource(ctx, resource).await.map_err(|err| {
            GraphError::McpResource(format!("{}:{}: {err}", resource.server(), resource.uri()))
        })?);
    }
    Ok(resolved)
}

async fn resolve_resource(
    ctx: &Context,
    resource: &McpResourceRef,
) -> Result<ResolvedResource, McpError> {
    let server = ctx
        .mcp_server(resource.server())
        .ok_or_else(|| McpError::MissingServer(resource.server().into()))?;
    let uri = expand_uri(resource)?;
    let client = connect(server).await?;
    let response = client
        .read_resource(ReadResourceRequestParams::new(&uri))
        .await
        .map_err(|err| McpError::Transport(err.to_string()))?;
    let text = text_contents(response.contents)?;
    Ok(ResolvedResource {
        server: resource.server().into(),
        uri,
        text,
    })
}

async fn connect(
    server: &McpServer,
) -> Result<rmcp::service::RunningService<rmcp::RoleClient, ()>, McpError> {
    validate_server(server)?;
    let mut config = StreamableHttpClientTransportConfig::with_uri(server.url.clone());
    if let Some(token) = &server.bearer_token {
        config = config.auth_header(token.clone());
    }
    config = config.custom_headers(parse_headers(&server.headers)?);
    let transport = StreamableHttpClientTransport::from_config(config);
    ().serve(transport)
        .await
        .map_err(|err| McpError::Transport(err.to_string()))
}

fn validate_server(server: &McpServer) -> Result<(), McpError> {
    if server.id.trim().is_empty() {
        return Err(McpError::InvalidConfiguration(
            "server id must not be empty".into(),
        ));
    }
    if !(server.url.starts_with("http://") || server.url.starts_with("https://")) {
        return Err(McpError::InvalidConfiguration(
            "server URL must use http or https".into(),
        ));
    }
    Ok(())
}

fn parse_headers(
    headers: &BTreeMap<String, String>,
) -> Result<HashMap<HeaderName, HeaderValue>, McpError> {
    headers
        .iter()
        .map(|(name, value)| {
            let name = HeaderName::try_from(name).map_err(|err| {
                McpError::InvalidConfiguration(format!("invalid header name: {err}"))
            })?;
            let value = HeaderValue::try_from(value).map_err(|err| {
                McpError::InvalidConfiguration(format!("invalid header value: {err}"))
            })?;
            Ok((name, value))
        })
        .collect()
}

fn expand_uri(resource: &McpResourceRef) -> Result<String, McpError> {
    let mut uri = resource.uri().to_owned();
    for (name, value) in resource.arguments() {
        let pattern = format!("{{{name}}}");
        let encoded = utf8_percent_encode(value, NON_ALPHANUMERIC).to_string();
        uri = uri.replace(&pattern, &encoded);
    }
    if uri.contains('{') || uri.contains('}') {
        return Err(McpError::InvalidConfiguration(format!(
            "unresolved or unsupported URI template '{uri}'"
        )));
    }
    Ok(uri)
}

fn text_contents(contents: Vec<ResourceContents>) -> Result<String, McpError> {
    let mut text = Vec::new();
    for content in contents {
        match content {
            ResourceContents::TextResourceContents { text: value, .. } => text.push(value),
            ResourceContents::BlobResourceContents { uri, .. } => {
                return Err(McpError::UnsupportedContent(format!(
                    "resource '{uri}' returned blob content"
                )));
            }
            _ => {
                return Err(McpError::UnsupportedContent(
                    "resource returned an unknown content variant".into(),
                ));
            }
        }
    }
    if text.is_empty() {
        return Err(McpError::UnsupportedContent(
            "resource returned no text content".into(),
        ));
    }
    Ok(text.join("\n"))
}

#[cfg(test)]
mod tests {
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
    use crate::graph::{Agent, AgentConfig, Flow, Step, compile};
    use crate::testing::ScriptedFactory;

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
            "openai:///test-model",
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
    async fn streamable_http_catalog_and_resource_resolution_work() {
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
    }

    /// Verifies missing credentials fail and blob resources remain unsupported.
    #[tokio::test]
    async fn streamable_http_rejects_unauthorized_and_blob_resources() {
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
    }

    /// Verifies restored agents use checkpointed resource text without network access.
    #[tokio::test]
    async fn restored_agent_does_not_reread_mcp_resources() {
        let (url, cancellation) = spawn_resource_server().await;
        let flow = compile(resource_flow).unwrap();
        let mut runtime = flow
            .start(
                ResourceInput {
                    question: "question".into(),
                },
                test_context(url),
            )
            .unwrap();
        assert_eq!(runtime.next().await.unwrap(), Step::Continue);
        let snapshot = runtime.snapshot().unwrap();
        let encoded = serde_json::to_string(&snapshot).unwrap();
        assert!(encoded.contains("text for docs://a-first"));
        cancellation.cancel();

        let factory =
            ScriptedFactory::new().then_output(serde_json::json!({ "answer": "from checkpoint" }));
        let mut restored = flow
            .restore(snapshot, Context::default().with_client_factory(factory))
            .unwrap();
        let step = restored.next().await.unwrap();
        let Step::Done(value) = step else {
            panic!("restored agent should complete");
        };
        assert_eq!(
            flow.decode_output(value).unwrap(),
            ResourceOutput {
                answer: "from checkpoint".into()
            }
        );
    }
}
