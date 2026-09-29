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
#[path = "tests/mcp.rs"]
mod tests;
