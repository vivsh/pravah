use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "mcp")]
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::clients::{ClientFactory, DefaultClientFactory};
use crate::deps::{Deps, DepsError};
#[cfg(feature = "mcp")]
use crate::graph::{McpError, McpResourceInfo, McpServer};

/// Settings used to build a [`Context`].
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FlowConf {
    /// Base directory for relative-path resolution and path-escape checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<PathBuf>,
    /// Commands tools may execute.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<String>,
    /// HTTP timeout in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_timeout_secs: Option<u64>,
}

#[derive(Clone)]
struct ContextInner {
    working_dir: PathBuf,
    commands: Vec<String>,
    deps: Deps,
    http_client: Option<reqwest::Client>,
    http_timeout_secs: u64,
    client_factory: Arc<dyn ClientFactory>,
    #[cfg(feature = "mcp")]
    mcp_servers: Arc<BTreeMap<String, McpServer>>,
}

/// Shared execution context passed to every step and tool.
/// Cloning is cheap because the inner state is reference-counted.
#[derive(Clone)]
pub struct Context(Arc<ContextInner>);

impl Default for Context {
    fn default() -> Self {
        Self::new(FlowConf::default())
    }
}

impl Context {
    /// Builds a context from [`FlowConf`].
    pub fn new(conf: FlowConf) -> Self {
        let working_dir = conf
            .working_dir
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::env::temp_dir()));
        Self(Arc::new(ContextInner {
            working_dir,
            commands: conf.commands,
            deps: Deps::default(),
            http_client: None,
            http_timeout_secs: conf.http_timeout_secs.unwrap_or(30),
            client_factory: Arc::new(DefaultClientFactory),
            #[cfg(feature = "mcp")]
            mcp_servers: Arc::new(BTreeMap::new()),
        }))
    }

    /// Replaces the command allowlist.
    pub fn with_commands(self, commands: Vec<String>) -> Self {
        let inner = Arc::unwrap_or_clone(self.0);
        Self(Arc::new(ContextInner { commands, ..inner }))
    }

    /// Replaces the dependency container.
    pub fn with_deps(self, deps: Deps) -> Self {
        let inner = Arc::unwrap_or_clone(self.0);
        Self(Arc::new(ContextInner { deps, ..inner }))
    }

    /// Installs a shared HTTP client.
    pub fn with_http_client(self, http_client: reqwest::Client) -> Self {
        let inner = Arc::unwrap_or_clone(self.0);
        Self(Arc::new(ContextInner {
            http_client: Some(http_client),
            ..inner
        }))
    }

    /// Replaces the LLM client factory used by graph agents in this context.
    ///
    /// The factory is runtime-only and must be installed again after restoring
    /// a serialized workflow snapshot.
    pub fn with_client_factory(self, client_factory: impl ClientFactory + 'static) -> Self {
        let inner = Arc::unwrap_or_clone(self.0);
        Self(Arc::new(ContextInner {
            client_factory: Arc::new(client_factory),
            ..inner
        }))
    }

    /// Registers one runtime-only Streamable HTTP MCP resource server.
    #[cfg(feature = "mcp")]
    pub fn with_mcp_server(self, server: McpServer) -> Self {
        let inner = Arc::unwrap_or_clone(self.0);
        let mut servers = (*inner.mcp_servers).clone();
        servers.insert(server.id().to_owned(), server);
        Self(Arc::new(ContextInner {
            mcp_servers: Arc::new(servers),
            ..inner
        }))
    }

    /// Lists concrete resources and templates from a configured MCP server.
    #[cfg(feature = "mcp")]
    pub async fn mcp_resources(&self, server: &str) -> Result<Vec<McpResourceInfo>, McpError> {
        crate::graph::mcp::list_resources(self, server).await
    }

    /// Base directory for relative-path resolution and path-escape checks.
    pub fn working_dir(&self) -> &Path {
        &self.0.working_dir
    }

    /// Command allowlist.
    pub fn commands(&self) -> &[String] {
        &self.0.commands
    }

    /// Registered shared services.
    pub fn deps(&self) -> &Deps {
        &self.0.deps
    }

    /// Returns the shared HTTP client.
    /// Builds a default client on demand when none was installed.
    pub fn http_client(&self) -> reqwest::Client {
        self.0.http_client.clone().unwrap_or_else(|| {
            reqwest::Client::builder()
                .timeout(Duration::from_secs(self.0.http_timeout_secs))
                .build()
                .unwrap_or_default()
        })
    }

    /// Retrieves a required dependency by type.
    /// Returns [`DepsError::MissingDependency`] when the type was not registered.
    pub fn require<T: std::any::Any + Send + Sync + 'static>(&self) -> Result<&T, DepsError> {
        self.0.deps.require::<T>()
    }

    pub(crate) fn client_factory(&self) -> &dyn ClientFactory {
        self.0.client_factory.as_ref()
    }

    #[cfg(feature = "mcp")]
    pub(crate) fn mcp_server(&self, id: &str) -> Option<&McpServer> {
        self.0.mcp_servers.get(id)
    }
}

#[cfg(test)]
mod tests {
    use crate::clients::ClientOptions;

    use super::*;

    /// Verifies graph contexts provide Rath's default client factory without setup.
    #[test]
    fn context_installs_default_graph_client_factory() {
        let context = Context::default();
        let client = context
            .client_factory()
            .create("ollama:///qwen3:8b", ClientOptions::default())
            .expect("default client factory should create an Ollama client");

        assert_eq!(client.model_url().model, "qwen3:8b");
    }
}
