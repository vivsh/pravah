use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::deps::{Deps, DepsError};

/// Settings used to build a [`Context`].
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FlowConf {
    /// Base directory for relative paths.
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
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(std::env::temp_dir);
        Self(Arc::new(ContextInner {
            working_dir,
            commands: conf.commands,
            deps: Deps::default(),
            http_client: None,
            http_timeout_secs: conf.http_timeout_secs.unwrap_or(30),
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

    /// Base directory for relative paths.
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
}

