use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::deps::{Deps, DepsError};

#[derive(Clone)]
struct ContextInner {
    working_dir: PathBuf,
    commands: Vec<String>,
    deps: Deps,
    http_client: Option<reqwest::Client>,
}

/// Shared execution context threaded through every tool call and flow step.
///
/// Cheap to clone — the inner state is reference-counted via [`Arc`].
/// Build with [`Context::new`], then chain `.with_*` methods as needed.
#[derive(Clone)]
pub struct Context(Arc<ContextInner>);

impl Context {
    /// Creates a context rooted at `working_dir` with no commands, no deps, and no HTTP client.
    pub fn new(working_dir: PathBuf) -> Self {
        Self(Arc::new(ContextInner {
            working_dir,
            commands: Vec::new(),
            deps: Deps::default(),
            http_client: None,
        }))
    }

    /// Replaces the command allowlist. Returns `self` for chaining.
    pub fn with_commands(self, commands: Vec<String>) -> Self {
        let inner = Arc::unwrap_or_clone(self.0);
        Self(Arc::new(ContextInner { commands, ..inner }))
    }

    /// Replaces the dependency container. Returns `self` for chaining.
    pub fn with_deps(self, deps: Deps) -> Self {
        let inner = Arc::unwrap_or_clone(self.0);
        Self(Arc::new(ContextInner { deps, ..inner }))
    }

    /// Installs a shared HTTP client. Returns `self` for chaining.
    pub fn with_http_client(self, http_client: reqwest::Client) -> Self {
        let inner = Arc::unwrap_or_clone(self.0);
        Self(Arc::new(ContextInner {
            http_client: Some(http_client),
            ..inner
        }))
    }

    /// Root directory all relative paths are resolved against.
    pub fn working_dir(&self) -> &Path {
        &self.0.working_dir
    }

    /// Allowlist of command names tools may execute.
    pub fn commands(&self) -> &[String] {
        &self.0.commands
    }

    /// Dependency container for optional services (e.g. search engine).
    pub fn deps(&self) -> &Deps {
        &self.0.deps
    }

    /// Returns the shared HTTP client, or builds a default with a 30-second timeout.
    pub fn http_client(&self) -> reqwest::Client {
        self.0.http_client.clone().unwrap_or_else(|| {
            reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default()
        })
    }

    /// Retrieves a required service from `deps` by type.
    ///
    /// Returns [`DepsError::MissingDependency`] if `T` has not been registered.
    pub fn require<T: std::any::Any + Send + Sync + 'static>(&self) -> Result<&T, DepsError> {
        self.0.deps.require::<T>()
    }
}
