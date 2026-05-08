use std::future::Future;
use std::marker::PhantomData;
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

use crate::clients::ToolCall;
use crate::context::Context;
use crate::deps::DepsError;

/// Error produced when a tool invocation fails.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to serialize output: {0}")]
    Serialize(serde_json::Error),
    #[error("failed to deserialize input: {0}")]
    Deserialize(serde_json::Error),
    #[error("unknown tool '{0}")]
    UnknownTool(String),
    #[error("path '{0}' escapes the working directory")]
    PathEscape(String),
    #[error("command '{0}' is not in the allowed list")]
    ForbiddenCommand(String),
    #[error("http error: {0}")]
    Http(String),
    #[error("{0}")]
    Missing(#[from] DepsError),
    #[error("{0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// Auto-generated `submit` sentinel signalling a flow state transition.
    /// Caught by the orchestrator before reaching history; never propagates to user code.
    #[doc(hidden)]
    #[error("exit signal from tool")]
    Exit(serde_json::Value),
    /// A tool requesting user input before the flow can continue.
    /// Produced by [`ToolError::suspend`]; caught by the orchestrator, which transitions
    /// to the matching resume variant and returns [`crate::flows::FlowError::Suspended`].
    #[error("suspend signal from tool")]
    Suspend(serde_json::Value),
}

impl ToolError {
    /// Serializes `val` and wraps it as a suspend signal.
    ///
    /// On serialization failure, returns [`ToolError::Serialize`] instead of panicking.
    pub fn suspend<T: serde::Serialize + 'static>(val: T) -> Self {
        match serde_json::to_value(&val) {
            Ok(value) => ToolError::Suspend(value),
            Err(e) => ToolError::Serialize(e),
        }
    }
}

impl Context {
    /// Returns `Ok(())` if `cmd` appears in the `commands` allowlist.
    pub fn check_command(&self, cmd: &str) -> Result<(), ToolError> {
        if self.commands().iter().any(|c| c == cmd) {
            Ok(())
        } else {
            Err(ToolError::ForbiddenCommand(cmd.to_owned()))
        }
    }

    /// Resolves `raw` to an absolute path and verifies it stays within `working_dir`.
    ///
    /// Relative paths are resolved against `working_dir`. `..` components are
    /// collapsed without hitting the filesystem, so this check is safe for
    /// paths that do not yet exist (e.g. write targets).
    pub fn resolve(&self, raw: &str) -> Result<PathBuf, ToolError> {
        let path = Path::new(raw);
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.working_dir().join(path)
        };
        let normalized = normalize_path(&abs);
        let wd = normalize_path(self.working_dir());
        if !normalized.starts_with(&wd) {
            return Err(ToolError::PathEscape(raw.to_owned()));
        }
        Ok(normalized)
    }
}

/// Collapses `.` and `..` components without touching the filesystem.
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

/// The output of a single tool invocation.
#[derive(Debug)]
pub struct ToolOutput {
    pub call: ToolCall,
    pub value: serde_json::Value,
}

/// Metadata the orchestrator sends to the LLM to advertise a tool.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema `object` describing the tool's input shape.
    pub parameters: Value,
}

/// Typed tool trait where `Self` is both the tool and its deserialized input.
///
/// Implement this trait on a struct that derives [`serde::Deserialize`] and
/// [`JsonSchema`]. The struct's fields become the LLM-callable arguments.
/// [`ToolDefinition`] is derived automatically via [`Tool::definition`].
pub trait Tool: DeserializeOwned + JsonSchema + Sized + Send {
    /// Typed output this tool produces. Must be `Serialize` so `ErasedTool` can
    /// convert it to `serde_json::Value` after dispatch without the caller
    /// building raw JSON.
    type Output: serde::Serialize + Send;

    fn name() -> &'static str;

    fn description() -> &'static str;

    /// Execute the tool, consuming `self` (the parsed input).
    fn call(self, ctx: Context) -> impl Future<Output = Result<Self::Output, ToolError>> + Send;

    /// Derives a [`ToolDefinition`] from this tool's metadata and input schema.
    fn definition() -> ToolDefinition {
        let parameters = serde_json::to_value(schemars::schema_for!(Self))
            .unwrap_or_else(|e| {
                tracing::error!(tool = Self::name(), error = %e, "tool schema serialization failed; parameters will be empty");
                Value::Object(Default::default())
            });
        ToolDefinition {
            name: Self::name().to_owned(),
            description: Self::description().to_owned(),
            parameters,
        }
    }
}

/// Creates a heap-allocated type-erased dispatcher for tool type `T`.
/// Crate-internal; used by [`ToolBoxBuilder::tool`] and `commons`.
pub(crate) fn make_dispatcher<T: Tool + 'static>() -> Box<dyn ErasedTool> {
    Box::new(ToolDispatcher::<T>(PhantomData))
}

/// Object-safe wrapper around [`Tool`] for use in heterogeneous collections.
///
/// Do not implement this directly — use the blanket impl via [`Tool`].
pub(crate) trait ErasedTool: Send + Sync {
    fn name(&self) -> &str;
    fn definition(&self) -> ToolDefinition;
    /// Deserializes `args` into the concrete tool type, calls it, returns the output.
    fn call_raw<'a>(
        &'a self,
        ctx: Context,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ToolError>> + Send + 'a>>;
}

/// Zero-sized adapter that makes any [`Tool`] object-safe as [`ErasedTool`].
struct ToolDispatcher<T>(PhantomData<fn() -> T>);

impl<T: Tool + 'static> ErasedTool for ToolDispatcher<T> {
    fn name(&self) -> &str {
        T::name()
    }

    fn definition(&self) -> ToolDefinition {
        T::definition()
    }

    fn call_raw<'a>(
        &'a self,
        ctx: Context,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let input: T = serde_json::from_value(args).map_err(ToolError::Deserialize)?;
            let output = input.call(ctx).await?;
            serde_json::to_value(output).map_err(ToolError::Serialize)
        })
    }
}

/// Stateless registry of tools. Shareable across agents via `Arc<ToolBox>`.
///
/// Build with [`ToolBox::builder`]; dispatch with [`ToolBox::call`].
pub struct ToolBox {
    tools: Vec<Box<dyn ErasedTool>>,
    exit_name: String,
}

impl ToolBox {
    /// Starts building a new [`ToolBox`].
    pub fn builder() -> ToolBoxBuilder {
        ToolBoxBuilder {
            tools: Vec::new(),
            exit_name: "submit".to_owned(),
        }
    }

    /// Returns the name used for the auto-generated exit sentinel tool.
    pub fn exit_name(&self) -> &str {
        &self.exit_name
    }

    /// Returns the [`ToolDefinition`] for every registered tool.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    /// Returns `true` if no tools have been registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Appends a pre-boxed tool. `pub(crate)` so only `commons` can inject the sentinel.
    pub(crate) fn push_erased(&mut self, tool: Box<dyn ErasedTool>) {
        self.tools.push(tool);
    }

    /// Dispatches `tool_call` to the matching tool, using `ctx` for execution.
    pub async fn call(&self, tool_call: &ToolCall, ctx: Context) -> Result<ToolOutput, ToolError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == tool_call.name)
            .ok_or_else(|| ToolError::UnknownTool(tool_call.name.clone()))?;
        let call = tool_call.clone();
        tool.call_raw(ctx, tool_call.args.clone())
            .await
            .map(|o| ToolOutput { call, value: o })
    }
}

/// Builder for [`ToolBox`].
pub struct ToolBoxBuilder {
    tools: Vec<Box<dyn ErasedTool>>,
    exit_name: String,
}

impl ToolBoxBuilder {
    /// Overrides the name of the auto-generated exit sentinel tool (default: `"submit"`).
    pub fn with_exit_name(mut self, name: impl Into<String>) -> Self {
        self.exit_name = name.into();
        self
    }

    /// Returns the current exit sentinel tool name.
    pub fn exit_name(&self) -> &str {
        &self.exit_name
    }

    /// Registers a tool type `T`. Call multiple times to add more tools.
    pub fn tool<T: Tool + 'static>(mut self) -> Self {
        self.tools.push(make_dispatcher::<T>());
        self
    }

    /// Finishes the builder and returns the registered toolbox.
    pub fn build(self) -> ToolBox {
        ToolBox {
            tools: self.tools,
            exit_name: self.exit_name,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serde_json::json;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::context::FlowConf;
    use crate::tools::fs::{ReadFile, WriteFile};

    fn ctx(dir: &std::path::Path) -> Context {
        Context::new(FlowConf {
            working_dir: Some(dir.to_path_buf()),
            ..Default::default()
        })
    }

    /// Verifies that all registered tool definitions are collected with correct names.
    #[test]
    fn toolbox_collects_definitions() {
        let tb = ToolBox::builder()
            .tool::<ReadFile>()
            .tool::<WriteFile>()
            .build();
        let defs = tb.definitions();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "read_file");
        assert_eq!(defs[1].name, "write_file");
    }

    /// Verifies that `call` dispatches to the correct tool and returns its output.
    #[tokio::test]
    async fn toolbox_dispatches_read_file() {
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "hello toolbox").unwrap();
        let path = tmp.path().to_string_lossy().into_owned();

        let tb = ToolBox::builder().tool::<ReadFile>().build();
        let tc = ToolCall {
            id: "1".into(),
            name: "read_file".into(),
            args: json!({ "path": path }),
            thought_signatures: None,
        };
        let output = tb
            .call(&tc, ctx(tmp.path().parent().unwrap()))
            .await
            .unwrap();
        let result = output.value;
        assert_eq!(result["content"], "hello toolbox");
    }

    /// Verifies that calling an unregistered tool name returns `ToolError::UnknownTool`.
    #[tokio::test]
    async fn toolbox_unknown_tool_returns_error() {
        let tb = ToolBox::builder().tool::<ReadFile>().build();
        let tc = ToolCall {
            id: "x".into(),
            name: "no_such_tool".into(),
            args: json!({}),
            thought_signatures: None,
        };
        let err = tb
            .call(&tc, ctx(std::path::Path::new("/tmp")))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::UnknownTool(n) if n == "no_such_tool"));
    }
}
