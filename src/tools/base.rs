use std::future::Future;
use std::marker::PhantomData;
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

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
    /// A tool requesting external input before the flow can continue.
    /// Caught by the orchestrator, which surfaces the deserialized input as
    /// [`crate::flows::FlowStep::Suspend`] so the caller can inspect and downcast it,
    /// then call `resume()` with a value matching `output_type`.
    #[error("suspend signal from tool")]
    Suspend { value: SuspendedValue, output_type: String },
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
        let working_dir = normalize_path(self.working_dir());
        let path = Path::new(raw);
        let requested = if path.is_absolute() {
            normalize_path(path)
        } else {
            normalize_path(&working_dir.join(path))
        };
        if !requested.starts_with(&working_dir) {
            return Err(ToolError::PathEscape(raw.to_owned()));
        }
        let canonical_root = canonical_working_dir(&working_dir)?;
        let relative = requested
            .strip_prefix(&working_dir)
            .map_err(|_| ToolError::PathEscape(raw.to_owned()))?;
        resolve_within_root(raw, &canonical_root, relative)
    }
}

fn canonical_working_dir(path: &Path) -> Result<PathBuf, ToolError> {
    match std::fs::canonicalize(path) {
        Ok(canonical) => Ok(canonical),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(path.to_path_buf()),
        Err(error) => Err(ToolError::Io(error)),
    }
}

fn resolve_within_root(raw: &str, root: &Path, relative: &Path) -> Result<PathBuf, ToolError> {
    let mut resolved = root.to_path_buf();

    for component in relative.components() {
        match component {
            Component::CurDir => continue,
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(part) => {
                resolved.push(part);
                match std::fs::symlink_metadata(&resolved) {
                    Ok(meta) if meta.file_type().is_symlink() => {
                        let canonical = std::fs::canonicalize(&resolved)?;
                        if !canonical.starts_with(root) {
                            return Err(ToolError::PathEscape(raw.to_owned()));
                        }
                        resolved = canonical;
                    }
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(ToolError::Io(error)),
                }
            }
            Component::RootDir => {
                resolved = root.to_path_buf();
            }
            Component::Prefix(_) => {
                return Err(ToolError::PathEscape(raw.to_owned()));
            }
        }

        if !resolved.starts_with(root) {
            return Err(ToolError::PathEscape(raw.to_owned()));
        }
    }

    Ok(resolved)
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

/// Typed input from an LLM suspend-tool call surfaced as [`crate::flows::FlowStep::Suspend`].
///
/// Downcast to the concrete input type registered via [`ToolBox::suspend`] using
/// [`downcast`](Self::downcast) or [`downcast_ref`](Self::downcast_ref).
pub struct SuspendedValue(Box<dyn std::any::Any + Send>);

impl SuspendedValue {
    pub(crate) fn new<T: std::any::Any + Send + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }

    /// Attempts to downcast to `T`, consuming `self`. Returns `Err(self)` if the type doesn't match.
    pub fn downcast<T: 'static>(self) -> Result<T, Self> {
        self.0.downcast::<T>().map(|b| *b).map_err(SuspendedValue)
    }

    /// Borrows the inner value as `&T`, returning `None` if the type doesn't match.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

impl std::fmt::Debug for SuspendedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SuspendedValue(..)")
    }
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
    type Output: serde::Serialize + JsonSchema + DeserializeOwned + Send;

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

    /// Schema name of the type the LLM provides as args (the tool's input type).
    fn input_type(&self) -> String;
    fn output_type(&self) -> String;

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

    fn input_type(&self) -> String {
        T::schema_name().into()
    }

    fn output_type(&self) -> String {
        T::Output::schema_name().into()
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

/// Registers a tool type `T` — suspension tools are added via [`.suspend()`](ToolBox::suspend).
///
/// Build with [`ToolBox::new`]; dispatch with [`ToolBox::call`].
pub struct ToolBox {
    tools: Vec<Box<dyn ErasedTool>>,
    exit_name: String,
    /// Type-erased closures that inject `FlowNode::AgentTool` / `FlowNode::FlowTool` nodes
    /// into the outer graph at `FlowBuilder::agent` time. Stored here so `ToolBox` stays
    /// crate-agnostic; drained by `ToolBox::with_agent` in `commons.rs`.
    /// `Fn + Sync` so that `Arc<ToolBox>` is `Send + Sync`.
    pub(crate) graph_injectors: Vec<Box<dyn Fn(&str, &mut crate::flows::FlowGraph) + Send + Sync>>,
}

impl ToolBox {
    /// Creates an empty [`ToolBox`] with the default exit sentinel name (`"submit"`).
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            exit_name: "submit".to_owned(),
            graph_injectors: Vec::new(),
        }
    }

    /// Overrides the name of the auto-generated exit sentinel tool (default: `"submit"`).
    pub fn with_exit_name(mut self, name: impl Into<String>) -> Self {
        self.exit_name = name.into();
        self
    }

    /// Registers a tool type `T`. Call multiple times to add more tools.
    pub fn tool<T: Tool + 'static>(mut self) -> Self {
        self.tools.push(make_dispatcher::<T>());
        self
    }

    /// Registers a typed suspension point. The LLM calls this tool with a value of type `T`;
    /// the flow pauses and surfaces a [`SuspendedValue`] wrapping the deserialized `T` as
    /// [`crate::flows::FlowStep::Suspend`]. Call `resume()` with a value of type `Out`.
    pub fn suspend<T, Out>(mut self) -> Self
    where
        T: JsonSchema + serde::de::DeserializeOwned + Send + 'static,
        Out: JsonSchema + 'static,
    {
        let name = T::schema_name();
        let out_name = Out::schema_name();
        let parameters = serde_json::to_value(schemars::schema_for!(T))
            .unwrap_or_else(|_| Value::Object(Default::default()));
        let def = ToolDefinition {
            name: name.clone(),
            description: format!("Pause and await external fulfillment. Resume with `{out_name}`."),
            parameters,
        };
        let deserialize: Arc<dyn Fn(Value) -> Result<SuspendedValue, serde_json::Error> + Send + Sync> =
            Arc::new(|args| serde_json::from_value::<T>(args).map(SuspendedValue::new));
        self.tools.push(Box::new(SuspendTool { def, input_type: name.into(), output_type: out_name.into(), deserialize }));
        self
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

    /// Returns the number of registered tools.
    pub(crate) fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns the name of the tool at position `i`.
    ///
    /// # Panics
    /// Panics if `i >= len()`.
    pub(crate) fn name_at(&self, i: usize) -> &str {
        self.tools[i].name()
    }

    pub(crate) fn input_type_at(&self, i: usize) -> String {
        self.tools[i].input_type()
    }

    pub(crate) fn output_type_at(&self, i: usize) -> String {
        self.tools[i].output_type()
    }

    /// Invokes the tool at position `i` by its slot index.
    pub(crate) async fn call_at_index(
        &self,
        i: usize,
        args: Value,
        ctx: Context,
    ) -> Result<Value, ToolError> {
        self.tools[i].call_raw(ctx, args).await
    }

    /// Appends a pre-boxed tool. `pub(crate)` so only `commons` can inject the sentinel.
    pub(crate) fn push_erased(&mut self, tool: Box<dyn ErasedTool>) {
        self.tools.push(tool);
    }

    /// Dispatches `tool_call` to the matching tool, using `ctx` for execution.
    pub async fn call(&self, tool_call: &ToolCall, ctx: Context) -> Result<Value, ToolError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == tool_call.name)
            .ok_or_else(|| ToolError::UnknownTool(tool_call.name.clone()))?;
        tool.call_raw(ctx, tool_call.args.clone()).await
    }
}

/// Hidden tool that returns `ToolError::Suspend` — registered via [`ToolBox::suspend`].
struct SuspendTool {
    def: ToolDefinition,
    input_type: String,
    output_type: String,
    deserialize: Arc<dyn Fn(Value) -> Result<SuspendedValue, serde_json::Error> + Send + Sync>,
}

impl ErasedTool for SuspendTool {
    fn name(&self) -> &str {
        &self.def.name
    }

    fn definition(&self) -> ToolDefinition {
        self.def.clone()
    }

    fn input_type(&self) -> String {
        self.input_type.clone()
    }

    fn output_type(&self) -> String {
        self.output_type.clone()
    }

    fn call_raw<'a>(
        &'a self,
        _ctx: Context,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'a>> {
        let deserialize = self.deserialize.clone();
        let output_type = self.output_type.clone();
        Box::pin(async move {
            let value = (deserialize)(args).map_err(ToolError::Deserialize)?;
            Err(ToolError::Suspend { value, output_type })
        })
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
        let tb = ToolBox::new()
            .tool::<ReadFile>()
            .tool::<WriteFile>();
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

        let tb = ToolBox::new().tool::<ReadFile>();
        let tc = ToolCall {
            id: "1".into(),
            name: "read_file".into(),
            args: json!({ "path": path }),
            thought_signatures: None,
        };
        let result = tb
            .call(&tc, ctx(tmp.path().parent().unwrap()))
            .await
            .unwrap();
        assert_eq!(result["content"], "hello toolbox");
    }

    /// Verifies that calling an unregistered tool name returns `ToolError::UnknownTool`.
    #[tokio::test]
    async fn toolbox_unknown_tool_returns_error() {
        let tb = ToolBox::new().tool::<ReadFile>();
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
