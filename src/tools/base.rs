use std::future::Future;
use std::marker::PhantomData;
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

use crate::clients::{Attachment, ToolCall};
use crate::context::Context;
use crate::deps::DepsError;

/// Error returned by tool execution.
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
}

impl ToolError {
    /// Returns `true` when the error should abort the flow instead of going back to the model.
    pub fn is_fatal(&self) -> bool {
        matches!(self, ToolError::PathEscape(_) | ToolError::ForbiddenCommand(_))
    }
}

/// Output returned by a [`Tool`] implementation.
/// Use [`ToolOutput::plain`] for tools that return only JSON data.
/// Use [`ToolOutput::with_attachment`] or set `attachments` directly when the
/// tool also produces binary data (images, audio, etc.) that the model should see.
pub struct ToolOutput<T: serde::Serialize> {
    /// The structured data output, serialized as JSON and returned to the LLM.
    pub data: T,
    /// Binary or URL attachments propagated alongside the tool result.
    pub attachments: Vec<Attachment>,
}

impl<T: serde::Serialize> ToolOutput<T> {
    /// Creates a plain output with no attachments.
    pub fn plain(data: T) -> Self {
        Self { data, attachments: Vec::new() }
    }

    /// Creates an output with a single attachment.
    pub fn with_attachment(data: T, attachment: Attachment) -> Self {
        Self { data, attachments: vec![attachment] }
    }
}

impl<T: serde::Serialize> From<T> for ToolOutput<T> {
    fn from(data: T) -> Self {
        Self::plain(data)
    }
}

/// Internal tool dispatch outcome.
/// `Exit` and `Suspend` are control signals, not user-facing errors.
pub(crate) enum ToolOutcome {
    /// Normal tool result plus any binary attachments the tool produced.
    Value(Value, Vec<Attachment>),
    /// Exit sentinel result carrying the agent's final output.
    Exit(Value),
    /// Suspend tool result carrying the pending value and expected resume type.
    Suspend { value: SuspendedValue, output_type: String },
}

impl Context {
    /// Rejects commands outside the configured allowlist.
    pub fn check_command(&self, cmd: &str) -> Result<(), ToolError> {
        if self.commands().iter().any(|c| c == cmd) {
            Ok(())
        } else {
            Err(ToolError::ForbiddenCommand(cmd.to_owned()))
        }
    }

    /// Resolves a path and rejects escapes from `working_dir`.
    /// Relative paths are normalized without requiring the target to exist.
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

/// Collapses `.` and `..` without touching the filesystem.
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

/// Type-erased input captured from a suspend tool call.
/// Downcast it with [`downcast`](Self::downcast) or [`downcast_ref`](Self::downcast_ref).
pub struct SuspendedValue(Box<dyn std::any::Any + Send>);

impl SuspendedValue {
    pub(crate) fn new<T: std::any::Any + Send + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }

    /// Attempts to downcast to `T`.
    /// Returns `Err(self)` when the type does not match.
    pub fn downcast<T: 'static>(self) -> Result<T, Self> {
        self.0.downcast::<T>().map(|b| *b).map_err(SuspendedValue)
    }

    /// Borrows the inner value as `&T`.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

impl std::fmt::Debug for SuspendedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SuspendedValue(..)")
    }
}

/// Tool metadata exposed to the model.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool input.
    pub parameters: Value,
}

/// Typed tool trait.
/// Implement it on the struct that represents the tool input.
/// The tool name defaults to the snake_case form of the struct name.
/// Override with `#[schemars(rename = "my_name")]`.
/// The tool description is taken from the struct's doc comment.
pub trait Tool: DeserializeOwned + JsonSchema + Sized + Send {
    /// Typed output produced by the tool.
    type Output: serde::Serialize + JsonSchema + DeserializeOwned + Send;

    /// Executes the tool.
    fn call(self, ctx: Context) -> impl Future<Output = Result<ToolOutput<Self::Output>, ToolError>> + Send;

    /// Builds the advertised [`ToolDefinition`].
    fn definition() -> ToolDefinition {
        let name = pascal_to_snake(Self::schema_name().as_ref());
        let schema = schemars::schema_for!(Self);
        let description = schema
            .schema
            .metadata
            .as_ref()
            .and_then(|m| m.description.as_deref())
            .unwrap_or_default()
            .to_owned();
        let parameters = serde_json::to_value(schema).unwrap_or_else(|e| {
            tracing::error!(tool = %name, error = %e, "tool schema serialization failed; parameters will be empty");
            Value::Object(Default::default())
        });
        ToolDefinition { name, description, parameters }
    }
}

/// Converts a PascalCase (or camelCase) identifier to snake_case.
/// Handles acronym runs: `HTTPRequest` → `http_request`.
fn pascal_to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_uppercase() && i > 0 {
            let prev = chars[i - 1];
            let next = chars.get(i + 1).copied();
            if prev.is_lowercase()
                || prev.is_ascii_digit()
                || (prev.is_uppercase() && next.map_or(false, |n| n.is_lowercase()))
            {
                out.push('_');
            }
        }
        out.extend(c.to_lowercase());
    }
    out
}

/// Creates a type-erased dispatcher for tool type `T`.
pub(crate) fn make_dispatcher<T: Tool + 'static>() -> Box<dyn ErasedTool> {
    Box::new(ToolDispatcher::<T> {
        name: pascal_to_snake(T::schema_name().as_ref()),
        _phantom: PhantomData,
    })
}

/// Object-safe wrapper around [`Tool`] for heterogeneous collections.
pub(crate) trait ErasedTool: Send + Sync {
    fn name(&self) -> &str;
    fn definition(&self) -> ToolDefinition;

    /// Schema name of the tool input type.
    fn input_type(&self) -> String;
    fn output_type(&self) -> String;

    /// Returns `false` when graph injection already registered the tool node.
    fn needs_tool_node(&self) -> bool {
        true
    }

    /// Deserializes `args`, calls the concrete tool, and returns the outcome.
    fn call_raw<'a>(
        &'a self,
        ctx: Context,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutcome, ToolError>> + Send + 'a>>;
}

/// Adapter that exposes any [`Tool`] as an [`ErasedTool`], caching the derived name.
struct ToolDispatcher<T> {
    name: String,
    _phantom: PhantomData<fn() -> T>,
}

impl<T: Tool + 'static> ErasedTool for ToolDispatcher<T> {
    fn name(&self) -> &str {
        &self.name
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
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutcome, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let input: T = serde_json::from_value(args).map_err(ToolError::Deserialize)?;
            let output = input.call(ctx).await?;
            let value = serde_json::to_value(output.data).map_err(ToolError::Serialize)?;
            Ok(ToolOutcome::Value(value, output.attachments))
        })
    }
}

/// Registry of tools exposed to an agent.
pub struct ToolBox {
    tools: Vec<Box<dyn ErasedTool>>,
    exit_name: String,
    /// Closures that inject flow-backed and agent-backed tool nodes during agent registration.
    pub(crate) graph_injectors: Vec<Box<dyn Fn(&str, &mut crate::flows::FlowGraph) + Send + Sync>>,
}

impl ToolBox {
    /// Creates an empty toolbox.
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            exit_name: "submit".to_owned(),
            graph_injectors: Vec::new(),
        }
    }

    /// Renames the auto-generated exit sentinel tool.
    pub fn with_exit_name(mut self, name: impl Into<String>) -> Self {
        self.exit_name = name.into();
        self
    }

    /// Registers a tool type.
    pub fn tool<T: Tool + 'static>(mut self) -> Self {
        self.tools.push(make_dispatcher::<T>());
        self
    }

    /// Registers a typed suspension point.
    /// `resume()` must later supply a value of type `Out`.
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

    /// Returns the exit sentinel name.
    pub fn exit_name(&self) -> &str {
        &self.exit_name
    }

    /// Returns all advertised tool definitions.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    /// Returns `true` when no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Returns the number of registered tools.
    pub(crate) fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns the tool name at position `i`.
    /// Panics when `i` is out of bounds.
    pub(crate) fn name_at(&self, i: usize) -> &str {
        self.tools[i].name()
    }

    pub(crate) fn input_type_at(&self, i: usize) -> String {
        self.tools[i].input_type()
    }

    pub(crate) fn output_type_at(&self, i: usize) -> String {
        self.tools[i].output_type()
    }

    /// Returns `false` when graph injection already registered the tool node.
    pub(crate) fn needs_tool_node_at(&self, i: usize) -> bool {
        self.tools[i].needs_tool_node()
    }

    /// Invokes the tool at position `i`.
    pub(crate) async fn call_at_index(
        &self,
        i: usize,
        args: Value,
        ctx: Context,
    ) -> Result<ToolOutcome, ToolError> {
        self.tools[i].call_raw(ctx, args).await
    }

    /// Appends a pre-boxed tool.
    pub(crate) fn push_erased(&mut self, tool: Box<dyn ErasedTool>) {
        self.tools.push(tool);
    }

    /// Dispatches a tool call by name.
    pub async fn call(&self, tool_call: &ToolCall, ctx: Context) -> Result<Value, ToolError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == tool_call.name)
            .ok_or_else(|| ToolError::UnknownTool(tool_call.name.clone()))?;
        match tool.call_raw(ctx, tool_call.args.clone()).await? {
            ToolOutcome::Value(v, _) => Ok(v),
            ToolOutcome::Exit(_) | ToolOutcome::Suspend { .. } => {
                unreachable!("internal sentinel tools must not be dispatched via ToolBox::call")
            }
        }
    }
}

/// Internal suspend tool registered by [`ToolBox::suspend`].
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
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutcome, ToolError>> + Send + 'a>> {
        let deserialize = self.deserialize.clone();
        let output_type = self.output_type.clone();
        Box::pin(async move {
            let value = (deserialize)(args).map_err(ToolError::Deserialize)?;
            Ok(ToolOutcome::Suspend { value, output_type })
        })
    }
}



#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::context::FlowConf;

    fn ctx() -> Context {
        Context::new(FlowConf::default())
    }

    /// Simple inline tool used only in these unit tests.
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[schemars(rename = "greet")]
    struct Greet {
        name: String,
    }

    #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    struct GreetOutput {
        message: String,
    }

    impl Tool for Greet {
        type Output = GreetOutput;
        async fn call(self, _ctx: Context) -> Result<ToolOutput<Self::Output>, ToolError> {
            Ok(ToolOutput::plain(GreetOutput { message: format!("hello {}", self.name) }))
        }
    }

    /// `ToolBox::definitions` collects tool definitions with derived names.
    #[test]
    fn toolbox_collects_definitions() {
        let tb = ToolBox::new().tool::<Greet>();
        let defs = tb.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "greet");
    }

    /// `ToolBox::call` dispatches by name and returns the serialised output.
    #[tokio::test]
    async fn toolbox_dispatches_tool() {
        let tb = ToolBox::new().tool::<Greet>();
        let tc = ToolCall {
            id: "1".into(),
            name: "greet".into(),
            args: json!({ "name": "world" }),
            thought_signatures: None,
        };
        let result = tb.call(&tc, ctx()).await.unwrap();
        assert_eq!(result["message"], "hello world");
    }

    /// Calling an unregistered tool name returns `ToolError::UnknownTool`.
    #[tokio::test]
    async fn toolbox_unknown_tool_returns_error() {
        let tb = ToolBox::new().tool::<Greet>();
        let tc = ToolCall {
            id: "x".into(),
            name: "no_such_tool".into(),
            args: json!({}),
            thought_signatures: None,
        };
        let err = tb.call(&tc, ctx()).await.unwrap_err();
        assert!(matches!(err, ToolError::UnknownTool(n) if n == "no_such_tool"));
    }

    /// `pascal_to_snake` converts common PascalCase patterns correctly.
    #[test]
    fn pascal_to_snake_converts_correctly() {
        assert_eq!(pascal_to_snake("ReadFile"), "read_file");
        assert_eq!(pascal_to_snake("RunCommand"), "run_command");
        assert_eq!(pascal_to_snake("HTTPRequest"), "http_request");
        assert_eq!(pascal_to_snake("MultiPatchFile"), "multi_patch_file");
        assert_eq!(pascal_to_snake("already_snake"), "already_snake");
        assert_eq!(pascal_to_snake("Broken2"), "broken2");
    }
}
