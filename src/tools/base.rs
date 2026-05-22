use std::future::Future;
use std::path::{Component, Path, PathBuf};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

use serde::Serialize;

use crate::clients::Message;
use crate::context::Context;
use crate::deps::DepsError;

/// Error returned by tool execution.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool or resource not found.
    #[error("not found: {0}")]
    NotFound(String),
    /// The model passed a value with the wrong JSON shape.
    #[error("type error in tool input: {0}")]
    TypeError(serde_json::Error),
    /// Argument or constraint violation the model can correct.
    #[error("{0}")]
    Validation(String),
    /// Path escape or forbidden command attempt.
    #[error("security violation: {0}")]
    Security(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("http error: {0}")]
    Http(String),
    #[error("failed to serialize output: {0}")]
    Serialize(serde_json::Error),
    #[error("{0}")]
    Other(String),
    /// Aborts the flow immediately; the model cannot recover from this.
    #[error("{0}")]
    Fatal(String),
}

impl ToolError {
    /// Returns `true` only for [`ToolError::Fatal`], which aborts the flow.
    /// All other variants are serialized as structured JSON and sent back to the model.
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::Fatal(_))
    }

    /// Short identifier for the variant, used in the `error_kind` field of the JSON response.
    pub fn error_kind(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "NotFound",
            Self::TypeError(_) => "TypeError",
            Self::Validation(_) => "Validation",
            Self::Security(_) => "Security",
            Self::Io(_) => "Io",
            Self::Http(_) => "Http",
            Self::Serialize(_) => "Serialize",
            Self::Other(_) => "Other",
            Self::Fatal(_) => "Fatal",
        }
    }

    /// Serializes this error as a structured JSON string for use as a tool-result payload.
    ///
    /// `tool_name` is the registered name of the tool (e.g. `"ReadFile"`).
    pub fn to_json(&self, tool_name: &str) -> String {
        serde_json::json!({
            "tool": tool_name,
            "ok": false,
            "error_kind": self.error_kind(),
            "message": self.to_string(),
            "recoverable": true,
        })
        .to_string()
    }

    /// Converts this error into a tool-result [`Message`] that is sent back to the model.
    ///
    /// Only call this for non-fatal errors; fatal errors should abort the flow via [`FlowError`].
    pub fn into_error_message(self, tool_name: &str) -> Message {
        Message::tool_output(String::new(), self.to_json(tool_name))
    }
}

impl From<DepsError> for ToolError {
    fn from(e: DepsError) -> Self {
        ToolError::Fatal(e.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for ToolError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        ToolError::Other(e.to_string())
    }
}

/// Constrains what types may be used as tool output.
///
/// The default implementation serializes `self` as JSON. Override `to_message`
/// to attach binary data or produce a custom text payload.
pub trait ToolOutput: Serialize + DeserializeOwned + JsonSchema + Send + 'static {
    fn to_message(self) -> Result<Message, ToolError> {
        let content = serde_json::to_string(&self).map_err(ToolError::Serialize)?;
        Ok(Message::tool_output(String::new(), content))
    }
}

/// Stateless tool that can be registered with `FlowBuilder::tool`.
pub trait Tool {
    type Input: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    type Output: Serialize + DeserializeOwned + JsonSchema + Send + 'static;

    /// Converts a tool output value into a [`Message`] sent back to the model.
    ///
    /// The default implementation serializes `output` as JSON. Override to attach
    /// binary data or produce a custom text payload.
    fn to_message(output: Self::Output) -> Result<Message, ToolError> {
        let content = serde_json::to_string(&output).map_err(ToolError::Serialize)?;
        Ok(Message::tool_output(String::new(), content))
    }

    fn call(input: Self::Input, ctx: Context) -> impl Future<Output = Result<Self::Output, ToolError>> + Send;
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



/// Converts a PascalCase (or camelCase) identifier to snake_case.
/// Handles acronym runs: `HTTPRequest` → `http_request`.
pub(crate) fn pascal_to_snake(s: &str) -> String {
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

impl Context {
    /// Rejects commands outside the configured allowlist.
    pub fn check_command(&self, cmd: &str) -> Result<(), ToolError> {
        if self.commands().iter().any(|c| c == cmd) {
            Ok(())
        } else {
            Err(ToolError::Security(format!("command '{cmd}' is not in the allowed list")))
        }
    }

    /// Resolves a path and rejects escapes from the configured working directory.
    pub fn resolve(&self, raw: &str) -> Result<PathBuf, ToolError> {
        let path = Path::new(raw);
        let working_dir = normalize_path(self.working_dir());
        let requested = if path.is_absolute() {
            normalize_path(path)
        } else {
            normalize_path(&working_dir.join(path))
        };
        if !requested.starts_with(&working_dir) {
            return Err(ToolError::Security(format!("path '{raw}' escapes the working directory")));
        }
        let canonical_root = canonical_working_dir(&working_dir)?;
        let Ok(relative) = requested.strip_prefix(&working_dir) else {
            return Err(ToolError::Security(format!("path '{raw}' escapes the working directory")));
        };
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
                            return Err(ToolError::Security(format!("path '{raw}' escapes the working directory")));
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
                return Err(ToolError::Security(format!("path '{raw}' escapes the working directory")));
            }
        }

        if !resolved.starts_with(root) {
            return Err(ToolError::Security(format!("path '{raw}' escapes the working directory")));
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

#[cfg(test)]
mod tests {
    use super::*;

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

