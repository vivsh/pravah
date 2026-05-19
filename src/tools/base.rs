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
    Other(String),
}

impl ToolError {
    /// Returns `true` when the error should abort the flow instead of going back to the model.
    pub fn is_fatal(&self) -> bool {
        matches!(self, ToolError::PathEscape(_) | ToolError::ForbiddenCommand(_))
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

