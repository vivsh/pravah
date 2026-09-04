//! Shared error and model-facing metadata for standalone graph tools.

pub(crate) mod base;

pub use crate::context::Context;
pub use base::{ToolDefinition, ToolError};
