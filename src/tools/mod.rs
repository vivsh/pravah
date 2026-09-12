//! Shared error and model-facing metadata for standalone graph tools.

pub(crate) mod base;
mod schema;

pub use crate::context::Context;
pub use base::{ToolDefinition, ToolError};
pub(crate) use schema::tool_definition;
