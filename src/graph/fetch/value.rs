//! Structured request embedding without recursively recoding an existing Value body.

use super::*;
use crate::graph::{GraphError, from_value, to_value};
use std::borrow::Cow;

fn invalid() -> GraphError {
    GraphError::FetchValidation("invalid embedded Fetch request".into())
}

pub(super) fn object<const N: usize>(
    fields: [(&'static str, Value); N],
) -> Result<Value, GraphError> {
    Value::from_shared_object(
        fields
            .into_iter()
            .map(|(key, value)| (Cow::Borrowed(key), value))
            .collect(),
    )
    .map_err(|_| invalid())
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, GraphError> {
    value.get(key).ok_or_else(invalid)
}

impl FetchRequest {
    /// Encodes the envelope while moving an already encoded structured body unchanged.
    pub(crate) fn into_value(self) -> Result<Value, GraphError> {
        let body = match self.body {
            Some(FetchBody::Value(data)) => object([("kind", "value".into()), ("data", data)])?,
            body => to_value(body).map_err(|_| invalid())?,
        };
        object([
            ("method", self.method.into()),
            ("url", self.url.into()),
            ("headers", to_value(self.headers).map_err(|_| invalid())?),
            ("body", body),
        ])
    }

    /// Enforces the closed Serde envelope while sharing a structured body on local delivery.
    pub(crate) fn from_value(value: &Value) -> Result<Self, GraphError> {
        if value
            .object_entries()
            .ok_or_else(invalid)?
            .any(|(key, _)| !["method", "url", "headers", "body"].contains(&key))
        {
            return Err(invalid());
        }
        Ok(Self {
            method: from_value(field(value, "method")?.clone()).map_err(|_| invalid())?,
            url: from_value(field(value, "url")?.clone()).map_err(|_| invalid())?,
            headers: from_value(field(value, "headers")?.clone()).map_err(|_| invalid())?,
            body: read_body(value.get("body"))?,
        })
    }
}

fn read_body(value: Option<&Value>) -> Result<Option<FetchBody>, GraphError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    if value.get("kind").and_then(Value::as_str) == Some("value") {
        Ok(Some(FetchBody::Value(field(value, "data")?.clone())))
    } else {
        from_value(value.clone()).map(Some).map_err(|_| invalid())
    }
}

#[cfg(test)]
#[path = "tests/value.rs"]
mod tests;
