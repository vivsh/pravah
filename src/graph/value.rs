//! Compact, format-neutral values carried by the graph VM.

use std::fmt;
use std::sync::Arc;

use serde::{
    Serialize,
    de::{self, DeserializeOwned},
    ser,
};
use thiserror::Error;

/// Failure converting a Serde value into Pravah's runtime value domain.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ValueError {
    /// A floating-point value was not finite.
    #[error("non-finite floating-point values are unsupported")]
    NonFiniteNumber,
    /// An integer could not be represented by the runtime's 64-bit domain.
    #[error("integer is outside Pravah's 64-bit runtime value domain")]
    IntegerOutOfRange,
    /// An object contained the same key more than once.
    #[error("duplicate object key '{0}'")]
    DuplicateKey(String),
    /// A Serde value cannot be represented by the OpenAPI-compatible domain.
    #[error("{0}")]
    Unsupported(String),
}

impl ser::Error for ValueError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::Unsupported(msg.to_string())
    }
}

impl de::Error for ValueError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::Unsupported(msg.to_string())
    }
}

#[derive(Clone)]
enum Repr {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    String(Arc<str>),
    Array(Arc<Vec<Value>>),
    Object(Arc<Vec<(Arc<str>, Value)>>),
}

/// A compact OpenAPI-compatible value used by Pravah's runtime.
///
/// Strings and composite values use immutable shared storage, making VM edge
/// reads, fan-out, checkpoints, and snapshots cheap to clone. The internal
/// representation is intentionally private and is not a persistence contract.
#[derive(Clone)]
pub struct Value(Repr);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Repr::Null, Repr::Null) => true,
            (Repr::Bool(left), Repr::Bool(right)) => left == right,
            (Repr::I64(left), Repr::I64(right)) => left == right,
            (Repr::U64(left), Repr::U64(right)) => left == right,
            (Repr::I64(left), Repr::U64(right)) => u64::try_from(*left).ok() == Some(*right),
            (Repr::U64(left), Repr::I64(right)) => Some(*left) == u64::try_from(*right).ok(),
            (Repr::F64(left), Repr::F64(right)) => left == right,
            (Repr::String(left), Repr::String(right)) => left == right,
            (Repr::Array(left), Repr::Array(right)) => left == right,
            (Repr::Object(left), Repr::Object(right)) => left == right,
            _ => false,
        }
    }
}

impl Value {
    /// The null runtime value.
    pub const NULL: Self = Self(Repr::Null);

    /// Creates a finite floating-point value.
    pub fn number(value: f64) -> Result<Self, ValueError> {
        if value.is_finite() {
            Ok(Self(Repr::F64(value)))
        } else {
            Err(ValueError::NonFiniteNumber)
        }
    }

    /// Creates an immutable array from runtime values.
    pub fn array(values: impl IntoIterator<Item = Value>) -> Self {
        let values = values.into_iter().collect::<Vec<_>>();
        Self(Repr::Array(Arc::new(values)))
    }

    /// Creates an immutable object with deterministically ordered keys.
    pub fn object<K>(entries: impl IntoIterator<Item = (K, Value)>) -> Result<Self, ValueError>
    where
        K: Into<String>,
    {
        let entries = entries
            .into_iter()
            .map(|(key, value)| (Arc::from(key.into()), value))
            .collect::<Vec<_>>();
        Self::from_shared_object(entries)
    }

    fn from_shared_object(mut entries: Vec<(Arc<str>, Value)>) -> Result<Self, ValueError> {
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        for pair in entries.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(ValueError::DuplicateKey(pair[0].0.to_string()));
            }
        }
        Ok(Self(Repr::Object(Arc::new(entries))))
    }

    /// Returns whether this value is null.
    pub fn is_null(&self) -> bool {
        matches!(self.0, Repr::Null)
    }

    /// Returns the boolean value, when present.
    pub fn as_bool(&self) -> Option<bool> {
        match self.0 {
            Repr::Bool(value) => Some(value),
            _ => None,
        }
    }

    /// Returns whether this value is a boolean.
    pub fn is_bool(&self) -> bool {
        matches!(self.0, Repr::Bool(_))
    }

    /// Returns whether this value is numeric.
    pub fn is_number(&self) -> bool {
        matches!(self.0, Repr::I64(_) | Repr::U64(_) | Repr::F64(_))
    }

    /// Returns the signed integer value, when present or safely convertible.
    pub fn as_i64(&self) -> Option<i64> {
        match self.0 {
            Repr::I64(value) => Some(value),
            Repr::U64(value) => i64::try_from(value).ok(),
            _ => None,
        }
    }

    /// Returns the unsigned integer value, when present or safely convertible.
    pub fn as_u64(&self) -> Option<u64> {
        match self.0 {
            Repr::U64(value) => Some(value),
            Repr::I64(value) => u64::try_from(value).ok(),
            _ => None,
        }
    }

    /// Returns this value as a floating-point number, including integers.
    pub fn as_f64(&self) -> Option<f64> {
        match self.0 {
            Repr::F64(value) => Some(value),
            Repr::I64(value) => Some(value as f64),
            Repr::U64(value) => Some(value as f64),
            _ => None,
        }
    }

    /// Returns the string contents, when present.
    pub fn as_str(&self) -> Option<&str> {
        match &self.0 {
            Repr::String(value) => Some(value),
            _ => None,
        }
    }

    /// Returns whether this value is a string.
    pub fn is_string(&self) -> bool {
        matches!(self.0, Repr::String(_))
    }

    /// Returns the array contents, when present.
    pub fn as_array(&self) -> Option<&[Value]> {
        match &self.0 {
            Repr::Array(values) => Some(values),
            _ => None,
        }
    }

    /// Returns whether this value is an array.
    pub fn is_array(&self) -> bool {
        matches!(self.0, Repr::Array(_))
    }

    /// Returns whether this value is an object.
    pub fn is_object(&self) -> bool {
        matches!(self.0, Repr::Object(_))
    }

    /// Returns an iterator over deterministically ordered object entries.
    pub fn object_entries(&self) -> Option<impl ExactSizeIterator<Item = (&str, &Value)>> {
        match &self.0 {
            Repr::Object(entries) => Some(entries.iter().map(|(key, value)| (key.as_ref(), value))),
            _ => None,
        }
    }

    /// Looks up an object member by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        let Repr::Object(entries) = &self.0 else {
            return None;
        };
        entries
            .binary_search_by(|(candidate, _)| candidate.as_ref().cmp(key))
            .ok()
            .and_then(|index| entries.get(index))
            .map(|(_, value)| value)
    }
}

impl Default for Value {
    fn default() -> Self {
        Self::NULL
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Repr::Null => formatter.write_str("Null"),
            Repr::Bool(value) => formatter.debug_tuple("Bool").field(value).finish(),
            Repr::I64(value) => formatter.debug_tuple("I64").field(value).finish(),
            Repr::U64(value) => formatter.debug_tuple("U64").field(value).finish(),
            Repr::F64(value) => formatter.debug_tuple("F64").field(value).finish(),
            Repr::String(value) => formatter.debug_tuple("String").field(value).finish(),
            Repr::Array(values) => formatter.debug_tuple("Array").field(values).finish(),
            Repr::Object(entries) => formatter
                .debug_map()
                .entries(entries.iter().map(|(key, value)| (key, value)))
                .finish(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string(self) {
            Ok(value) => formatter.write_str(&value),
            Err(_) => fmt::Debug::fmt(self, formatter),
        }
    }
}

macro_rules! from_signed {
    ($($type:ty),+ $(,)?) => {
        $(impl From<$type> for Value {
            fn from(value: $type) -> Self {
                Self(Repr::I64(i64::from(value)))
            }
        })+
    };
}

macro_rules! from_unsigned {
    ($($type:ty),+ $(,)?) => {
        $(impl From<$type> for Value {
            fn from(value: $type) -> Self {
                Self(Repr::U64(u64::from(value)))
            }
        })+
    };
}

from_signed!(i8, i16, i32, i64);
from_unsigned!(u8, u16, u32, u64);

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self(Repr::Bool(value))
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self(Repr::String(Arc::from(value)))
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self(Repr::String(Arc::from(value)))
    }
}

impl From<Vec<Value>> for Value {
    fn from(value: Vec<Value>) -> Self {
        Self::array(value)
    }
}

/// Converts a serializable Rust value directly into a runtime value.
pub fn to_value<T: Serialize>(value: T) -> Result<Value, ValueError> {
    serde_impl::to_value(value)
}

/// Converts a runtime value directly into a deserializable Rust value.
pub fn from_value<T: DeserializeOwned>(value: Value) -> Result<T, ValueError> {
    T::deserialize(value)
}

mod serde_impl;
#[cfg(test)]
mod tests;
