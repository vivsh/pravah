use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::graph::{GraphError, UntypedGraph};

const FINGERPRINT_VERSION: u32 = 1;
const DIGEST_LEN: usize = 32;

/// Stable identity of a complete serialized graph.
///
/// The digest includes embedded graphs, handler keys, schemas, and executable
/// payloads. Its canonical encoding is versioned independently of Rust's hash
/// implementations and the VM's in-memory representation.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphFingerprint([u8; DIGEST_LEN]);

impl GraphFingerprint {
    pub(super) fn calculate(graph: &UntypedGraph) -> Result<Self, GraphError> {
        let value = serde_json::to_value(graph).map_err(|err| GraphError::JsonEncode {
            target: "graph fingerprint input".into(),
            reason: err.to_string(),
        })?;
        let mut digest = Sha256::new();
        write_canonical(&value, &mut digest);
        Ok(Self(digest.finalize().into()))
    }

    /// Returns the lowercase hexadecimal digest without the algorithm version.
    pub fn as_hex(&self) -> String {
        let mut output = String::with_capacity(DIGEST_LEN * 2);
        for byte in self.0 {
            use std::fmt::Write;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }

    fn encoded(&self) -> String {
        format!("v{FINGERPRINT_VERSION}:{}", self.as_hex())
    }

    fn decode(value: &str) -> Result<Self, String> {
        let expected_prefix = format!("v{FINGERPRINT_VERSION}:");
        let hex = value.strip_prefix(&expected_prefix).ok_or_else(|| {
            format!("unsupported graph fingerprint; expected prefix '{expected_prefix}'")
        })?;
        if hex.len() != DIGEST_LEN * 2 {
            return Err("graph fingerprint digest must contain 64 hexadecimal characters".into());
        }
        let mut digest = [0_u8; DIGEST_LEN];
        for (index, slot) in digest.iter_mut().enumerate() {
            let offset = index * 2;
            let high = decode_nibble(hex.as_bytes()[offset])?;
            let low = decode_nibble(hex.as_bytes()[offset + 1])?;
            *slot = (high << 4) | low;
        }
        Ok(Self(digest))
    }
}

impl fmt::Display for GraphFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.encoded())
    }
}

impl fmt::Debug for GraphFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("GraphFingerprint")
            .field(&self.encoded())
            .finish()
    }
}

impl Serialize for GraphFingerprint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.encoded())
    }
}

impl<'de> Deserialize<'de> for GraphFingerprint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::decode(&value).map_err(de::Error::custom)
    }
}

fn write_canonical(value: &JsonValue, digest: &mut Sha256) {
    match value {
        JsonValue::Null => digest.update(b"n"),
        JsonValue::Bool(false) => digest.update(b"f"),
        JsonValue::Bool(true) => digest.update(b"t"),
        JsonValue::Number(number) => write_bytes(b'd', number.to_string().as_bytes(), digest),
        JsonValue::String(value) => write_bytes(b's', value.as_bytes(), digest),
        JsonValue::Array(values) => {
            write_len(b'a', values.len(), digest);
            for value in values {
                write_canonical(value, digest);
            }
        }
        JsonValue::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
            write_len(b'o', entries.len(), digest);
            for (key, value) in entries {
                write_bytes(b'k', key.as_bytes(), digest);
                write_canonical(value, digest);
            }
        }
    }
}

fn write_len(tag: u8, len: usize, digest: &mut Sha256) {
    digest.update([tag]);
    digest.update((len as u128).to_be_bytes());
}

fn write_bytes(tag: u8, value: &[u8], digest: &mut Sha256) {
    write_len(tag, value.len(), digest);
    digest.update(value);
}

fn decode_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("graph fingerprint contains invalid hexadecimal data".into()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Verifies canonical object ordering does not affect fingerprint input.
    #[test]
    fn canonical_objects_ignore_insertion_order() {
        let mut first = Sha256::new();
        let mut second = Sha256::new();
        write_canonical(&json!({"a": 1, "b": 2}), &mut first);
        write_canonical(&json!({"b": 2, "a": 1}), &mut second);
        assert_eq!(first.finalize().as_slice(), second.finalize().as_slice());
    }

    /// Verifies fingerprint text rejects malformed and unsupported encodings.
    #[test]
    fn fingerprint_text_is_versioned_and_checked() {
        assert!(GraphFingerprint::decode("v2:00").is_err());
        assert!(GraphFingerprint::decode("v1:xyz").is_err());
        let value = GraphFingerprint([7; DIGEST_LEN]);
        assert_eq!(GraphFingerprint::decode(&value.encoded()), Ok(value));
    }
}
