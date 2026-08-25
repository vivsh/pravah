use std::io::{Error as IoError, ErrorKind};

use sqlx::encode::IsNull;
use sqlx::error::BoxDynError;
use sqlx::postgres::{PgArgumentBuffer, PgTypeInfo, PgValueRef};
use sqlx::{Decode, Encode, Postgres, Type};

use crate::Embedding;

/// PostgreSQL `vector` binary value bound directly through the pinned SQLx backend.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PgVector(Vec<f32>);

impl PgVector {
    /// Converts an already validated application embedding into its database value.
    pub(crate) fn from_embedding(embedding: &Embedding) -> Self {
        Self(embedding.values().to_vec())
    }

    /// Decodes and validates the pgvector binary wire representation.
    fn decode_bytes(bytes: &[u8]) -> Result<Self, BoxDynError> {
        let header = bytes
            .get(..4)
            .ok_or_else(|| invalid_vector("vector header is incomplete"))?;
        let dimensions = usize::from(u16::from_be_bytes([header[0], header[1]]));
        let unused = u16::from_be_bytes([header[2], header[3]]);
        if unused != 0 {
            return Err(invalid_vector("vector reserved header must be zero"));
        }
        let expected = 4_usize
            .checked_add(dimensions.checked_mul(4).ok_or_else(|| {
                invalid_vector("vector dimensions overflow the binary representation")
            })?)
            .ok_or_else(|| invalid_vector("vector size overflows the binary representation"))?;
        if bytes.len() != expected {
            return Err(invalid_vector(
                "vector payload length does not match its dimensions",
            ));
        }
        let values = bytes[4..]
            .chunks_exact(4)
            .map(|chunk| f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        validate_values(&values)?;
        Ok(Self(values))
    }
}

impl Type<Postgres> for PgVector {
    fn type_info() -> PgTypeInfo {
        PgTypeInfo::with_name("vector")
    }
}

impl Encode<'_, Postgres> for PgVector {
    fn encode_by_ref(&self, buffer: &mut PgArgumentBuffer) -> Result<IsNull, BoxDynError> {
        validate_values(&self.0)?;
        let dimensions = u16::try_from(self.0.len())?;
        buffer.extend(&dimensions.to_be_bytes());
        buffer.extend(&0_u16.to_be_bytes());
        for value in &self.0 {
            buffer.extend(&value.to_be_bytes());
        }
        Ok(IsNull::No)
    }
}

impl Decode<'_, Postgres> for PgVector {
    fn decode(value: PgValueRef<'_>) -> Result<Self, BoxDynError> {
        let bytes = <&[u8] as Decode<Postgres>>::decode(value)?;
        Self::decode_bytes(bytes)
    }
}

fn validate_values(values: &[f32]) -> Result<(), BoxDynError> {
    if values.is_empty() {
        return Err(invalid_vector("vector must contain at least one value"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_vector("vector values must be finite"));
    }
    Ok(())
}

fn invalid_vector(message: &str) -> BoxDynError {
    IoError::new(ErrorKind::InvalidData, message).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies a valid pgvector binary payload decodes without changing values.
    #[test]
    fn valid_binary_vector_decodes() {
        let bytes = [0, 2, 0, 0, 0x3f, 0x80, 0, 0, 0x40, 0, 0, 0];
        let vector = PgVector::decode_bytes(&bytes).unwrap();
        assert_eq!(vector.0, vec![1.0, 2.0]);
    }

    /// Verifies malformed payload lengths are rejected before any element access.
    #[test]
    fn malformed_binary_vector_is_rejected() {
        let error = PgVector::decode_bytes(&[0, 2, 0, 0, 0, 0, 0, 0]).unwrap_err();
        assert!(error.to_string().contains("payload length"));
    }
}
