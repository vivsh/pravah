//! Deterministic adapters from pinned upstream JSON into application evidence.

mod locomo;
mod longmemeval;

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::{DatasetKind, EvaluationDataset, EvaluationError, EvidenceGranularity};

/// Pinned LoCoMo repository revision.
pub const LOCOMO_REVISION: &str = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376";
/// SHA-256 of the pinned `data/locomo10.json` source.
pub const LOCOMO_SHA256: &str = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4";
/// Pinned cleaned LongMemEval dataset revision.
pub const LONGMEMEVAL_REVISION: &str = "98d7416c24c778c2fee6e6f3006e7a073259d48f";
/// SHA-256 of the pinned cleaned LongMemEval-S source.
pub const LONGMEMEVAL_S_SHA256: &str =
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442";
/// SHA-256 of the pinned LongMemEval oracle source, useful for adapter smoke tests.
pub const LONGMEMEVAL_ORACLE_SHA256: &str =
    "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c";
/// SHA-256 of the pinned cleaned LongMemEval-M source.
pub const LONGMEMEVAL_M_SHA256: &str =
    "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f";

/// Loads and checksum-validates one official dataset source.
pub async fn load_official(
    path: impl AsRef<Path>,
    kind: DatasetKind,
    granularity: EvidenceGranularity,
) -> Result<EvaluationDataset, EvaluationError> {
    let path = path.as_ref();
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|error| EvaluationError::io(path, error))?;
    let checksum = sha256(&bytes);
    validate_checksum(kind, &checksum)?;
    match kind {
        DatasetKind::LoCoMo => locomo::normalize(&bytes, granularity, checksum),
        DatasetKind::LongMemEval => longmemeval::normalize(&bytes, granularity, checksum),
    }
}

fn validate_checksum(kind: DatasetKind, checksum: &str) -> Result<(), EvaluationError> {
    let valid = match kind {
        DatasetKind::LoCoMo => checksum == LOCOMO_SHA256,
        DatasetKind::LongMemEval => [
            LONGMEMEVAL_S_SHA256,
            LONGMEMEVAL_ORACLE_SHA256,
            LONGMEMEVAL_M_SHA256,
        ]
        .contains(&checksum),
    };
    if valid {
        return Ok(());
    }
    Err(EvaluationError::dataset(
        dataset_name(kind),
        "source",
        format!("checksum {checksum} does not match a pinned official file"),
    ))
}

fn dataset_name(kind: DatasetKind) -> &'static str {
    match kind {
        DatasetKind::LoCoMo => "LoCoMo",
        DatasetKind::LongMemEval => "LongMemEval",
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies unknown dataset contents fail before schema parsing.
    #[tokio::test]
    async fn rejects_unpinned_source() {
        let directory = tempfile::tempdir().expect("temporary directory should be available");
        let path = directory.path().join("dataset.json");
        tokio::fs::write(&path, b"[]")
            .await
            .expect("fixture write should succeed");
        let error = load_official(&path, DatasetKind::LoCoMo, EvidenceGranularity::Session)
            .await
            .expect_err("unpinned data must fail");
        assert!(error.to_string().contains("checksum"));
    }
}
