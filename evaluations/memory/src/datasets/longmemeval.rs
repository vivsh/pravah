use std::collections::BTreeSet;

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::LONGMEMEVAL_REVISION;
use crate::{
    DatasetKind, EvaluationDataset, EvaluationError, EvaluationEvidence, EvaluationGroup,
    EvaluationQuestion, EvidenceGranularity,
};

#[derive(Deserialize)]
struct SourceItem {
    question_id: String,
    question_type: String,
    question: String,
    answer: Value,
    question_date: String,
    haystack_session_ids: Vec<String>,
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<SourceTurn>>,
    #[serde(default)]
    answer_session_ids: Vec<String>,
}

#[derive(Deserialize)]
struct SourceTurn {
    role: String,
    content: String,
    #[serde(default)]
    has_answer: bool,
}

pub(super) fn normalize(
    bytes: &[u8],
    granularity: EvidenceGranularity,
    checksum: String,
) -> Result<EvaluationDataset, EvaluationError> {
    let items: Vec<SourceItem> = serde_json::from_slice(bytes)?;
    let mut groups = Vec::with_capacity(items.len());
    let mut ids = BTreeSet::new();
    for item in items {
        validate_item(&item)?;
        if !ids.insert(item.question_id.clone()) {
            return Err(dataset_error(&item.question_id, "duplicate question_id"));
        }
        groups.push(normalize_item(item, granularity)?);
    }
    Ok(EvaluationDataset {
        kind: DatasetKind::LongMemEval,
        revision: LONGMEMEVAL_REVISION.to_owned(),
        source_sha256: checksum,
        granularity,
        normalization_warnings: Vec::new(),
        groups,
    })
}

fn validate_item(item: &SourceItem) -> Result<(), EvaluationError> {
    let session_count = item.haystack_sessions.len();
    if item.haystack_session_ids.len() != session_count
        || item.haystack_dates.len() != session_count
    {
        return Err(dataset_error(
            &item.question_id,
            "session ids, dates, and session arrays must have equal lengths",
        ));
    }
    if item.question.trim().is_empty() {
        return Err(dataset_error(
            &item.question_id,
            "question must not be empty",
        ));
    }
    Ok(())
}

fn normalize_item(
    item: SourceItem,
    granularity: EvidenceGranularity,
) -> Result<EvaluationGroup, EvaluationError> {
    let evidence = normalize_evidence(&item, granularity)?;
    let relevant = relevant_keys(&item, granularity);
    let asked_at = parse_time(&item.question_date, &item.question_id)?;
    let abstention = item.question_id.ends_with("_abs");
    let question = EvaluationQuestion {
        id: item.question_id.clone(),
        question: item.question,
        expected_answer: display_value(item.answer),
        category: item.question_type,
        asked_at: Some(asked_at),
        relevant_evidence_keys: if abstention { Vec::new() } else { relevant },
        abstention,
    };
    Ok(EvaluationGroup {
        id: item.question_id,
        evidence,
        questions: vec![question],
    })
}

fn normalize_evidence(
    item: &SourceItem,
    granularity: EvidenceGranularity,
) -> Result<Vec<EvaluationEvidence>, EvaluationError> {
    let mut evidence = Vec::new();
    for (index, turns) in item.haystack_sessions.iter().enumerate() {
        let session_id = &item.haystack_session_ids[index];
        let observed_at = parse_time(&item.haystack_dates[index], session_id)?;
        match granularity {
            EvidenceGranularity::Session => {
                evidence.push(session_evidence(session_id, observed_at, turns));
            }
            EvidenceGranularity::Turn => {
                evidence.extend(turn_evidence(session_id, observed_at, turns));
            }
        }
    }
    Ok(evidence)
}

fn session_evidence(
    session_id: &str,
    observed_at: DateTime<Utc>,
    turns: &[SourceTurn],
) -> EvaluationEvidence {
    let content = turns
        .iter()
        .map(|turn| format!("{}: {}", turn.role.trim(), turn.content.trim()))
        .collect::<Vec<_>>()
        .join("\n");
    EvaluationEvidence {
        evidence_key: session_id.to_owned(),
        content,
        observed_at: Some(observed_at),
        source_session_id: session_id.to_owned(),
        source_turn_id: None,
    }
}

fn turn_evidence(
    session_id: &str,
    observed_at: DateTime<Utc>,
    turns: &[SourceTurn],
) -> Vec<EvaluationEvidence> {
    turns
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            let key = turn_key(session_id, index);
            EvaluationEvidence {
                evidence_key: key.clone(),
                content: format!("{}: {}", turn.role.trim(), turn.content.trim()),
                observed_at: Some(observed_at),
                source_session_id: session_id.to_owned(),
                source_turn_id: Some(key),
            }
        })
        .collect()
}

fn relevant_keys(item: &SourceItem, granularity: EvidenceGranularity) -> Vec<String> {
    match granularity {
        EvidenceGranularity::Session => item.answer_session_ids.clone(),
        EvidenceGranularity::Turn => item
            .haystack_sessions
            .iter()
            .enumerate()
            .flat_map(|(session_index, turns)| {
                turns
                    .iter()
                    .enumerate()
                    .filter(|(_, turn)| turn.has_answer)
                    .map(move |(turn_index, _)| {
                        turn_key(&item.haystack_session_ids[session_index], turn_index)
                    })
            })
            .collect(),
    }
}

fn turn_key(session_id: &str, index: usize) -> String {
    format!("{session_id}:turn:{index}")
}

fn display_value(value: Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn parse_time(value: &str, location: &str) -> Result<DateTime<Utc>, EvaluationError> {
    NaiveDateTime::parse_from_str(value, "%Y/%m/%d (%a) %H:%M")
        .map(|value| value.and_utc())
        .map_err(|error| dataset_error(location, error.to_string()))
}

fn dataset_error(location: impl Into<String>, message: impl Into<String>) -> EvaluationError {
    EvaluationError::dataset("LongMemEval", location, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(question_id: &str) -> Vec<u8> {
        format!(
            r#"[{{"question_id":"{question_id}","question_type":"single-session-user","question":"What?","answer":"blue","question_date":"2023/04/10 (Mon) 23:07","haystack_session_ids":["s1"],"haystack_dates":["2023/04/10 (Mon) 17:50"],"haystack_sessions":[[{{"role":"user","content":"It is blue","has_answer":true}},{{"role":"assistant","content":"Noted"}}]],"answer_session_ids":["s1"]}}]"#
        )
        .into_bytes()
    }

    /// Verifies turn ground truth uses only upstream answer-bearing turns.
    #[test]
    fn turn_granularity_uses_answer_flags() {
        let dataset = normalize(
            &fixture("q1"),
            EvidenceGranularity::Turn,
            "fixture".to_owned(),
        )
        .expect("fixture should normalize");
        assert_eq!(
            dataset.groups[0].questions[0].relevant_evidence_keys,
            ["s1:turn:0"]
        );
    }

    /// Verifies official abstention IDs carry no retrieval ground truth.
    #[test]
    fn abstention_questions_have_no_relevant_evidence() {
        let dataset = normalize(
            &fixture("q1_abs"),
            EvidenceGranularity::Session,
            "fixture".to_owned(),
        )
        .expect("fixture should normalize");
        let question = &dataset.groups[0].questions[0];
        assert!(question.abstention);
        assert!(question.relevant_evidence_keys.is_empty());
    }
}
