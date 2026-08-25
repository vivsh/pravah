use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::LOCOMO_REVISION;
use crate::{
    DatasetKind, EvaluationDataset, EvaluationError, EvaluationEvidence, EvaluationGroup,
    EvaluationQuestion, EvidenceGranularity,
};

#[derive(Deserialize)]
struct SourceItem {
    sample_id: String,
    conversation: BTreeMap<String, Value>,
    qa: Vec<SourceQuestion>,
}

#[derive(Deserialize)]
struct SourceTurn {
    speaker: String,
    dia_id: String,
    text: String,
}

#[derive(Deserialize)]
struct SourceQuestion {
    question: String,
    #[serde(default)]
    answer: Option<Value>,
    #[serde(default)]
    adversarial_answer: Option<Value>,
    category: Value,
    #[serde(default)]
    evidence: Vec<String>,
}

pub(super) fn normalize(
    bytes: &[u8],
    granularity: EvidenceGranularity,
    checksum: String,
) -> Result<EvaluationDataset, EvaluationError> {
    let items: Vec<SourceItem> = serde_json::from_slice(bytes)?;
    let mut groups = Vec::with_capacity(items.len());
    let mut ids = BTreeSet::new();
    let mut warnings = Vec::new();
    for item in items {
        if !ids.insert(item.sample_id.clone()) {
            return Err(dataset_error(&item.sample_id, "duplicate sample_id"));
        }
        groups.push(normalize_item(item, granularity, &mut warnings)?);
    }
    Ok(EvaluationDataset {
        kind: DatasetKind::LoCoMo,
        revision: LOCOMO_REVISION.to_owned(),
        source_sha256: checksum,
        granularity,
        normalization_warnings: warnings,
        groups,
    })
}

fn normalize_item(
    item: SourceItem,
    granularity: EvidenceGranularity,
    warnings: &mut Vec<String>,
) -> Result<EvaluationGroup, EvaluationError> {
    let sessions = source_sessions(&item)?;
    let mapping = evidence_mapping(&sessions, granularity);
    let evidence = normalize_evidence(&sessions, granularity);
    let mut questions = Vec::with_capacity(item.qa.len());
    for (index, question) in item.qa.into_iter().enumerate() {
        questions.push(normalize_question(
            &item.sample_id,
            index,
            question,
            &mapping,
            warnings,
        )?);
    }
    Ok(EvaluationGroup {
        id: item.sample_id,
        evidence,
        questions,
    })
}

fn source_sessions(item: &SourceItem) -> Result<Vec<SourceSession>, EvaluationError> {
    let mut sessions = Vec::new();
    for (key, value) in &item.conversation {
        let Some(number) = session_number(key) else {
            continue;
        };
        let turns: Vec<SourceTurn> = serde_json::from_value(value.clone()).map_err(|error| {
            dataset_error(format!("{}.{}", item.sample_id, key), error.to_string())
        })?;
        let date_key = format!("{key}_date_time");
        let observed_at = item
            .conversation
            .get(&date_key)
            .and_then(Value::as_str)
            .map(parse_locomo_time)
            .transpose()?;
        sessions.push(SourceSession {
            number,
            key: key.clone(),
            observed_at,
            turns,
        });
    }
    sessions.sort_by_key(|session| session.number);
    Ok(sessions)
}

struct SourceSession {
    number: u32,
    key: String,
    observed_at: Option<DateTime<Utc>>,
    turns: Vec<SourceTurn>,
}

fn session_number(key: &str) -> Option<u32> {
    key.strip_prefix("session_")?.parse().ok()
}

fn evidence_mapping(
    sessions: &[SourceSession],
    granularity: EvidenceGranularity,
) -> BTreeMap<String, String> {
    let mut mapping = BTreeMap::new();
    for session in sessions {
        for turn in &session.turns {
            let key = match granularity {
                EvidenceGranularity::Session => session.key.clone(),
                EvidenceGranularity::Turn => turn.dia_id.clone(),
            };
            mapping.insert(turn.dia_id.clone(), key);
        }
    }
    mapping
}

fn normalize_evidence(
    sessions: &[SourceSession],
    granularity: EvidenceGranularity,
) -> Vec<EvaluationEvidence> {
    sessions
        .iter()
        .flat_map(|session| match granularity {
            EvidenceGranularity::Session => vec![session_evidence(session)],
            EvidenceGranularity::Turn => session
                .turns
                .iter()
                .map(|turn| turn_evidence(session, turn))
                .collect(),
        })
        .collect()
}

fn session_evidence(session: &SourceSession) -> EvaluationEvidence {
    let content = session
        .turns
        .iter()
        .map(|turn| format!("{}: {}", turn.speaker.trim(), turn.text.trim()))
        .collect::<Vec<_>>()
        .join("\n");
    EvaluationEvidence {
        evidence_key: session.key.clone(),
        content,
        observed_at: session.observed_at,
        source_session_id: session.key.clone(),
        source_turn_id: None,
    }
}

fn turn_evidence(session: &SourceSession, turn: &SourceTurn) -> EvaluationEvidence {
    EvaluationEvidence {
        evidence_key: turn.dia_id.clone(),
        content: format!("{}: {}", turn.speaker.trim(), turn.text.trim()),
        observed_at: session.observed_at,
        source_session_id: session.key.clone(),
        source_turn_id: Some(turn.dia_id.clone()),
    }
}

fn normalize_question(
    sample_id: &str,
    index: usize,
    question: SourceQuestion,
    mapping: &BTreeMap<String, String>,
    warnings: &mut Vec<String>,
) -> Result<EvaluationQuestion, EvaluationError> {
    let mut relevant = Vec::new();
    for annotation in question.evidence {
        for evidence_id in evidence_ids(&annotation) {
            if let Some(key) = mapping.get(&evidence_id) {
                if !relevant.contains(key) {
                    relevant.push(key.clone());
                }
            } else {
                warnings.push(format!(
                    "{sample_id}.qa[{index}]: ignored unknown dialogue id {evidence_id}"
                ));
            }
        }
    }
    let answer = question
        .answer
        .or(question.adversarial_answer)
        .ok_or_else(|| {
            dataset_error(
                format!("{sample_id}.qa[{index}].answer"),
                "answer or adversarial_answer is required",
            )
        })?;
    Ok(EvaluationQuestion {
        id: format!("{sample_id}:q{:03}", index + 1),
        question: required_text(sample_id, index, "question", question.question)?,
        expected_answer: display_value(answer),
        category: display_value(question.category),
        asked_at: None,
        relevant_evidence_keys: relevant,
        abstention: false,
    })
}

fn evidence_ids(annotation: &str) -> Vec<String> {
    annotation
        .split(|character: char| character.is_whitespace() || character == ';')
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .strip_prefix("D:")
                .map_or_else(|| value.to_owned(), |rest| format!("D{rest}"))
        })
        .collect()
}

fn required_text(
    sample_id: &str,
    index: usize,
    field: &str,
    value: String,
) -> Result<String, EvaluationError> {
    if value.trim().is_empty() {
        return Err(dataset_error(
            format!("{sample_id}.qa[{index}].{field}"),
            "value must not be empty",
        ));
    }
    Ok(value)
}

fn display_value(value: Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn parse_locomo_time(value: &str) -> Result<DateTime<Utc>, EvaluationError> {
    let parsed = NaiveDateTime::parse_from_str(value, "%I:%M %P on %d %B, %Y")
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%I:%M %P on %e %B, %Y"))
        .map_err(|error| dataset_error("conversation date", error.to_string()))?;
    Ok(parsed.and_utc())
}

fn dataset_error(location: impl Into<String>, message: impl Into<String>) -> EvaluationError {
    EvaluationError::dataset("LoCoMo", location, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::LOCOMO_SHA256;

    /// Verifies exact dialogue ground truth becomes exact turn evidence keys.
    #[test]
    fn turn_granularity_preserves_dialogue_ids() {
        let bytes = br#"[{"sample_id":"conv-1","conversation":{"session_1":[{"speaker":"A","dia_id":"D1:1","text":"hello"}],"session_1_date_time":"10:04 am on 19 December, 2023"},"qa":[{"question":"What?","answer":"hello","category":1,"evidence":["D1:1"]}]}]"#;
        let dataset = normalize(bytes, EvidenceGranularity::Turn, LOCOMO_SHA256.to_owned())
            .expect("fixture should normalize");
        assert_eq!(dataset.groups[0].evidence[0].evidence_key, "D1:1");
        assert_eq!(
            dataset.groups[0].questions[0].relevant_evidence_keys,
            ["D1:1"]
        );
    }

    /// Verifies session normalization concatenates speaker-labelled turns.
    #[test]
    fn session_granularity_concatenates_turns() {
        let bytes = br#"[{"sample_id":"conv-1","conversation":{"session_1":[{"speaker":"A","dia_id":"D1:1","text":"hello"},{"speaker":"B","dia_id":"D1:2","text":"hi"}]},"qa":[]}]"#;
        let dataset = normalize(
            bytes,
            EvidenceGranularity::Session,
            LOCOMO_SHA256.to_owned(),
        )
        .expect("fixture should normalize");
        assert_eq!(dataset.groups[0].evidence[0].content, "A: hello\nB: hi");
    }
}
