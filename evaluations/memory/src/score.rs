use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::QuestionObservation;

/// Retrieval quality for one upstream category.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryScore {
    /// Non-abstention questions with retrieval ground truth.
    pub evaluated_questions: usize,
    /// Fraction of relevant evidence keys found in the ranked result list.
    pub mean_evidence_recall: f64,
    /// Fraction of questions with at least one relevant evidence hit.
    pub hit_rate: f64,
    /// Mean reciprocal rank of the first relevant evidence hit.
    pub mean_reciprocal_rank: f64,
}

/// Evidence-level retrieval report independent of answer-generation judging.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalScore {
    /// Questions included in retrieval metrics.
    pub evaluated_questions: usize,
    /// Official abstention questions excluded from retrieval recall.
    pub abstention_questions: usize,
    /// Macro-average fraction of relevant evidence recovered.
    pub mean_evidence_recall: f64,
    /// Fraction of evaluated questions with at least one relevant evidence hit.
    pub hit_rate: f64,
    /// Macro-average reciprocal rank of the first relevant evidence hit.
    pub mean_reciprocal_rank: f64,
    /// Metrics split by upstream category.
    pub categories: BTreeMap<String, CategoryScore>,
}

/// Scores ranked evidence provenance without invoking a generative judge.
pub fn score_retrieval(observations: &[QuestionObservation]) -> RetrievalScore {
    let mut overall = Accumulator::default();
    let mut categories = BTreeMap::<String, Accumulator>::new();
    let mut abstention_questions = 0;
    for observation in observations {
        if observation.abstention || observation.relevant_evidence_keys.is_empty() {
            abstention_questions += usize::from(observation.abstention);
            continue;
        }
        let metrics = question_metrics(observation);
        overall.add(metrics);
        categories
            .entry(observation.category.clone())
            .or_default()
            .add(metrics);
    }
    RetrievalScore {
        evaluated_questions: overall.count,
        abstention_questions,
        mean_evidence_recall: overall.mean_recall(),
        hit_rate: overall.hit_rate(),
        mean_reciprocal_rank: overall.mean_reciprocal_rank(),
        categories: categories
            .into_iter()
            .map(|(category, metrics)| (category, metrics.finish()))
            .collect(),
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct QuestionMetrics {
    recall: f64,
    hit: f64,
    reciprocal_rank: f64,
}

fn question_metrics(observation: &QuestionObservation) -> QuestionMetrics {
    let relevant = observation
        .relevant_evidence_keys
        .iter()
        .collect::<BTreeSet<_>>();
    let retrieved = observation
        .retrieved
        .iter()
        .map(|claim| &claim.evidence_key)
        .collect::<BTreeSet<_>>();
    let hit_count = relevant.intersection(&retrieved).count();
    let first_rank = observation
        .retrieved
        .iter()
        .find(|claim| relevant.contains(&claim.evidence_key))
        .map_or(0.0, |claim| 1.0 / f64::from(claim.rank));
    QuestionMetrics {
        recall: hit_count as f64 / relevant.len() as f64,
        hit: f64::from(hit_count > 0),
        reciprocal_rank: first_rank,
    }
}

#[derive(Default)]
struct Accumulator {
    count: usize,
    recall: f64,
    hit: f64,
    reciprocal_rank: f64,
}

impl Accumulator {
    fn add(&mut self, metrics: QuestionMetrics) {
        self.count += 1;
        self.recall += metrics.recall;
        self.hit += metrics.hit;
        self.reciprocal_rank += metrics.reciprocal_rank;
    }

    fn mean_recall(&self) -> f64 {
        safe_mean(self.recall, self.count)
    }

    fn hit_rate(&self) -> f64 {
        safe_mean(self.hit, self.count)
    }

    fn mean_reciprocal_rank(&self) -> f64 {
        safe_mean(self.reciprocal_rank, self.count)
    }

    fn finish(self) -> CategoryScore {
        CategoryScore {
            evaluated_questions: self.count,
            mean_evidence_recall: self.mean_recall(),
            hit_rate: self.hit_rate(),
            mean_reciprocal_rank: self.mean_reciprocal_rank(),
        }
    }
}

fn safe_mean(total: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[cfg(test)]
mod tests {
    use pravah_memory::MemoryId;

    use super::*;
    use crate::RetrievedClaim;

    fn claim(evidence_key: &str, rank: u32) -> RetrievedClaim {
        RetrievedClaim {
            memory_id: MemoryId::new(),
            evidence_key: evidence_key.to_owned(),
            text: evidence_key.to_owned(),
            rank,
            score: 1.0,
            support_count: 1,
        }
    }

    /// Verifies evidence recall deduplicates several claims from one evidence item.
    #[test]
    fn scores_unique_evidence_keys() {
        let observation = QuestionObservation {
            group_id: "g".to_owned(),
            question_id: "q".to_owned(),
            category: "fact".to_owned(),
            relevant_evidence_keys: vec!["e1".to_owned(), "e2".to_owned()],
            abstention: false,
            retrieved: vec![claim("e1", 1), claim("e1", 2)],
            retrieval_latency_us: 10,
        };
        let score = score_retrieval(&[observation]);
        assert_eq!(score.mean_evidence_recall, 0.5);
        assert_eq!(score.mean_reciprocal_rank, 1.0);
    }

    /// Verifies abstention items are counted but excluded from retrieval recall.
    #[test]
    fn excludes_abstentions_from_retrieval_metrics() {
        let observation = QuestionObservation {
            group_id: "g".to_owned(),
            question_id: "q_abs".to_owned(),
            category: "abstention".to_owned(),
            relevant_evidence_keys: Vec::new(),
            abstention: true,
            retrieved: Vec::new(),
            retrieval_latency_us: 10,
        };
        let score = score_retrieval(&[observation]);
        assert_eq!(score.evaluated_questions, 0);
        assert_eq!(score.abstention_questions, 1);
    }
}
