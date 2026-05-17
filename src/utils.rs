use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use thiserror::Error;

/// Error returned by DAG helpers.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DagError<Id> {
    #[error("duplicate node id: {0:?}")]
    DuplicateId(Id),
    #[error("unknown dependency: {0:?}")]
    UnknownDependency(Id),
    #[error("cyclic dependency detected")]
    Cycle,
}

/// Returns a topological ordering.
/// Every id must be unique and every dependency must exist.
/// Returns [`DagError::Cycle`] when the graph is cyclic.
pub fn topo_sort<Id>(nodes: &[(Id, &[Id])]) -> Result<Vec<Id>, DagError<Id>>
where
    Id: Hash + Eq + Clone,
{
    let layers = topo_layers(nodes)?;
    Ok(layers.into_iter().flatten().collect())
}

/// Returns nodes grouped into parallel layers.
/// Nodes in the same layer do not depend on each other.
/// Returns [`DagError::Cycle`] when no valid layering exists.
pub fn topo_layers<Id>(nodes: &[(Id, &[Id])]) -> Result<Vec<Vec<Id>>, DagError<Id>>
where
    Id: Hash + Eq + Clone,
{
    let mut id_set: HashSet<&Id> = HashSet::with_capacity(nodes.len());
    for (id, _) in nodes {
        if !id_set.insert(id) {
            return Err(DagError::DuplicateId(id.clone()));
        }
    }
    for (_, deps) in nodes {
        for dep in *deps {
            if !id_set.contains(dep) {
                return Err(DagError::UnknownDependency(dep.clone()));
            }
        }
    }

    let mut in_degree: HashMap<&Id, usize> = nodes.iter().map(|(id, _)| (id, 0)).collect();
    let mut succs: HashMap<&Id, Vec<&Id>> = HashMap::new();

    for (id, deps) in nodes {
        for dep in *deps {
            *in_degree.entry(id).or_default() += 1;
            succs.entry(dep).or_default().push(id);
        }
    }

    let mut queue: VecDeque<&Id> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| *id)
        .collect();

    let mut layers: Vec<Vec<Id>> = Vec::new();
    let mut visited = 0usize;

    while !queue.is_empty() {
        let layer_size = queue.len();
        let mut layer = Vec::with_capacity(layer_size);
        for _ in 0..layer_size {
            let Some(id) = queue.pop_front() else {
                break;
            };
            layer.push(id.clone());
            visited += 1;
            if let Some(nexts) = succs.get(id) {
                for &s in nexts {
                    let d = in_degree.entry(s).or_default();
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(s);
                    }
                }
            }
        }
        layers.push(layer);
    }

    if visited == nodes.len() {
        Ok(layers)
    } else {
        Err(DagError::Cycle)
    }
}

/// Cosine similarity between two vectors.
/// Returns `0.0` for mismatched lengths or zero-magnitude inputs.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Euclidean distance between two vectors.
/// Returns `0.0` when the lengths differ.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Euclidean similarity score `1 / (1 + distance)`.
pub fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    1.0 / (1.0 + euclidean_distance(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent nodes share the first layer.
    #[test]
    fn topo_layers_no_deps_single_layer() {
        let nodes = [
            ("a", [].as_slice()),
            ("b", [].as_slice()),
            ("c", [].as_slice()),
        ];
        let layers = topo_layers(&nodes).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].len(), 3);
    }

    /// A linear chain produces one node per layer.
    #[test]
    fn topo_layers_linear_chain() {
        let nodes = [
            ("a", [].as_slice()),
            ("b", ["a"].as_slice()),
            ("c", ["b"].as_slice()),
        ];
        let layers = topo_layers(&nodes).unwrap();
        assert_eq!(layers, vec![vec!["a"], vec!["b"], vec!["c"]]);
    }

    /// A diamond keeps independent branches in the same layer.
    #[test]
    fn topo_layers_diamond() {
        let nodes = [
            ("a", [].as_slice()),
            ("b", ["a"].as_slice()),
            ("c", ["a"].as_slice()),
            ("d", ["b", "c"].as_slice()),
        ];
        let layers = topo_layers(&nodes).unwrap();
        assert_eq!(layers[0], vec!["a"]);
        assert_eq!(layers[1].len(), 2);
        assert!(layers[1].contains(&"b") && layers[1].contains(&"c"));
        assert_eq!(layers[2], vec!["d"]);
    }

    /// Cycles return `DagError::Cycle`.
    #[test]
    fn topo_layers_cycle_returns_error() {
        let nodes = [("a", ["b"].as_slice()), ("b", ["a"].as_slice())];
        assert_eq!(topo_layers(&nodes), Err(DagError::Cycle));
    }

    /// Unknown dependencies are rejected.
    #[test]
    fn topo_layers_unknown_dep_returns_error() {
        let nodes = [("a", ["z"].as_slice())];
        assert_eq!(topo_layers(&nodes), Err(DagError::UnknownDependency("z")));
    }

    /// Duplicate ids are rejected.
    #[test]
    fn topo_layers_duplicate_id_returns_error() {
        let nodes = [("a", [].as_slice()), ("a", [].as_slice())];
        assert_eq!(topo_layers(&nodes), Err(DagError::DuplicateId("a")));
    }

    /// Identical vectors have similarity 1.0.
    #[test]
    fn cosine_identical_vectors() {
        assert_eq!(cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]), 1.0);
    }

    /// Orthogonal vectors have similarity 0.0.
    #[test]
    fn cosine_orthogonal_vectors() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    /// Zero vectors return 0.0.
    #[test]
    fn cosine_zero_vector_returns_zero() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    /// Identical points have distance 0.
    #[test]
    fn euclidean_same_point() {
        assert_eq!(euclidean_distance(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    /// Known triangle distances are preserved.
    #[test]
    fn euclidean_known_distance() {
        assert!((euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }

    /// Identical vectors have similarity 1.0.
    #[test]
    fn euclidean_similarity_identical() {
        assert_eq!(euclidean_similarity(&[1.0, 2.0], &[1.0, 2.0]), 1.0);
    }
}
