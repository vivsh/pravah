use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use thiserror::Error;

/// Error produced by DAG algorithms.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DagError<Id> {
    #[error("duplicate node id: {0:?}")]
    DuplicateId(Id),
    #[error("unknown dependency: {0:?}")]
    UnknownDependency(Id),
    #[error("cyclic dependency detected")]
    Cycle,
}

// ── Topological sort ──────────────────────────────────────────────────────────

/// Returns nodes in a valid topological order (dependencies before dependants).
///
/// `nodes` is a slice of `(id, deps)` pairs. Each `id` must be unique;
/// every entry in `deps` must appear as a node `id`.
///
/// Returns [`DagError::Cycle`] if the graph contains a cycle.
pub fn topo_sort<Id>(nodes: &[(Id, &[Id])]) -> Result<Vec<Id>, DagError<Id>>
where
    Id: Hash + Eq + Clone,
{
    let layers = topo_layers(nodes)?;
    Ok(layers.into_iter().flatten().collect())
}

/// Returns nodes grouped into parallel layers.
///
/// All nodes within a layer may execute concurrently — no node in a layer
/// depends on another node in the same layer. Layers are ordered: later layers
/// depend on earlier ones.
///
/// ```rust
/// use pravah::utils::topo_layers;
///
/// let a = "a";
/// let b = "b";
/// let c = "c";
/// let no_deps: [&str; 0] = [];
/// let b_deps = [a];
/// let c_deps = [a];
/// let nodes = [
///     (a, no_deps.as_slice()),
///     (b, b_deps.as_slice()),
///     (c, c_deps.as_slice()),
/// ];
/// let layers = topo_layers(&nodes).unwrap();
/// assert_eq!(layers[0], vec!["a"]);
/// assert_eq!(layers[1].len(), 2); // b and c can run concurrently
/// ```
pub fn topo_layers<Id>(nodes: &[(Id, &[Id])]) -> Result<Vec<Vec<Id>>, DagError<Id>>
where
    Id: Hash + Eq + Clone,
{
    // Validate: unique ids, all deps known
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

    // Kahn's algorithm
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

// ── Vector similarity ─────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length vectors.
///
/// Returns `0.0` if either vector is zero-length (all zeros).
/// Panics in debug mode if the slices have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Euclidean distance between two equal-length vectors.
///
/// Panics in debug mode if the slices have different lengths.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "euclidean_distance: length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Euclidean similarity: `1 / (1 + distance)`, always in `(0, 1]`.
///
/// Higher is more similar. Useful when a score (rather than a distance) is needed.
pub fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    1.0 / (1.0 + euclidean_distance(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── topo_layers ───────────────────────────────────────────────────────────

    /// Independent nodes all land in layer 0.
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

    /// Linear chain a→b→c produces three layers in order.
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

    /// Diamond: b and c are independent (same layer), d depends on both.
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

    /// A cycle returns `DagError::Cycle`.
    #[test]
    fn topo_layers_cycle_returns_error() {
        let nodes = [("a", ["b"].as_slice()), ("b", ["a"].as_slice())];
        assert_eq!(topo_layers(&nodes), Err(DagError::Cycle));
    }

    /// An unknown dependency returns `DagError::UnknownDependency`.
    #[test]
    fn topo_layers_unknown_dep_returns_error() {
        let nodes = [("a", ["z"].as_slice())];
        assert_eq!(topo_layers(&nodes), Err(DagError::UnknownDependency("z")));
    }

    /// A duplicate id returns `DagError::DuplicateId`.
    #[test]
    fn topo_layers_duplicate_id_returns_error() {
        let nodes = [("a", [].as_slice()), ("a", [].as_slice())];
        assert_eq!(topo_layers(&nodes), Err(DagError::DuplicateId("a")));
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

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

    /// Zero vector returns 0.0 without panicking.
    #[test]
    fn cosine_zero_vector_returns_zero() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    // ── euclidean_distance ────────────────────────────────────────────────────

    /// Same point → distance 0.
    #[test]
    fn euclidean_same_point() {
        assert_eq!(euclidean_distance(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    /// 3-4-5 right triangle.
    #[test]
    fn euclidean_known_distance() {
        assert!((euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }

    /// Similarity of identical vectors is 1.0.
    #[test]
    fn euclidean_similarity_identical() {
        assert_eq!(euclidean_similarity(&[1.0, 2.0], &[1.0, 2.0]), 1.0);
    }
}
