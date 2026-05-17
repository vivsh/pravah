use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

/// Returned when a required dependency is missing.
#[derive(Debug, Error)]
pub enum DepsError {
    #[error("missing dependency: {0}")]
    MissingDependency(String),
}

/// Type-indexed dependency container.
/// Each type can be registered once.
/// Values are stored as `Arc<Arc<T>>` so `get_arc` can return a shared handle
/// without rebuilding the stored entry.
#[derive(Default, Clone)]
pub struct Deps(HashMap<TypeId, Arc<dyn Any + Send + Sync>>);

impl Deps {
    /// Registers the value for type `T`.
    /// Replaces any existing value of the same type.
    pub fn insert<T: Any + Send + Sync + 'static>(&mut self, val: Arc<T>) -> &mut Self {
        self.0.insert(TypeId::of::<T>(), Arc::new(val));
        self
    }

    /// Returns the registered value by reference.
    pub fn get<T: Any + Send + Sync + 'static>(&self) -> Option<&T> {
        self.0
            .get(&TypeId::of::<T>())
            .and_then(|any| any.downcast_ref::<Arc<T>>())
            .map(Arc::as_ref)
    }

    /// Returns a cloned `Arc<T>`.
    pub fn get_arc<T: Any + Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.0
            .get(&TypeId::of::<T>())
            .and_then(|any| any.downcast_ref::<Arc<T>>())
            .cloned()
    }

    /// Returns the registered value or a missing-dependency error.
    pub fn require<T: Any + Send + Sync + 'static>(&self) -> Result<&T, DepsError> {
        self.get::<T>()
            .ok_or_else(|| DepsError::MissingDependency(std::any::type_name::<T>().to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MyService {
        value: u32,
    }

    /// `get` returns the registered value.
    #[test]
    fn get_returns_registered_value() {
        let mut deps = Deps::default();
        deps.insert(Arc::new(MyService { value: 42 }));
        assert_eq!(deps.get::<MyService>().unwrap().value, 42);
    }

    /// `get_arc` returns a cloned handle.
    #[test]
    fn get_arc_returns_cloned_arc() {
        let mut deps = Deps::default();
        deps.insert(Arc::new(MyService { value: 7 }));
        let arc = deps.get_arc::<MyService>().unwrap();
        assert_eq!(arc.value, 7);
    }

    /// `require` fails when the type was never registered.
    #[test]
    fn require_missing_returns_error() {
        let deps = Deps::default();
        let err = deps.require::<MyService>().unwrap_err();
        assert!(matches!(err, DepsError::MissingDependency(_)));
    }

    /// `get` returns `None` for missing types.
    #[test]
    fn get_absent_returns_none() {
        let deps = Deps::default();
        assert!(deps.get::<MyService>().is_none());
    }

    /// Cloning `Deps` keeps sharing the same stored values.
    #[test]
    fn deps_clone_shares_arcs() {
        let mut deps = Deps::default();
        deps.insert(Arc::new(MyService { value: 99 }));
        let cloned = deps.clone();
        assert_eq!(cloned.get::<MyService>().unwrap().value, 99);
    }
}
