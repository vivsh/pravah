use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

/// Error returned when a required dependency is not present in [`Deps`].
#[derive(Debug, Error)]
pub enum DepsError {
    #[error("missing dependency: {0}")]
    MissingDependency(String),
}

/// Type-erased dependency container.
///
/// Values are stored as `Arc<Arc<T>>` so that the inner `Arc<T>` can be
/// retrieved by value (via `get_arc`) without cloning `T` or rebuilding the
/// container entry on each lookup.
/// Each type `T` may have at most one registration.
#[derive(Default, Clone)]
pub struct Deps(HashMap<TypeId, Arc<dyn Any + Send + Sync>>);

impl Deps {
    /// Register `val` as the single instance of type `T`.
    ///
    /// Overwrites any previously registered value of the same type.
    pub fn insert<T: Any + Send + Sync + 'static>(&mut self, val: Arc<T>) -> &mut Self {
        self.0.insert(TypeId::of::<T>(), Arc::new(val));
        self
    }

    /// Returns a reference to the registered `T`, or `None` if absent.
    pub fn get<T: Any + Send + Sync + 'static>(&self) -> Option<&T> {
        self.0
            .get(&TypeId::of::<T>())
            .and_then(|any| any.downcast_ref::<Arc<T>>())
            .map(Arc::as_ref)
    }

    /// Returns a clone of the `Arc<T>` wrapper, or `None` if absent.
    pub fn get_arc<T: Any + Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.0
            .get(&TypeId::of::<T>())
            .and_then(|any| any.downcast_ref::<Arc<T>>())
            .cloned()
    }

    /// Returns a reference to the registered `T`.
    ///
    /// Returns [`DepsError::MissingDependency`] if `T` has not been registered.
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

    /// `get` retrieves the registered value by type.
    #[test]
    fn get_returns_registered_value() {
        let mut deps = Deps::default();
        deps.insert(Arc::new(MyService { value: 42 }));
        assert_eq!(deps.get::<MyService>().unwrap().value, 42);
    }

    /// `get_arc` returns a cloned Arc to the value.
    #[test]
    fn get_arc_returns_cloned_arc() {
        let mut deps = Deps::default();
        deps.insert(Arc::new(MyService { value: 7 }));
        let arc = deps.get_arc::<MyService>().unwrap();
        assert_eq!(arc.value, 7);
    }

    /// `require` returns `DepsError::MissingDependency` for absent types.
    #[test]
    fn require_missing_returns_error() {
        let deps = Deps::default();
        let err = deps.require::<MyService>().unwrap_err();
        assert!(matches!(err, DepsError::MissingDependency(_)));
    }

    /// `get` returns `None` when the type is not registered.
    #[test]
    fn get_absent_returns_none() {
        let deps = Deps::default();
        assert!(deps.get::<MyService>().is_none());
    }

    /// `Deps` is `Clone` — cloning shares the underlying `Arc`s.
    #[test]
    fn deps_clone_shares_arcs() {
        let mut deps = Deps::default();
        deps.insert(Arc::new(MyService { value: 99 }));
        let cloned = deps.clone();
        assert_eq!(cloned.get::<MyService>().unwrap().value, 99);
    }
}
