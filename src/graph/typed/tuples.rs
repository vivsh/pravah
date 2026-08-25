use super::*;

pub trait SplitOutputs: 'static {
    /// Typed flow tuple returned by `split`.
    type Flows;

    fn type_specs() -> Vec<TypeSpec>;
    fn schema_names() -> Vec<String>;
    fn encode_outputs(self, node: &str) -> Result<Vec<Value>, GraphError>;
    fn into_flows(state: Arc<Mutex<TypedBuildState>>, edges: Vec<EdgeId>) -> Self::Flows;
    fn fallback_flows(state: Arc<Mutex<TypedBuildState>>, edge: EdgeId) -> Self::Flows;
}

/// Flow-handle tuple helper for typed graph `merge`.
///
/// Implemented for `Flow<B>` and flow tuples up to total merge arity 16.
/// The merge closure receives values in flow order.
pub trait MergeFlows<Head>: 'static {
    /// Tuple of values passed to the merge closure.
    type Values;

    fn edge_ids(&self) -> Vec<EdgeId>;
    fn same_graph(&self, state: &Arc<Mutex<TypedBuildState>>) -> bool;
    fn decode_values(values: Vec<Value>, node: &str) -> Result<Self::Values, GraphError>;
}

macro_rules! impl_split_outputs {
    ($($T:ident : $idx:tt),+) => {
        impl<$($T),+> SplitOutputs for ($($T,)+)
        where
            $($T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,)+
        {
            type Flows = ($(Flow<$T>,)+);

            fn type_specs() -> Vec<TypeSpec> {
                vec![$(type_spec::<$T>(),)+]
            }

            fn schema_names() -> Vec<String> {
                vec![$($T::schema_name(),)+]
            }

            fn encode_outputs(self, node: &str) -> Result<Vec<Value>, GraphError> {
                Ok(vec![$(encode_value(self.$idx, node)?,)+])
            }

            fn into_flows(state: Arc<Mutex<TypedBuildState>>, edges: Vec<EdgeId>) -> Self::Flows {
                ($(
                    Flow::from_typed(typed_edge(
                        Arc::clone(&state),
                        edges.get($idx).copied().unwrap_or(EdgeId(0)),
                    )),
                )+)
            }

            fn fallback_flows(state: Arc<Mutex<TypedBuildState>>, edge: EdgeId) -> Self::Flows {
                ($(
                    {
                        let _ = stringify!($T);
                        Flow::from_typed(typed_edge(Arc::clone(&state), edge))
                    },
                )+)
            }
        }
    };
}

macro_rules! impl_merge_pair_nodes {
    ($A:ident : $a_idx:tt, $B:ident : $b_idx:tt) => {
        impl<$A, $B> MergeFlows<$A> for Flow<$B>
        where
            $A: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
            $B: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
        {
            type Values = ($A, $B);

            fn edge_ids(&self) -> Vec<EdgeId> {
                vec![self.edge]
            }

            fn same_graph(&self, state: &Arc<Mutex<TypedBuildState>>) -> bool {
                Arc::ptr_eq(state, &self.state)
            }

            fn decode_values(values: Vec<Value>, node: &str) -> Result<Self::Values, GraphError> {
                if values.len() != 2 {
                    return Err(GraphError::OutputArity {
                        node: node.to_string(),
                        expected: 2,
                        got: values.len(),
                    });
                }
                let mut values = values.into_iter();
                Ok((
                    from_value::<$A>(values.next().ok_or_else(|| {
                        GraphError::Invalid(format!("node '{node}' received too few input values"))
                    })?)
                    .map_err(|err| {
                        GraphError::Invalid(format!(
                            "node '{node}' failed to decode input '{}': {err}",
                            $A::schema_name()
                        ))
                    })?,
                    from_value::<$B>(values.next().ok_or_else(|| {
                        GraphError::Invalid(format!("node '{node}' received too few input values"))
                    })?)
                    .map_err(|err| {
                        GraphError::Invalid(format!(
                            "node '{node}' failed to decode input '{}': {err}",
                            $B::schema_name()
                        ))
                    })?,
                ))
            }
        }
    };
}

macro_rules! impl_merge_nodes {
    ($A:ident : $a_idx:tt, $($T:ident : $idx:tt),+) => {
        impl<$A, $($T),+> MergeFlows<$A> for ($(Flow<$T>,)+)
        where
            $A: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
            $($T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,)+
        {
            type Values = ($A, $($T,)+);

            fn edge_ids(&self) -> Vec<EdgeId> {
                vec![$(self.$idx.edge,)+]
            }

            fn same_graph(&self, state: &Arc<Mutex<TypedBuildState>>) -> bool {
                true $(&& Arc::ptr_eq(state, &self.$idx.state))+
            }

            fn decode_values(values: Vec<Value>, node: &str) -> Result<Self::Values, GraphError> {
                let expected = 1 + count_idents!($($T),+);
                if values.len() != expected {
                    return Err(GraphError::OutputArity {
                        node: node.to_string(),
                        expected,
                        got: values.len(),
                    });
                }
                let mut values = values.into_iter();
                Ok((
                    from_value::<$A>(
                        values
                            .next()
                            .ok_or_else(|| GraphError::Invalid(format!(
                                "node '{node}' received too few input values"
                            )))?,
                    )
                    .map_err(|err| GraphError::Invalid(format!(
                        "node '{node}' failed to decode input '{}': {err}",
                        $A::schema_name()
                    )))?,
                    $(
                        from_value::<$T>(
                            values
                                .next()
                                .ok_or_else(|| GraphError::Invalid(format!(
                                    "node '{node}' received too few input values"
                                )))?,
                        )
                        .map_err(|err| GraphError::Invalid(format!(
                            "node '{node}' failed to decode input '{}': {err}",
                            $T::schema_name()
                        )))?,
                    )+
                ))
            }
        }
    };
}

macro_rules! count_idents {
    ($($T:ident),+) => {
        <[()]>::len(&[$(count_idents!(@one $T)),+])
    };
    (@one $T:ident) => {
        ()
    };
}

impl_split_outputs!(A:0, B:1);
impl_split_outputs!(A:0, B:1, C:2);
impl_split_outputs!(A:0, B:1, C:2, D:3);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14);
impl_split_outputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15);

impl_merge_pair_nodes!(A:0, B:1);
impl_merge_nodes!(A:0, B:0, C:1);
impl_merge_nodes!(A:0, B:0, C:1, D:2);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7, J:8);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7, J:8, K:9);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7, J:8, K:9, L:10);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7, J:8, K:9, L:10, M:11);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7, J:8, K:9, L:10, M:11, N:12);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7, J:8, K:9, L:10, M:11, N:12, O:13);
impl_merge_nodes!(A:0, B:0, C:1, D:2, E:3, F:4, G:5, H:6, I:7, J:8, K:9, L:10, M:11, N:12, O:13, P:14);
