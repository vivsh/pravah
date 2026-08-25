use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::nodes::{StateNode, node};
use crate::legacy::errors::FlowError;

/// Tuple helper for [`FlowBuilder::split`].
/// Arities 2-16 are covered automatically.
#[doc(hidden)]
#[allow(private_interfaces)]
pub trait SplitOutputs: 'static {
    fn schema_names() -> Vec<String>;
    fn into_nodes(self) -> Result<Vec<StateNode>, FlowError>;
}

/// Tuple helper for [`FlowBuilder::merge`].
/// Arities 2-16 are covered automatically.
#[doc(hidden)]
pub trait MergeInputs: Sized + 'static {
    fn schema_names() -> Vec<String>;
    fn from_values(values: &[Value]) -> Result<Self, FlowError>;
}

macro_rules! impl_split_outputs {
    ($($T:ident : $idx:tt),+) => {
        #[allow(private_interfaces)]
        impl<$($T),+> SplitOutputs for ($($T,)+)
        where
            $($T: Serialize + DeserializeOwned + JsonSchema + 'static,)+
        {
            fn schema_names() -> Vec<String> {
                vec![$($T::schema_name().into(),)+]
            }

            fn into_nodes(self) -> Result<Vec<StateNode>, FlowError> {
                Ok(vec![$(node(self.$idx)?,)+])
            }
        }
    };
}

macro_rules! impl_merge_inputs {
    ($($T:ident : $idx:tt),+) => {
        impl<$($T),+> MergeInputs for ($($T,)+)
        where
            $($T: Serialize + DeserializeOwned + JsonSchema + 'static,)+
        {
            fn schema_names() -> Vec<String> {
                vec![$($T::schema_name().into(),)+]
            }

            fn from_values(values: &[Value]) -> Result<Self, FlowError> {
                Ok(($(
                    serde_json::from_value::<$T>(values[$idx].clone())
                        .map_err(FlowError::Deserialize)?,
                )+))
            }
        }
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

impl_merge_inputs!(A:0, B:1);
impl_merge_inputs!(A:0, B:1, C:2);
impl_merge_inputs!(A:0, B:1, C:2, D:3);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14);
impl_merge_inputs!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15);
