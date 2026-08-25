//! Typed composition that reuses the same subflow at two call sites.
//!
//! This example is deterministic and requires no external services.

use pravah::graph::{self, Flow, GraphError, Step};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Increment(i64);

fn increment(root: Flow<Increment>) -> Flow<Increment> {
    root.map_named("increment", |Increment(value)| Increment(value + 1))
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Twice(i64);

fn twice(root: Flow<Twice>) -> Flow<i64> {
    root.map(|Twice(value)| Increment(value))
        .flow(increment)
        .flow(increment)
        .map(|Increment(value)| value)
}

#[tokio::main]
async fn main() -> Result<(), GraphError> {
    let flow = graph::compile(twice)?;
    let mut runtime = flow.runtime(Twice(40))?;
    let ctx = Context::new(FlowConf::default());

    loop {
        match runtime.next(ctx.clone()).await? {
            Step::Continue => {}
            Step::Done(value) => {
                println!("{}", flow.decode_output(value)?);
                return Ok(());
            }
            Step::Suspend(_) => {
                return Err(GraphError::Invalid(
                    "composition example unexpectedly suspended".into(),
                ));
            }
        }
    }
}
