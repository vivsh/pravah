//! Typed graph workflow with maps, variables, branches, subflows, and `each`.
//!
//! This example is deterministic and requires no external services.

use either::Either;
use pravah::graph;
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Amount {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Bonus {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AddThreeFlow {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Small {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Large {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ChoiceFlow {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct BatchFlow {
    values: Vec<AddThreeFlow>,
}

fn add_three(root: graph::Flow<AddThreeFlow>) -> graph::Flow<Amount> {
    root.map(|input| Amount {
        value: input.value + 3,
    })
}

fn choose(root: graph::Flow<ChoiceFlow>) -> graph::Flow<Amount> {
    root.either(|input| {
        if input.value < 10 {
            Either::Left(Small { value: input.value })
        } else {
            Either::Right(Large { value: input.value })
        }
    })
    .branch(
        |small| {
            small.map(|input| Amount {
                value: input.value + 1,
            })
        },
        |large| {
            large.map(|input| Amount {
                value: input.value * 2,
            })
        },
    )
}

fn batch(root: graph::Flow<BatchFlow>) -> graph::Flow<Vec<Amount>> {
    root.map(|input| input.values).each(add_three)
}

fn amount(root: graph::Flow<Amount>) -> graph::Flow<Amount> {
    let bonus = root.local(Bonus { value: 10 });

    root.load(bonus.clone(), |mut amount, bonus| {
        amount.value += bonus.value;
        amount
    })
    .map(|amount| AddThreeFlow {
        value: amount.value,
    })
    .flow(add_three)
    .store(bonus, |amount, _old_bonus| Bonus {
        value: amount.value,
    })
    .map(|mut amount| {
        amount.value *= 2;
        amount
    })
}

#[tokio::main]
async fn main() -> Result<(), graph::GraphError> {
    let flow = graph::compile(amount)?;
    println!(
        "typed graph has {} nodes and {} edges",
        flow.graph().nodes.len(),
        flow.graph().edges.len()
    );

    let mut runtime = flow.runtime(Amount { value: 5 })?;
    let ctx = Context::new(FlowConf::default());

    loop {
        match runtime.next(ctx.clone()).await? {
            graph::Step::Continue => {
                println!(
                    "continue; active frame depth = {}",
                    runtime.state().frame_depth()
                );
            }
            graph::Step::Done(value) => {
                let output = flow.decode_output(value)?;
                println!("done: {output:?}");
                break;
            }
            graph::Step::Suspend(payload) => {
                return Err(graph::GraphError::Invalid(format!(
                    "typed graph unexpectedly suspended: {payload}"
                )));
            }
        }
    }

    run_to_done(
        graph::compile(choose)?,
        ChoiceFlow { value: 12 },
        ctx.clone(),
    )
    .await?;
    run_to_done(
        graph::compile(batch)?,
        BatchFlow {
            values: vec![AddThreeFlow { value: 1 }, AddThreeFlow { value: 2 }],
        },
        ctx.clone(),
    )
    .await?;
    Ok(())
}

async fn run_to_done<I, O>(
    flow: graph::CompiledFlow<I, O>,
    input: I,
    ctx: Context,
) -> Result<(), graph::GraphError>
where
    I: 'static + Serialize + for<'de> Deserialize<'de> + JsonSchema,
    O: 'static + std::fmt::Debug + Serialize + for<'de> Deserialize<'de> + JsonSchema,
{
    let mut runtime = flow.runtime(input)?;
    loop {
        match runtime.next(ctx.clone()).await? {
            graph::Step::Continue => {}
            graph::Step::Done(value) => {
                println!("done: {:?}", flow.decode_output(value)?);
                return Ok(());
            }
            graph::Step::Suspend(payload) => {
                return Err(graph::GraphError::Invalid(format!(
                    "typed graph unexpectedly suspended: {payload}"
                )));
            }
        }
    }
}
