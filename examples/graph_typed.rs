//! Typed graph workflow with maps, variables, branches, subflows, and `each`.
//!
//! This example is deterministic and requires no external services.
//! `Amount` intentionally does not implement `Clone`; storing local state does
//! not require cloning application flow values.

use either::Either;
use pravah::{CompiledFlow, Context, Flow, FlowConf, GraphError, Step, compile};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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

fn add_three(root: Flow<AddThreeFlow>) -> Flow<Amount> {
    root.map(|input| Amount {
        value: input.value + 3,
    })
}

fn choose(root: Flow<ChoiceFlow>) -> Flow<Amount> {
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

fn batch(root: Flow<BatchFlow>) -> Flow<Vec<Amount>> {
    root.map(|input| input.values).each(add_three)
}

fn amount(root: Flow<Amount>) -> Flow<Amount> {
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
async fn main() -> Result<(), GraphError> {
    let flow = compile(amount)?;
    println!(
        "typed graph has {} nodes and {} edges",
        flow.graph().nodes.len(),
        flow.graph().edges.len()
    );

    let ctx = Context::new(FlowConf::default());
    let mut runtime = flow.start(Amount { value: 5 }, ctx.clone())?;

    loop {
        match runtime.next().await? {
            Step::Continue => {
                println!(
                    "continue; active frame depth = {}",
                    runtime.state().frame_depth()
                );
            }
            Step::Done(value) => {
                let output = flow.decode_output(value)?;
                println!("done: {output:?}");
                break;
            }
            Step::Suspend(payload) => {
                return Err(GraphError::Invalid(format!(
                    "typed graph unexpectedly suspended: {payload}"
                )));
            }
        }
    }

    run_to_done(compile(choose)?, ChoiceFlow { value: 12 }, ctx.clone()).await?;
    run_to_done(
        compile(batch)?,
        BatchFlow {
            values: vec![AddThreeFlow { value: 1 }, AddThreeFlow { value: 2 }],
        },
        ctx.clone(),
    )
    .await?;
    Ok(())
}

async fn run_to_done<I, O>(
    flow: CompiledFlow<I, O>,
    input: I,
    ctx: Context,
) -> Result<(), GraphError>
where
    I: 'static + Serialize + for<'de> Deserialize<'de> + JsonSchema,
    O: 'static + std::fmt::Debug + Serialize + for<'de> Deserialize<'de> + JsonSchema,
{
    let mut runtime = flow.start(input, ctx)?;
    loop {
        match runtime.next().await? {
            Step::Continue => {}
            Step::Done(value) => {
                println!("done: {:?}", flow.decode_output(value)?);
                return Ok(());
            }
            Step::Suspend(payload) => {
                return Err(GraphError::Invalid(format!(
                    "typed graph unexpectedly suspended: {payload}"
                )));
            }
        }
    }
}
