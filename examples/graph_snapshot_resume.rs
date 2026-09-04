//! Graph suspension, state-only snapshot restoration, and resumption.
//!
//! This example is deterministic and requires no external services.

use pravah::{Context, Flow, FlowConf, GraphError, Snapshot, Step};

#[tokio::main]
async fn main() -> Result<(), GraphError> {
    let flow = Flow::<i64>::root()
        .map_named("prepare_approval", |value| value + 1)
        .suspend::<i64>()
        .finish::<i64>()?;
    let ctx = Context::new(FlowConf::default());
    let mut runtime = flow.start(40, ctx)?;

    assert_eq!(runtime.next().await?, Step::Continue);
    let Step::Suspend(payload) = runtime.next().await? else {
        return Err(GraphError::Invalid(
            "snapshot example did not suspend".into(),
        ));
    };
    println!("suspended with {payload}");

    let encoded = serde_json::to_string_pretty(&runtime.snapshot()?).map_err(|err| {
        GraphError::JsonEncode {
            target: "snapshot".into(),
            reason: err.to_string(),
        }
    })?;
    let snapshot: Snapshot =
        serde_json::from_str(&encoded).map_err(|err| GraphError::JsonDecode {
            target: "snapshot".into(),
            reason: err.to_string(),
        })?;
    let mut restored = flow.restore(snapshot, Context::new(FlowConf::default()))?;

    match restored.resume(42_i64).await? {
        Step::Done(value) => println!("resumed with {value}"),
        other => {
            return Err(GraphError::Invalid(format!(
                "snapshot example expected completion, got {other:?}"
            )));
        }
    }
    Ok(())
}
