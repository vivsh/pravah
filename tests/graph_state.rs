use pravah::graph::{AgentError, AgentResponse, SNAPSHOT_VERSION, Value};
use pravah::{Agent, AgentConfig, Context, Flow, GraphError, Snapshot, Step, compile};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Encode(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    Decode(#[from] ciborium::de::Error<std::io::Error>),
    #[error(transparent)]
    Value(#[from] pravah::graph::ValueError),
    #[error("{0}")]
    Missing(&'static str),
}

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Application {
    owner: String,
    updates: u32,
}

fn state(updates: u32) -> Application {
    Application {
        owner: "private".into(),
        updates,
    }
}

fn pure(root: Flow<u32>) -> Flow<u32> {
    root.map(|input| input + 1)
}

fn child(root: Flow<u32>) -> Flow<u32> {
    root.map(|input| input * 2)
}

fn composed(root: Flow<Vec<u32>>) -> Flow<Vec<u32>> {
    root.each(child)
}

fn pause(root: Flow<u32>) -> Flow<String> {
    root.suspend::<String>()
}

/// Reads and writes are independent of graph values, UUIDs and independent executions.
#[test]
fn initial_state_is_isolated_and_not_graph_data() -> Result<(), TestError> {
    let flow = compile(pure)?;
    let mut first = flow.start_with_state(2, state(1), Uuid::nil())?;
    let second = flow.start_with_state(2, state(8), Uuid::nil())?;
    assert_eq!(
        first.snapshot()?.graph_fingerprint(),
        second.snapshot()?.graph_fingerprint()
    );
    assert_eq!(first.get_state::<Application>()?, state(1));
    first.set_state(state(3))?;
    assert_eq!(second.get_state::<Application>()?, state(8));
    let before = serde_json::to_value(first.snapshot()?)?;
    first.set_state(state(4))?;
    let after = serde_json::to_value(first.snapshot()?)?;
    assert_eq!(
        before.pointer("/state/frames"),
        after.pointer("/state/frames")
    );
    assert!(matches!(first.next()?, Step::Done(value) if value == Value::from(3u32)));
    assert_eq!(first.get_state::<Application>()?, state(4));
    assert!(flow.graph().variables.is_empty());
    assert!(first.history().entries().is_empty());
    Ok(())
}

/// Child frame creation/removal and completed snapshots never own or discard application state.
#[test]
fn state_survives_children_and_completed_codecs() -> Result<(), TestError> {
    let flow = compile(composed)?;
    let mut runtime = flow.start_with_state(vec![2, 3], state(0), Uuid::nil())?;
    for _ in 0..100 {
        match runtime.next()? {
            Step::Continue => runtime.set_state(state(5))?,
            Step::Done(value) => {
                assert_eq!(pravah::graph::from_value::<Vec<u32>>(value)?, vec![4, 6]);
                break;
            }
            _ => return Err(TestError::Missing("unexpected external operation")),
        }
    }
    assert_eq!(runtime.state().frame_depth(), 0);
    for snapshot in roundtrips(&runtime.snapshot()?)? {
        let mut restored = flow.restore(snapshot)?;
        assert_eq!(restored.get_state::<Application>()?, state(5));
        restored.set_state(state(7))?;
        assert_eq!(restored.get_state::<Application>()?, state(7));
    }
    Ok(())
}

/// Explicit input suspensions permit application updates without altering their payload or resume type.
#[test]
fn suspended_state_updates_are_independent() -> Result<(), TestError> {
    let flow = compile(pause)?;
    let mut runtime = flow.start_with_state(4, state(0), Uuid::nil())?;
    assert!(matches!(runtime.next()?, Step::Suspend(_)));
    runtime.set_state(state(1))?;
    for snapshot in roundtrips(&runtime.snapshot()?)? {
        let mut restored = flow.restore(snapshot)?;
        assert_eq!(restored.get_state::<Application>()?, state(1));
        restored.set_state(state(2))?;
        restored.resume("answer")?;
        assert!(matches!(restored.next()?, Step::Done(value) if value == Value::from("answer")));
        assert_eq!(restored.get_state::<Application>()?, state(2));
    }
    Ok(())
}

#[derive(Deserialize, JsonSchema)]
struct Fallible(u32);

impl Serialize for Fallible {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            99 => Err(serde::ser::Error::custom("cannot encode")),
            98 => serializer.serialize_str("wrong shape"),
            value => serializer.serialize_u32(value),
        }
    }
}

/// Missing state, wrong types, bad Serde and invalid shapes reject atomically.
#[test]
fn rejected_state_operations_leave_execution_unchanged() -> Result<(), TestError> {
    let flow = compile(pure)?;
    let mut absent = flow.start(2, Uuid::nil())?;
    assert!(matches!(
        absent.get_state::<Application>(),
        Err(GraphError::ApplicationState(_))
    ));
    assert!(matches!(
        absent.set_state(state(0)),
        Err(GraphError::ApplicationState(_))
    ));
    assert!(
        serde_json::to_value(absent.snapshot()?)?
            .pointer("/state/application_state")
            .is_none()
    );
    assert!(flow.start_with_state(2, Fallible(99), Uuid::nil()).is_err());
    let mut runtime = flow.start_with_state(2, Fallible(1), Uuid::nil())?;
    let before = serde_json::to_vec(&runtime.snapshot()?)?;
    assert!(matches!(
        runtime.set_state(Fallible(99)),
        Err(GraphError::ValueConversion { .. })
    ));
    assert!(matches!(
        runtime.set_state(Fallible(98)),
        Err(GraphError::Schema { .. })
    ));
    assert!(matches!(
        runtime.get_state::<u32>(),
        Err(GraphError::ApplicationState(_))
    ));
    assert!(matches!(
        runtime.set_state(5u32),
        Err(GraphError::ApplicationState(_))
    ));
    assert_eq!(before, serde_json::to_vec(&runtime.snapshot()?)?);
    assert_eq!(runtime.get_state::<Fallible>()?.0, 1);
    Ok(())
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Bounded(#[schemars(range(max = 7))] u32);

/// Boundary validation enforces constraints beyond primitive shape checks before mutation.
#[test]
fn full_schema_constraints_are_enforced() -> Result<(), TestError> {
    let flow = compile(pure)?;
    assert!(matches!(
        flow.start_with_state(1, Bounded(8), Uuid::nil()),
        Err(GraphError::ApplicationState(_))
    ));
    let mut runtime = flow.start_with_state(1, Bounded(2), Uuid::nil())?;
    let before = serde_json::to_vec(&runtime.snapshot()?)?;
    assert!(matches!(
        runtime.set_state(Bounded(8)),
        Err(GraphError::ApplicationState(_))
    ));
    assert_eq!(before, serde_json::to_vec(&runtime.snapshot()?)?);
    let mut corrupt = serde_json::to_value(runtime.snapshot()?)?;
    *corrupt
        .pointer_mut("/state/application_state/1")
        .ok_or(TestError::Missing("state value"))? = json!(8);
    assert!(matches!(
        flow.restore(serde_json::from_value(corrupt)?),
        Err(GraphError::SnapshotValidation(_))
    ));
    Ok(())
}

fn agent(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}

async fn configure(input: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test",
        "Answer.",
        pravah::clients::Message::user(input),
    ))
}

fn agent_flow(root: Flow<String>) -> Flow<String> {
    root.agent(agent)
}

/// Pending requests and accepted failures prevent writes, including after restoration and retry.
#[test]
fn outstanding_and_accepted_agents_protect_state() -> Result<(), TestError> {
    let flow = compile(agent_flow)?;
    let mut runtime = flow.start_with_state("question".into(), state(0), Uuid::nil())?;
    let Step::Agent(request) = runtime.next()? else {
        return Err(TestError::Missing("configuration request"));
    };
    let before = serde_json::to_vec(&runtime.snapshot()?)?;
    assert!(matches!(
        runtime.set_state(state(1)),
        Err(GraphError::ApplicationStateBusy)
    ));
    assert_eq!(before, serde_json::to_vec(&runtime.snapshot()?)?);
    runtime.resume_agent(AgentResponse::new(
        request.id(),
        Err(AgentError::new("offline", "failed")),
    ))?;
    assert!(runtime.pending_agent().is_none());
    let before = serde_json::to_vec(&runtime.snapshot()?)?;
    assert!(matches!(
        runtime.set_state(state(1)),
        Err(GraphError::ApplicationStateBusy)
    ));
    assert!(matches!(
        runtime.next(),
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(before, serde_json::to_vec(&runtime.snapshot()?)?);
    for snapshot in roundtrips(&runtime.snapshot()?)? {
        let mut restored = flow.restore(snapshot)?;
        assert_eq!(restored.get_state::<Application>()?, state(0));
        assert!(matches!(
            restored.set_state(state(2)),
            Err(GraphError::ApplicationStateBusy)
        ));
        assert!(matches!(
            restored.next(),
            Err(GraphError::AgentFailed { .. })
        ));
    }
    Ok(())
}

/// A state-only snapshot cannot bypass the version gate or the schema validation after completion.
#[test]
fn corrupt_and_obsolete_snapshots_are_rejected() -> Result<(), TestError> {
    let flow = compile(pure)?;
    let mut runtime = flow.start_with_state(1, state(0), Uuid::nil())?;
    runtime.next()?;
    let original = serde_json::to_value(runtime.snapshot()?)?;
    for (path, value) in [
        (
            "/state/application_state/1",
            json!({"owner": 4, "updates": 0}),
        ),
        ("/state/application_state/0/name", json!("")),
        ("/state/application_state/0/schema/type", json!(7)),
    ] {
        let mut corrupt = original.clone();
        *corrupt
            .pointer_mut(path)
            .ok_or(TestError::Missing("state metadata"))? = value;
        assert!(matches!(
            flow.restore(serde_json::from_value(corrupt)?),
            Err(GraphError::SnapshotValidation(_))
        ));
    }
    let mut obsolete = original;
    *obsolete
        .get_mut("snapshot_version")
        .ok_or(TestError::Missing("snapshot version"))? = json!(SNAPSHOT_VERSION - 1);
    assert!(matches!(
        flow.restore(serde_json::from_value(obsolete)?),
        Err(GraphError::SnapshotVersion { .. })
    ));
    Ok(())
}

/// Uses independent codecs without changing the runtime's shared state representation.
fn roundtrips(snapshot: &Snapshot) -> Result<[Snapshot; 2], TestError> {
    let mut cbor = Vec::new();
    ciborium::into_writer(snapshot, &mut cbor)?;
    Ok([
        serde_json::from_slice(&serde_json::to_vec(snapshot)?)?,
        ciborium::from_reader(cbor.as_slice())?,
    ])
}
