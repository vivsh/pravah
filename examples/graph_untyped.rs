//! Untyped graph VM example.
//!
//! The graph is deterministic serializable data, while executable behavior is
//! supplied separately through a runtime handler registry.
//!
//! This example requires no external services.

use pravah::graph::{
    GraphError, HandlerKey, HandlerRegistry, NodeKind, PreparedGraph, Step, TypeSpec, UntypedGraph,
    UntypedGraphBuilder, Value, VarKey, VarScope,
};
use pravah::{Context, FlowConf};
use serde_json::json;

fn number_type(name: &str) -> TypeSpec {
    TypeSpec::new(name, json!({ "type": "number" }))
}

fn required_i64(value: &Value, label: &str) -> Result<i64, GraphError> {
    value
        .as_i64()
        .ok_or_else(|| GraphError::Invalid(format!("{label} must be an integer, got {value}")))
}

fn input_i64(inputs: &[Value], index: usize, label: &str) -> Result<i64, GraphError> {
    let value = inputs
        .get(index)
        .ok_or_else(|| GraphError::Invalid(format!("missing {label} at input {index}")))?;
    required_i64(value, label)
}

fn checked_add(left: i64, right: i64, label: &str) -> Result<i64, GraphError> {
    left.checked_add(right)
        .ok_or_else(|| GraphError::Invalid(format!("{label} overflowed i64")))
}

fn checked_double(value: i64) -> Result<i64, GraphError> {
    value
        .checked_mul(2)
        .ok_or_else(|| GraphError::Invalid("double overflowed i64".into()))
}

fn build_child_graph() -> Result<UntypedGraph, GraphError> {
    let mut child = UntypedGraphBuilder::new("child_add_three");
    let input = child.edge("child_input", number_type("Number"));
    let output = child.edge("child_output", number_type("Number"));

    child.set_entry(input).set_exit(output);
    child.node(
        "add_three",
        NodeKind::PureHandler {
            key: HandlerKey::new("add_three"),
        },
        vec![input],
        vec![output],
    );

    child.build()
}

fn build_parent_graph() -> Result<UntypedGraph, GraphError> {
    let child = build_child_graph()?;

    let mut parent = UntypedGraphBuilder::new("parent_vm");
    let input = parent.edge("input", number_type("Number"));
    let loaded = parent.edge("after_load", number_type("Number"));
    let after_child = parent.edge("after_child", number_type("Number"));
    let after_store = parent.edge("after_store", number_type("Number"));
    let output = parent.edge("output", number_type("Number"));

    let bonus = parent.variable_with_value(
        VarKey::new("rust", "Bonus"),
        number_type("Bonus"),
        VarScope::Local,
        Value::from(10_i64),
    );

    parent.set_entry(input).set_exit(output);
    parent.node(
        "load_bonus",
        NodeKind::Load {
            var: bonus,
            key: HandlerKey::new("add_state"),
        },
        vec![input],
        vec![loaded],
    );
    parent.node(
        "call_child",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![loaded],
        vec![after_child],
    );
    parent.node(
        "store_latest",
        NodeKind::Store {
            var: bonus,
            key: HandlerKey::new("store_input"),
        },
        vec![after_child],
        vec![after_store],
    );
    parent.node(
        "double",
        NodeKind::PureHandler {
            key: HandlerKey::new("double"),
        },
        vec![after_store],
        vec![output],
    );

    parent.build()
}

fn registry() -> Result<HandlerRegistry, GraphError> {
    let mut registry = HandlerRegistry::new();

    registry.insert_value("add_state", |inputs: Vec<Value>| {
        let input = input_i64(&inputs, 0, "input")?;
        let state = input_i64(&inputs, 1, "state")?;
        Ok(vec![Value::from(checked_add(input, state, "add_state")?)])
    })?;
    registry.insert_value("add_three", |inputs: Vec<Value>| {
        let input = input_i64(&inputs, 0, "input")?;
        Ok(vec![Value::from(checked_add(input, 3, "add_three")?)])
    })?;
    registry.insert_value("store_input", |inputs: Vec<Value>| {
        let input = inputs
            .first()
            .ok_or_else(|| GraphError::Invalid("store_input requires one input".into()))?;
        Ok(vec![input.clone()])
    })?;
    registry.insert_value("double", |inputs: Vec<Value>| {
        let input = input_i64(&inputs, 0, "input")?;
        Ok(vec![Value::from(checked_double(input)?)])
    })?;

    Ok(registry)
}

#[tokio::main]
async fn main() -> Result<(), GraphError> {
    let graph = build_parent_graph()?;
    let graph_json = pravah::graph::serde::to_json_pretty(&graph)?;
    println!("serialized graph:\n{graph_json}\n");

    let prepared = PreparedGraph::new(graph, registry()?)?;
    let ctx = Context::new(FlowConf::default());
    let mut runtime = prepared.start(Value::from(5_i64), ctx)?;

    loop {
        match runtime.next().await? {
            Step::Continue => {
                println!(
                    "continue; active frame depth = {}",
                    runtime.state().frame_depth()
                );
            }
            Step::Done(value) => {
                println!("done: {value}");
                break;
            }
            Step::Suspend(payload) => {
                return Err(GraphError::Invalid(format!(
                    "untyped graph unexpectedly suspended: {payload}"
                )));
            }
        }
    }

    Ok(())
}
