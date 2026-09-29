//! Register one handler and build its graph directly.
//!
//! Prefer the typed API for ordinary applications. This example needs no services.

mod support;

use pravah::GraphError;
use pravah::graph::{
    HandlerKey, HandlerRegistry, NodeKind, PreparedGraph, Step, TypeSpec, UntypedGraph,
    UntypedGraphBuilder, Value,
};
use support::ExampleError;

/// Defines serializable structure; the executable handler is registered separately.
fn graph() -> Result<UntypedGraph, GraphError> {
    let mut graph = UntypedGraphBuilder::new("greeting");
    let text = TypeSpec::new("Text", serde_json::json!({"type": "string"}));
    let input = graph.edge("name", text.clone());
    let output = graph.edge("greeting", text);
    graph.set_entry(input).set_exit(output);
    graph.node(
        "greet",
        NodeKind::PureHandler {
            key: HandlerKey::new("greet"),
        },
        vec![input],
        vec![output],
    );
    graph.build()
}

fn greet(inputs: Vec<Value>) -> Result<Vec<Value>, GraphError> {
    let name = inputs
        .first()
        .and_then(Value::as_str)
        .ok_or_else(|| GraphError::Invalid("greet requires a string".into()))?;
    Ok(vec![Value::from(format!("Hello, {name}!"))])
}

/// Prepares the graph with its handler and steps until the result is available.
fn main() -> Result<(), ExampleError> {
    let mut handlers = HandlerRegistry::new();
    handlers.insert_value("greet", greet)?;
    let prepared = PreparedGraph::new(graph()?, handlers)?;
    let mut execution = prepared.start(Value::from("Pravah"), uuid::Uuid::now_v7())?;

    loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Done(value) => {
                println!("{value}");
                return Ok(());
            }
            _ => {
                return Err(ExampleError::from("greeting requested external input"));
            }
        }
    }
}
