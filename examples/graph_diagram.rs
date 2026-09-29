//! Draw a workflow that splits input, processes both branches, and joins their results.
//!
//! Writes target/diagrams/graph_diagram.dot without executing the workflow.
//! Render it separately: dot -Tpng target/diagrams/graph_diagram.dot -o target/diagrams/graph_diagram.png

mod support;

use pravah::graph::GraphDiagram;
use pravah::{Flow, compile};
use support::ExampleError;

fn report(root: Flow<String>) -> Flow<String> {
    let (heading, count) = root.split(|text| (text.clone(), text));
    let heading = heading.map(|text| text.to_uppercase());
    let count = count.map(|text| text.chars().count());
    heading.merge(count, |(heading, count)| {
        format!("{heading}: {count} characters")
    })
}

fn main() -> Result<(), ExampleError> {
    let workflow = compile(report)?;
    let diagram = GraphDiagram::from_compiled_flow(&workflow);
    std::fs::create_dir_all("target/diagrams")?;
    std::fs::write("target/diagrams/graph_diagram.dot", diagram.dot())?;
    println!("{}", diagram.mermaid());
    println!("DOT saved to target/diagrams/graph_diagram.dot");
    Ok(())
}
