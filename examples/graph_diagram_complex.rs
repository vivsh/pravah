//! Builds a graph-backed flow diagram and renders it with Graphviz.
//!
//! Run with:
//!
//! ```text
//! cargo run --example graph_diagram_complex
//! ```
//!
//! Outputs:
//! - `target/diagrams/graph_diagram_complex.dot`
//! - `target/diagrams/graph_diagram_complex.png`
//!
//! Requires the Graphviz `dot` command.

mod support;

use std::path::PathBuf;
use std::process::Command;

use either::Either;
use pravah::graph;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use support::ExampleError;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Request {
    seed: i64,
    hot: bool,
    multiplier: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Counter {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Decision {
    value: i64,
    hot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Multiplier {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Normalized {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SmallPath {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct HotPath {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Score {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Report {
    normalized: i64,
    score: i64,
    total: i64,
}

fn normalize(root: graph::Flow<Counter>) -> graph::Flow<Normalized> {
    root.map(|counter| Normalized {
        value: counter.value.abs(),
    })
    .map(|normalized| Normalized {
        value: normalized.value + 10,
    })
}

fn report(root: graph::Flow<Request>) -> graph::Flow<Report> {
    let (counter, decision, multiplier) = root.split(|request| {
        (
            Counter {
                value: request.seed,
            },
            Decision {
                value: request.seed,
                hot: request.hot,
            },
            Multiplier {
                value: request.multiplier,
            },
        )
    });

    let repeat = counter.mark();
    let _loop_back = counter
        .clone()
        .map(|counter| Counter {
            value: counter.value + 1,
        })
        .goto(repeat);

    let normalized = counter.flow(normalize);
    let score = decision
        .either(|decision| {
            if decision.hot {
                Either::Right(HotPath {
                    value: decision.value,
                })
            } else {
                Either::Left(SmallPath {
                    value: decision.value,
                })
            }
        })
        .branch(
            |small| {
                small.map(|small| Score {
                    value: small.value + 2,
                })
            },
            |hot| {
                hot.map(|hot| Score {
                    value: hot.value * 3,
                })
            },
        );

    normalized
        .merge((score, multiplier), |(normalized, score, multiplier)| {
            let score = score.value * multiplier.value;
            Report {
                normalized: normalized.value,
                score,
                total: normalized.value + score,
            }
        })
        .map(|mut report| {
            report.total += 1;
            report
        })
}

fn main() -> Result<(), ExampleError> {
    let flow = graph::compile(report)?;
    let diagram = graph::GraphDiagram::from_compiled_flow(&flow);
    let out_dir = PathBuf::from("target/diagrams");
    std::fs::create_dir_all(&out_dir)?;

    let dot_path = out_dir.join("graph_diagram_complex.dot");
    let png_path = out_dir.join("graph_diagram_complex.png");
    std::fs::write(&dot_path, diagram.dot())?;

    let status = Command::new("dot")
        .arg("-Tpng")
        .arg(&dot_path)
        .arg("-o")
        .arg(&png_path)
        .status()?;
    if !status.success() {
        return Err(ExampleError::unexpected(format!(
            "Graphviz dot failed with status {status}"
        )));
    }

    println!("wrote {}", dot_path.display());
    println!("wrote {}", png_path.display());
    Ok(())
}
