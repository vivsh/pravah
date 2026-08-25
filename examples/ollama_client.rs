//! Direct Ollama client example for text and structured JSON output.
//!
//! Run with `cargo run --example ollama_client`. The example expects Ollama at
//! `http://localhost:11434` and uses `OLLAMA_MODEL` when it is set.

mod support;

use pravah::clients::{ClientOptions, ClientOutput, Message};
use serde_json::{Value, json};
use support::ExampleError;

fn require_output(output: ClientOutput, label: &str) -> Result<Value, ExampleError> {
    match output {
        ClientOutput::Output(value) => Ok(value),
        other => Err(ExampleError::unexpected(format!(
            "{label} returned {other:?}"
        ))),
    }
}

async fn run_text(llm_url: &str) -> Result<(), ExampleError> {
    println!("\n1. Plain text");
    let client = ClientOptions::default()
        .with_preamble("You are a helpful assistant. Be concise.")
        .create(llm_url)?;
    let messages = vec![Message::user(
        "What is the capital of France? One sentence.",
    )];
    let response = client.execute(&messages).await?;
    let output = require_output(response.output, "plain text request")?;
    println!("output: {output}");
    if let Some(usage) = response.usage {
        println!("tokens: in={:?} out={:?}", usage.input, usage.output);
    }
    Ok(())
}

async fn run_json_object(llm_url: &str) -> Result<(), ExampleError> {
    println!("\n2. JSON object mode");
    let input_schema = json!({
        "type": "object",
        "properties": { "topic": { "type": "string" } },
        "required": ["topic"]
    });
    let client = ClientOptions::default()
        .with_preamble(
            "Return a JSON object with a 'summary' string and a 'facts' array of strings.",
        )
        .with_input_schema(input_schema)
        .create(llm_url)?;
    let response = client
        .execute(&[Message::user(r#"{"topic": "the speed of light"}"#)])
        .await?;
    let output = require_output(response.output, "JSON object request")?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn query_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Search queries that would help answer the question"
            },
            "reasoning": { "type": "string" }
        },
        "required": ["queries", "reasoning"]
    })
}

fn print_queries(output: &Value) -> Result<(), ExampleError> {
    let queries = output
        .get("queries")
        .and_then(Value::as_array)
        .ok_or_else(|| ExampleError::unexpected("schema output omitted the queries array"))?;
    if queries.is_empty() {
        return Err(ExampleError::unexpected(
            "schema output returned an empty queries array",
        ));
    }
    println!("\nqueries ({} items):", queries.len());
    for query in queries {
        println!("- {query}");
    }
    Ok(())
}

async fn run_schema_output(llm_url: &str) -> Result<(), ExampleError> {
    println!("\n3. JSON Schema output");
    let input_schema = json!({
        "type": "object",
        "properties": { "question": { "type": "string" } },
        "required": ["question"]
    });
    let client = ClientOptions::default()
        .with_preamble("Produce useful web-search queries for the supplied question.")
        .with_input_schema(input_schema)
        .with_output_schema(query_schema())
        .create(llm_url)?;
    let response = client
        .execute(&[Message::user(
            r#"{"question": "What are the latest advances in fusion energy?"}"#,
        )])
        .await?;
    let output = require_output(response.output, "JSON Schema request")?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    print_queries(&output)
}

#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3-coder:30b".into());
    let llm_url = format!("ollama:///{model}?base_url=http://localhost:11434");
    run_text(&llm_url).await?;
    run_json_object(&llm_url).await?;
    run_schema_output(&llm_url).await
}
