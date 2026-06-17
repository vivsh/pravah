//! Direct Ollama client example — exercises plain text, json_object mode, and
//! json_object mode with an explicit output schema hint.
//!
//! Run with:
//!   cargo run --example ollama_client
//!
//! The example expects Ollama to be running at http://localhost:11434 with
//! qwen3:8b (or the model you pass via OLLAMA_MODEL).

use pravah::clients::{ClientOptions, ClientOutput, Message};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── model URL ────────────────────────────────────────────────────────────
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3-coder:30b".into());
    let llm_url = format!("ollama://localhost:11434/{model}");

    // ════════════════════════════════════════════════════════════════════════
    // 1. Plain text response
    // ════════════════════════════════════════════════════════════════════════
    println!("\n══ 1. Plain text ══");
    {
        let client = ClientOptions::default()
            .with_preamble("You are a helpful assistant. Be concise.")
            .create(&llm_url)?;

        let messages = vec![Message::user(
            "What is the capital of France? One sentence.",
        )];
        let response = client.execute(&messages).await?;

        match response.output {
            ClientOutput::Output(v) => println!("output: {v}"),
            other => println!("unexpected: {other:?}"),
        }
        if let Some(u) = response.usage {
            println!("tokens: in={:?} out={:?}", u.input, u.output);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // 2. JSON output — input_schema triggers json_object mode + schema hint
    //    (no output_schema, so the model decides the shape)
    // ════════════════════════════════════════════════════════════════════════
    println!("\n══ 2. JSON object mode (no output schema) ══");
    {
        let input_schema = json!({
            "type": "object",
            "properties": {
                "topic": { "type": "string" }
            },
            "required": ["topic"]
        });

        let client = ClientOptions::default()
            .with_preamble(
                "You are a helpful assistant. \
                 Return a JSON object with a 'summary' string and a 'facts' array of strings.",
            )
            .with_input_schema(input_schema)
            .create(&llm_url)?;

        let messages = vec![Message::user(r#"{"topic": "the speed of light"}"#)];
        let response = client.execute(&messages).await?;

        match response.output {
            ClientOutput::Output(v) => {
                println!("raw output value: {}", serde_json::to_string_pretty(&v)?);
            }
            other => println!("unexpected: {other:?}"),
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // 3. JSON output — with an explicit output_schema injected as a hint
    //    This is the path that was previously broken (json_schema → empty arrays)
    // ════════════════════════════════════════════════════════════════════════
    println!("\n══ 3. JSON object mode WITH output schema hint ══");
    {
        let input_schema = json!({
            "type": "object",
            "properties": {
                "question": { "type": "string" }
            },
            "required": ["question"]
        });

        let output_schema = json!({
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Search queries that would help answer the question"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why these queries were chosen"
                }
            },
            "required": ["queries", "reasoning"]
        });

        let client = ClientOptions::default()
            .with_preamble(
                "You are a search query generator. \
                 Given a question, produce useful search queries for a web search engine.",
            )
            .with_input_schema(input_schema)
            .with_output_schema(output_schema)
            .create(&llm_url)?;

        let messages = vec![Message::user(
            r#"{"question": "What are the latest advances in fusion energy?"}"#,
        )];
        let response = client.execute(&messages).await?;

        match response.output {
            ClientOutput::Output(v) => {
                println!("raw output value: {}", serde_json::to_string_pretty(&v)?);
                // verify the field is present and non-empty
                if let Some(queries) = v.get("queries").and_then(|q| q.as_array()) {
                    println!("\nqueries ({} items):", queries.len());
                    for q in queries {
                        println!("  - {q}");
                    }
                    if queries.is_empty() {
                        eprintln!("\n[FAIL] queries array is empty — schema hint not working");
                    } else {
                        println!("\n[OK] queries populated correctly");
                    }
                } else {
                    eprintln!("\n[FAIL] 'queries' field missing from output");
                }
            }
            other => println!("unexpected: {other:?}"),
        }
    }

    Ok(())
}
