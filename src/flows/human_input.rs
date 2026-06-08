use either::Either;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::context::Context;
use crate::flows::{Flow, Node};
use crate::flows::errors::FlowError;
use crate::tools::ToolOutput;

/// One option shown to the human.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Choice {
    /// Label shown to the human.
    pub label: String,
    /// Optional helper text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional icon hint for CLI and web renderers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    /// Optional preview image URL for web renderers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

/// Prompt sent to a human.
/// An empty `choices` list means free text.
/// When `allow_other` is true, the human may type a custom answer even if choices exist.
/// Put [`CliMode`] in [`Context`] when the answer should be read from stdin.
/// Without it, the flow suspends and waits for `resume()`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(
    description = "A question or decision point presented to a human. Set `choices` to offer \
    labelled options; omit them for free-text. Set `allow_other` to accept free-text alongside \
    choices."
)]
pub struct HumanInput {
    /// Prompt text.
    pub prompt: String,
    /// Ordered options.
    #[schemars(
        description = "Zero or more options. Index 0 = first item. Omit or leave empty to ask \
        for a free-text answer."
    )]
    pub choices: Vec<Choice>,
    /// Allows a custom text answer.
    pub allow_other: bool,
}

/// Human response to a [`HumanInput`] prompt.
/// Exactly one of `choice` or `text` should be present.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(
    description = "The human's answer. `choice` holds the zero-based index of a selected option. \
    `text` holds a free-text reply. Exactly one field is populated."
)]
pub struct HumanOutput {
    /// Zero-based choice index.
    pub choice: Option<usize>,
    /// Free-text answer.
    pub text: Option<String>,
}

impl ToolOutput for HumanOutput {}

/// Marker dependency that enables stdin input.
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use pravah::{Context, CliMode};
/// use pravah::deps::Deps;
///
/// let mut deps = Deps::default();
/// deps.insert(Arc::new(CliMode));
/// let ctx = Context::default().with_deps(deps);
/// ```
pub struct CliMode;

/// Wrapper used on the suspend path so the pending prompt has its own node type.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PendingHumanInput(pub HumanInput);

/// Internal routing type.
/// `Either<HumanOutput, PendingHumanInput>` cannot be used directly because the
/// `either` crate exposes the wrong `JsonSchema` version for this crate.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum HumanInputDecision {
    Done(HumanOutput),
    Pending(PendingHumanInput),
}

impl Flow for HumanInput {
    type Output = HumanOutput;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root
            .work(try_cli_input)
            .either(route_decision)
            .branch(|done| done, |pending| pending.suspend::<HumanOutput>())
    }
}

async fn try_cli_input(
    input: HumanInput,
    ctx: Context,
) -> Result<HumanInputDecision, FlowError> {
    if ctx.require::<CliMode>().is_ok() {
        let out = read_stdin(input).await?;
        Ok(HumanInputDecision::Done(out))
    } else {
        Ok(HumanInputDecision::Pending(PendingHumanInput(input)))
    }
}

fn route_decision(d: HumanInputDecision) -> Either<HumanOutput, PendingHumanInput> {
    match d {
        HumanInputDecision::Done(out) => Either::Left(out),
        HumanInputDecision::Pending(p) => Either::Right(p),
    }
}

async fn read_stdin(input: HumanInput) -> Result<HumanOutput, FlowError> {
    tokio::task::spawn_blocking(move || read_stdin_blocking(input))
        .await
        .map_err(|e| FlowError::Internal {
            handler: "human_input",
            detail: e.to_string(),
        })?
}

fn read_stdin_blocking(input: HumanInput) -> Result<HumanOutput, FlowError> {
    use std::io::BufRead;
    loop {
        print_prompt(&input);
        let mut buf = String::new();
        std::io::stdin()
            .lock()
            .read_line(&mut buf)
            .map_err(|e| FlowError::Internal {
                handler: "human_input",
                detail: e.to_string(),
            })?;
        let line = buf.trim();
        match parse_response(line, input.choices.len(), input.allow_other) {
            Ok(out) => return Ok(out),
            Err(e) => eprintln!("  Invalid input ({e}). Please try again."),
        }
    }
}

fn print_prompt(input: &HumanInput) {
    use std::io::Write;
    println!("\n{}", input.prompt);
    for (i, choice) in input.choices.iter().enumerate() {
        let icon = choice.icon.as_deref().map(|s| format!("{s} ")).unwrap_or_default();
        println!("  {}. {}{}", i + 1, icon, choice.label);
        if let Some(desc) = &choice.description {
            println!("     {desc}");
        }
    }
    if input.allow_other || input.choices.is_empty() {
        if !input.choices.is_empty() {
            println!("  (or type a custom answer)");
        }
    }
    print!("> ");
    let _ = std::io::stdout().flush();
}

fn parse_response(line: &str, choices_len: usize, allow_other: bool) -> Result<HumanOutput, FlowError> {
    if line.is_empty() {
        return Err(FlowError::Internal {
            handler: "human_input",
            detail: "empty input".into(),
        });
    }
    if choices_len == 0 {
        return Ok(HumanOutput { choice: None, text: Some(line.to_owned()) });
    }
    if let Ok(n) = line.parse::<usize>() {
        if n >= 1 && n <= choices_len {
            return Ok(HumanOutput { choice: Some(n - 1), text: None });
        }
    }
    if allow_other {
        return Ok(HumanOutput { choice: None, text: Some(line.to_owned()) });
    }
    Err(FlowError::Internal {
        handler: "human_input",
        detail: format!("expected a number between 1 and {choices_len}"),
    })
}
