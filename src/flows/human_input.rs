use either::Either;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::context::Context;
use crate::flows::errors::FlowError;
use crate::flows::flows::{Flow, FlowGraph};

/// A single selectable option presented to the human.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Choice {
    /// Short label displayed to the human and used by the LLM to identify the option.
    pub label: String,
    /// Optional elaboration shown below the label to help the human decide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional emoji or icon identifier (e.g. `"✅"`, `"warning"`) shown alongside the label.
    /// Renders in both CLI and web UIs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    /// Optional URL of a preview image. Ignored by the CLI renderer; web callers
    /// can display it when rendering the suspended [`PendingHumanInput`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

/// A prompt with optional choices sent to a human for review or decision.
///
/// When `choices` is empty the human is asked to type a free-text answer.
/// When `choices` is non-empty and `allow_other` is `true`, the human may
/// either select a choice or type a custom answer.
///
/// Embed in an agent toolbox with `ToolBox::flow::<HumanInput>()`.
/// Place [`CliMode`] in [`Context`] deps to read from stdin; otherwise the
/// flow suspends so a web handler can supply the answer via `resume()`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(
    description = "A question or decision point presented to a human. Set `choices` to offer \
    labelled options; omit them for free-text. Set `allow_other` to accept free-text alongside \
    choices."
)]
pub struct HumanInput {
    /// The question, instruction, or context to display.
    pub prompt: String,
    /// Ordered list of labelled options. Empty means free-text only.
    #[schemars(
        description = "Zero or more options. Index 0 = first item. Omit or leave empty to ask \
        for a free-text answer."
    )]
    pub choices: Vec<Choice>,
    /// When `true`, the human may type a custom answer instead of (or in addition to) a choice.
    pub allow_other: bool,
}

/// The human's response to a [`HumanInput`] prompt.
///
/// Exactly one of `choice` or `text` is set.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(
    description = "The human's answer. `choice` holds the zero-based index of a selected option. \
    `text` holds a free-text reply. Exactly one field is populated."
)]
pub struct HumanOutput {
    /// Zero-based index of the selected choice. `None` when the human typed a free-text answer.
    pub choice: Option<usize>,
    /// Free-text answer. Present when `allow_other` is `true` or `choices` is empty.
    pub text: Option<String>,
}

/// Marker placed in [`Context`] deps to enable stdin interaction.
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

/// Wraps a [`HumanInput`] on the suspend path so its node key differs from `"HumanInput"`.
///
/// In web mode, downcast a [`crate::flows::SuspendedValue`] to this type to
/// inspect the pending prompt, then call `resume()` with a [`HumanOutput`].
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PendingHumanInput(pub HumanInput);

/// Internal routing type — not part of the public API.
///
/// `Either<HumanOutput, PendingHumanInput>` cannot be used directly because the
/// `either` crate only implements `JsonSchema` for schemars v1 while this crate
/// uses v0.8.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum HumanInputDecision {
    Done(HumanOutput),
    Pending(PendingHumanInput),
}

impl Flow for HumanInput {
    type Output = HumanOutput;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(try_cli_input)
            .either(route_decision)
            .suspend::<PendingHumanInput, HumanOutput>()
            .build()
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
