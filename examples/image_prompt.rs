//! User-image example using `Agent::to_message` with a file attachment.

use argh::FromArgs;
use std::env;
use std::path::{Path, PathBuf};
use std::process;

use pravah::clients::{Attachment, Message, Role};
use pravah::flows::{Agent, AgentConfig, Flow, FlowBuilder, FlowError, FlowRuntime, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;


#[derive(Debug, Error)]
enum ExampleError {
    #[error("{0}")]
    Args(String),
    #[error("image_prompt only accepts image mime types, got '{0}'")]
    NonImageMime(String),
    #[error("unable to infer image mime type from '{0}'; pass --mime_type explicitly")]
    MissingMime(String),
    #[error("input image '{0}' does not exist")]
    MissingImage(String),
    #[error("input image '{0}' must point to a file")]
    InvalidImagePath(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Flow(#[from] FlowError),
}


#[derive(Debug, PartialEq, Eq)]
struct CliArgs {
    prompt: String,
    image_path: PathBuf,
    mime_type: String,
}

/// Upload an image as part of the initial user prompt.
#[derive(Debug, FromArgs, PartialEq, Eq)]
struct ImagePromptArgs {
    /// image path to upload with the initial user message
    #[argh(option)]
    image_path: PathBuf,

    /// image mime type; inferred from the file extension when omitted
    #[argh(option)]
    mime_type: Option<String>,

    /// prompt to send alongside the uploaded image
    #[argh(positional)]
    prompt: Vec<String>,
}


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct VisionPrompt {
    prompt: String,
    image_path: String,
    mime_type: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct VisionResult {
    summary: String,
    visible_text: Vec<String>,
    confidence: String,
}


impl Agent for VisionPrompt {
    type Output = VisionResult;

    fn to_message(self, _ctx: &Context) -> Result<Message, FlowError> {
        Ok(Message {
            role: Role::User,
            content: self.prompt,
            attachments: vec![Attachment::File {
                mime_type: self.mime_type,
                path: self.image_path,
            }],
            usage: None,
        })
    }

    fn build() -> AgentConfig {
        let model_url = env::var("PRAVAH_MODEL_URL")
            .unwrap_or_else(|_| "gemini:///gemini-2.5-flash-lite".to_string());
        AgentConfig::new(
            "You are a vision assistant. Inspect the uploaded image and return JSON with: \
             1. `summary`: one concise description of what the image shows, \
             2. `visible_text`: short OCR strings that are clearly readable, \
             3. `confidence`: one of high, medium, or low.",
            model_url,
        )
    }
}


impl Flow for VisionPrompt {
    type Output = VisionResult;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder.agent::<VisionPrompt>()
    }
}


fn parse_args() -> Result<CliArgs, ExampleError> {
    match parse_raw_args(env::args().skip(1)) {
        Ok(parsed) => validate_args(parsed),
        Err(early_exit) => match early_exit.status {
            Ok(()) => {
                println!("{}", early_exit.output);
                process::exit(0);
            }
            Err(()) => Err(ExampleError::Args(early_exit.output)),
        },
    }
}


fn parse_args_from<I>(args: I) -> Result<CliArgs, ExampleError>
where
    I: IntoIterator<Item = String>,
{
    let parsed = parse_raw_args(args).map_err(|early_exit| ExampleError::Args(early_exit.output))?;
    validate_args(parsed)
}


fn parse_raw_args<I>(args: I) -> Result<ImagePromptArgs, argh::EarlyExit>
where
    I: IntoIterator<Item = String>,
{
    let args = normalize_args(args.into_iter().collect::<Vec<_>>());
    let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    ImagePromptArgs::from_args(&["image_prompt"], &refs)
}


fn normalize_args(args: Vec<String>) -> Vec<String> {
    args.into_iter()
        .map(|arg| match arg.as_str() {
            "--image_path" => "--image-path".to_string(),
            "--mime_type" => "--mime-type".to_string(),
            _ => arg,
        })
        .collect()
}


fn validate_args(args: ImagePromptArgs) -> Result<CliArgs, ExampleError> {
    let prompt = args.prompt.join(" ");
    if prompt.is_empty() {
        return Err(ExampleError::Args(
            "missing required positional argument <prompt>".to_string(),
        ));
    }

    let image_path = args.image_path;
    if !image_path.exists() {
        return Err(ExampleError::MissingImage(image_path.display().to_string()));
    }
    let mime_type = match args.mime_type {
        Some(mime_type) => mime_type,
        None => infer_image_mime(&image_path)
            .ok_or_else(|| ExampleError::MissingMime(image_path.display().to_string()))?,
    };
    if !mime_type.starts_with("image/") {
        return Err(ExampleError::NonImageMime(mime_type));
    }

    Ok(CliArgs {
        prompt,
        image_path,
        mime_type,
    })
}


fn infer_image_mime(path: &Path) -> Option<String> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    let mime_type = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "svg" => "image/svg+xml",
        "heic" => "image/heic",
        "heif" => "image/heif",
        "avif" => "image/avif",
        _ => return None,
    };
    Some(mime_type.to_string())
}


fn prepare_runtime(args: CliArgs) -> Result<(VisionPrompt, Context), ExampleError> {
    let image_path = std::fs::canonicalize(&args.image_path)?;
    let working_dir = image_path
        .parent()
        .ok_or_else(|| ExampleError::InvalidImagePath(image_path.display().to_string()))?
        .to_path_buf();
    let image_name = image_path
        .file_name()
        .ok_or_else(|| ExampleError::InvalidImagePath(image_path.display().to_string()))?
        .to_string_lossy()
        .into_owned();

    let input = VisionPrompt {
        prompt: args.prompt,
        image_path: image_name,
        mime_type: args.mime_type,
    };
    let ctx = Context::new(FlowConf {
        working_dir: Some(working_dir),
        ..Default::default()
    });
    Ok((input, ctx))
}


#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    dotenvy::dotenv().ok();

    let args = parse_args()?;
    let (input, ctx) = prepare_runtime(args)?;
    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(result) => {
                println!("Summary: {}", result.summary);
                println!("Confidence: {}", result.confidence);
                if result.visible_text.is_empty() {
                    println!("Visible text: <none>");
                } else {
                    println!("Visible text:");
                    for line in result.visible_text {
                        println!("- {line}");
                    }
                }
                break;
            }
            FlowStep::Suspend(_) => return Err(FlowError::Internal {
                handler: "image_prompt_example",
                detail: "unexpected suspension in a single-agent example".into(),
            }
            .into()),
        }
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Flag-style args infer the mime type from the image extension when omitted.
    #[test]
    fn parse_args_from_flags_infers_mime() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("portrait.png");
        std::fs::write(&path, b"png").expect("image file should be written");

        let args = parse_args_from(vec![
            "--image_path".to_string(),
            path.display().to_string(),
            "who".to_string(),
            "is".to_string(),
            "this?".to_string(),
        ])
        .expect("flag args should parse");

        assert_eq!(args.image_path, path);
        assert_eq!(args.mime_type, "image/png");
        assert_eq!(args.prompt, "who is this?");
    }

    /// Explicit mime flags are accepted alongside a trailing positional prompt.
    #[test]
    fn parse_args_from_flags_accepts_explicit_mime() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("portrait.jpg");
        std::fs::write(&path, b"jpg").expect("image file should be written");

        let args = parse_args_from(vec![
            "--image_path".to_string(),
            path.display().to_string(),
            "--mime_type".to_string(),
            "image/jpeg".to_string(),
            "describe".to_string(),
            "the".to_string(),
            "photo".to_string(),
        ])
        .expect("flag args should parse");

        assert_eq!(args.mime_type, "image/jpeg");
        assert_eq!(args.prompt, "describe the photo");
    }

    /// Preparing the runtime rebases the attachment path under the image's parent directory.
    #[test]
    fn prepare_runtime_uses_image_parent_as_working_dir() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("guru_nanak.png");
        std::fs::write(&path, b"png").expect("image file should be written");

        let (input, ctx) = prepare_runtime(CliArgs {
            prompt: "who is this?".to_string(),
            image_path: path,
            mime_type: "image/png".to_string(),
        })
        .expect("runtime preparation should succeed");

        let canonical_dir = std::fs::canonicalize(dir.path())
            .expect("tempdir should canonicalize");
        assert_eq!(ctx.working_dir(), canonical_dir);
        assert_eq!(input.image_path, "guru_nanak.png");
    }
}