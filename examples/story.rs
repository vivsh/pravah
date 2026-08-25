//! Multi-agent comic writer example using the graph-backed runtime.
//!
//! Requires `GEMINI_API_KEY` and `FAL_KEY`, downloads generated panel images,
//! and reads a story prompt from standard input.

mod support;

use std::io::Write;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use either::Either;
use pravah::clients::Message;
use pravah::graph::{self, Agent, AgentConfig};
use pravah::{Context, FlowConf};
use rath::images::{ImageData, ImageOptions, ImageRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use support::ExampleError;

const MIN_PANEL_IMAGE_BYTES: usize = 32 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Dialogue {
    character: String,
    line: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct StoryTurn {
    panel_number: usize,
    recap: String,
    direction: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChoreographerBrief {
    panel_number: usize,
    recap: String,
    direction: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DialogueBrief {
    panel_number: usize,
    recap: String,
    direction: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CinematographerBrief {
    panel_number: usize,
    recap: String,
    direction: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ChoreoNotes {
    character_positions: String,
    movements: String,
    action_beat: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct DialogueNotes {
    lines: Vec<Dialogue>,
    subtext: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CinemaNote {
    shot_type: String,
    composition: String,
    lighting: String,
    colour_palette: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AllNotes {
    choreo: ChoreoNotes,
    dialogue: DialogueNotes,
    cinema: CinemaNote,
    panel_number: usize,
    recap: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DirectorBrief {
    panel_number: usize,
    recap: String,
    choreo: ChoreoNotes,
    dialogue: DialogueNotes,
    cinema: CinemaNote,
}
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ComicPanel {
    scene: String,
    dialogues: Vec<Dialogue>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DirectorPanel {
    panel_number: usize,
    recap: String,
    panel: ComicPanel,
    choreo: ChoreoNotes,
    dialogue: DialogueNotes,
    cinema: CinemaNote,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct IllustratorBrief {
    panel_number: usize,
    recap: String,
    panel: ComicPanel,
    choreo: ChoreoNotes,
    dialogue: DialogueNotes,
    cinema: CinemaNote,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct IllustrationPrompt {
    prompt: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct PanelRender {
    panel: DirectorPanel,
    illustration: IllustrationPrompt,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct UserInput {
    panel_number: usize,
    recap: String,
    panel: ComicPanel,
    choreo: ChoreoNotes,
    dialogue: DialogueNotes,
    cinema: CinemaNote,
    direction: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalSummary {
    panels_written: usize,
}

fn split_crew(
    turn: StoryTurn,
) -> (
    ChoreographerBrief,
    DialogueBrief,
    CinematographerBrief,
    StoryTurn,
) {
    (
        ChoreographerBrief {
            panel_number: turn.panel_number,
            recap: turn.recap.clone(),
            direction: turn.direction.clone(),
        },
        DialogueBrief {
            panel_number: turn.panel_number,
            recap: turn.recap.clone(),
            direction: turn.direction.clone(),
        },
        CinematographerBrief {
            panel_number: turn.panel_number,
            recap: turn.recap.clone(),
            direction: turn.direction.clone(),
        },
        turn,
    )
}

fn split_director(notes: AllNotes) -> (DirectorBrief, AllNotes) {
    (
        DirectorBrief {
            panel_number: notes.panel_number,
            recap: notes.recap.clone(),
            choreo: notes.choreo.clone(),
            dialogue: notes.dialogue.clone(),
            cinema: notes.cinema.clone(),
        },
        notes,
    )
}

fn merge_all_notes(
    (choreo, dialogue, cinema, turn): (ChoreoNotes, DialogueNotes, CinemaNote, StoryTurn),
) -> AllNotes {
    AllNotes {
        choreo,
        dialogue,
        cinema,
        panel_number: turn.panel_number,
        recap: turn.recap,
    }
}

fn merge_director((panel, notes): (ComicPanel, AllNotes)) -> DirectorPanel {
    DirectorPanel {
        panel_number: notes.panel_number,
        recap: notes.recap,
        panel,
        choreo: notes.choreo,
        dialogue: notes.dialogue,
        cinema: notes.cinema,
    }
}

fn split_illustrator(panel: DirectorPanel) -> (IllustratorBrief, DirectorPanel) {
    (
        IllustratorBrief {
            panel_number: panel.panel_number,
            recap: panel.recap.clone(),
            panel: ComicPanel {
                scene: panel.panel.scene.clone(),
                dialogues: panel.panel.dialogues.clone(),
            },
            choreo: panel.choreo.clone(),
            dialogue: panel.dialogue.clone(),
            cinema: panel.cinema.clone(),
        },
        panel,
    )
}

fn merge_panel_render((illustration, panel): (IllustrationPrompt, DirectorPanel)) -> PanelRender {
    PanelRender {
        panel,
        illustration,
    }
}

fn agent_config<T: Serialize>(
    input: T,
    instructions: &str,
) -> Result<AgentConfig, graph::GraphError> {
    let message = serde_json::to_string(&input).map_err(|err| graph::GraphError::JsonEncode {
        target: "story agent message".into(),
        reason: err.to_string(),
    })?;
    Ok(AgentConfig::new(
        "gemini:///gemini-2.5-flash-lite",
        instructions,
        Message::user(message),
    ))
}

async fn configure_choreographer(
    input: ChoreographerBrief,
    _ctx: Context,
) -> Result<AgentConfig, graph::GraphError> {
    agent_config(
        input,
        "You are the choreographer for a graphic novel production. You work \
             independently; the director will incorporate your notes later. \
             You receive a full recap of all prior panels, the current panel number, \
             and the director's brief for this panel. \
             Produce three fields: \
             `character_positions` — where each named character stands or sits, in \
             precise spatial terms (e.g. 'Kenji at frame-left, back to camera; Thorne \
             centred, arms crossed'); \
             `movements` — any physical motion in progress at this frozen moment \
             (a falling object, a hand mid-gesture, a body recoiling); \
             `action_beat` — the micro-moment the panel captures, with the specificity \
             of a stage direction (e.g. 'Kenji's cigarette slips from his fingers as \
             his eyes find the exit'). \
            Never write dialogue. Be spatial, physical, and precise.",
    )
}

fn choreographer(root: Agent<ChoreographerBrief>) -> Agent<ChoreoNotes> {
    root.configure(configure_choreographer)
}

async fn configure_dialogue(
    input: DialogueBrief,
    _ctx: Context,
) -> Result<AgentConfig, graph::GraphError> {
    agent_config(
        input,
        "You are the dialogue writer for a graphic novel production. You work \
             independently; the director will incorporate your lines later. \
             You receive a full recap of all prior panels, the current panel number, \
             and the director's brief for this panel. \
             Produce two fields: \
             `lines` — an array of {character, line} objects. A panel holds at most \
             2-3 short, sharp lines; a silent panel (empty array) is often more powerful. \
             Every line must feel lived-in and true to the character's established voice, \
             advance the story or reveal character, and fit the space of a comic panel. \
             Never repeat any line from the recap verbatim. \
             `subtext` — one private sentence describing what is emotionally unsaid \
             beneath the dialogue (or beneath the silence). This note is for the \
             director only and will never be printed.",
    )
}

fn dialogue_writer(root: Agent<DialogueBrief>) -> Agent<DialogueNotes> {
    root.configure(configure_dialogue)
}

async fn configure_cinematographer(
    input: CinematographerBrief,
    _ctx: Context,
) -> Result<AgentConfig, graph::GraphError> {
    agent_config(
        input,
        "You are the cinematographer for a graphic novel production. You work \
             independently; the director will incorporate your framing later. \
             You receive a full recap of all prior panels, the current panel number, \
             and the director's brief for this panel. \
             Produce four fields: \
             `shot_type` — e.g. extreme close-up, over-the-shoulder, low-angle wide \
             shot, bird's-eye, dutch angle, two-shot; \
             `composition` — where subjects sit in the frame, use of negative space, \
             foreground/background layering, leading lines that guide the eye; \
             `lighting` — quality (hard/soft), direction, colour temperature, and the \
             emotional mood the light creates; \
             `colour_palette` — 2-3 dominant hues and their symbolic or emotional \
             resonance for this moment in the story. \
             Never describe character actions or write dialogue. Think purely in images.",
    )
}

fn cinematographer(root: Agent<CinematographerBrief>) -> Agent<CinemaNote> {
    root.configure(configure_cinematographer)
}

async fn configure_director(
    input: DirectorBrief,
    _ctx: Context,
) -> Result<AgentConfig, graph::GraphError> {
    agent_config(
        input,
        "You are the director of a graphic novel. Your crew has delivered independent \
             notes for the next panel: a choreographer (character positions, movement, \
             action beat), a dialogue writer (spoken lines and private subtext), and a \
             cinematographer (shot type, composition, lighting, colour palette). \
             Synthesise all three departments into one cohesive final panel. Where \
             departments conflict, make the strongest storytelling choice and commit. \
             The recap shows every prior panel in full production detail — honour \
             continuity of character voice, physical space, and emotional arc. \
             Produce two fields: \
             `scene` — a single vivid sentence an illustrator will draw from. Fuse the \
             cinematographer's framing, the choreographer's physical beat, and the \
             emotional temperature of the moment into one image. Do not list or quote \
             crew notes — distil them into one authoritative visual statement. \
             `dialogues` — the final approved script as an array of {character, line} \
             objects. Honour the dialogue writer's intent; trim or go silent if the \
             scene already speaks.",
    )
}

fn director(root: Agent<DirectorBrief>) -> Agent<ComicPanel> {
    root.configure(configure_director)
}

async fn configure_illustrator(
    input: IllustratorBrief,
    _ctx: Context,
) -> Result<AgentConfig, graph::GraphError> {
    agent_config(
        input,
        "You are the production illustrator prompt designer for a professional comic book. \
             You receive the final director panel plus choreography, dialogue, subtext, camera, \
             lighting, colour palette, and recap continuity. \
             Produce exactly one image-generation prompt in the `prompt` field. \
             The prompt must describe a finished comic-book panel, not a generic illustration. \
             It must include: panel composition, character placement, camera angle, lighting, \
             colour palette, facial expression, action beat, speech bubbles, and one narrator \
             caption box. \
             Keep all visible text short enough to fit in the artwork: each speech bubble line \
             should be under 8 words, and the narrator caption under 12 words. If dialogue is \
             too long, adapt it into shorter comic lettering while preserving meaning. \
             Specify professional lettering: clean readable comic font, bubbles placed with \
             clear tails, caption boxes aligned to avoid covering faces or action. \
             Avoid black backgrounds, blank frames, silhouette-only compositions, or dark empty \
             space; every panel must be bright enough to inspect the characters and lettering. \
             Ask for a polished sequential-art page-panel style with crisp ink, strong shapes, \
             cinematic colour, and no extra text beyond the requested bubbles/captions.",
    )
}

fn illustrator(root: Agent<IllustratorBrief>) -> Agent<IllustrationPrompt> {
    root.configure(configure_illustrator)
}

async fn generate_panel_image(
    render: PanelRender,
    _ctx: Context,
) -> Result<DirectorPanel, graph::GraphError> {
    let dir = Path::new("target/story_panels");
    tokio::fs::create_dir_all(dir).await.map_err(|err| {
        graph::GraphError::Invalid(format!("failed to create image folder: {err}"))
    })?;
    let prompt_path = dir.join(format!("panel_{:03}.prompt.txt", render.panel.panel_number));
    tokio::fs::write(&prompt_path, &render.illustration.prompt)
        .await
        .map_err(|err| {
            graph::GraphError::Invalid(format!(
                "failed to write panel prompt '{}': {err}",
                prompt_path.display()
            ))
        })?;
    println!(
        "\n--- Illustrator prompt for panel {} ---\n{}\n--- end prompt ---\n",
        render.panel.panel_number, render.illustration.prompt
    );

    let client = ImageOptions {
        provider_config: Some(serde_json::json!({
            "num_images": 1,
            "enable_safety_checker": false,
            "output_format": "jpeg"
        })),
    }
    .create("fal:///fal-ai/flux/schnell")
    .map_err(|err| {
        graph::GraphError::Invalid(format!("failed to create Fal image client: {err}"))
    })?;

    let mut prompt = render.illustration.prompt.clone();
    let mut last_error = None;
    for attempt in 1..=2 {
        let response = client
            .generate_image(&ImageRequest {
                prompt: prompt.clone(),
                size: Some("landscape_4_3".to_string()),
                provider_config: Some(serde_json::json!({
                    "num_images": 1,
                    "enable_safety_checker": false,
                    "output_format": "jpeg"
                })),
                ..ImageRequest::default()
            })
            .await
            .map_err(|err| {
                graph::GraphError::Invalid(format!("failed to generate panel image: {err}"))
            })?;

        save_image_metadata(
            dir,
            render.panel.panel_number,
            attempt,
            &response.raw_metadata,
        )
        .await?;
        let image = response.images.first().ok_or_else(|| {
            graph::GraphError::Invalid("Fal image response did not include any images".to_string())
        })?;
        match save_image_data(dir, render.panel.panel_number, image).await {
            Ok(path) => {
                println!("panel image: {}", path.display());
                return Ok(render.panel);
            }
            Err(err) if attempt == 1 => {
                last_error = Some(err);
                prompt = fallback_panel_prompt(&render.panel);
                let retry_prompt_path = dir.join(format!(
                    "panel_{:03}.retry_prompt.txt",
                    render.panel.panel_number
                ));
                tokio::fs::write(&retry_prompt_path, &prompt)
                    .await
                    .map_err(|err| {
                        graph::GraphError::Invalid(format!(
                            "failed to write retry prompt '{}': {err}",
                            retry_prompt_path.display()
                        ))
                    })?;
            }
            Err(err) => {
                return Err(graph::GraphError::Invalid(format!(
                    "{err}; prompt saved at '{}'",
                    prompt_path.display()
                )));
            }
        }
    }

    let err = last_error
        .map(|err| err.to_string())
        .unwrap_or_else(|| "image generation did not produce a usable image".to_string());
    Err(graph::GraphError::Invalid(format!(
        "{err}; prompt saved at '{}'",
        prompt_path.display()
    )))
}

fn fallback_panel_prompt(panel: &DirectorPanel) -> String {
    let dialogue = panel
        .panel
        .dialogues
        .iter()
        .map(|line| format!("{} says: \"{}\"", line.character, line.line))
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "Professional full-colour comic book panel, bright visible interior lighting, no black \
         screen, no blank background. Scene: {} Characters and action: {} {} Camera: {}. \
         Composition: {}. Palette: {}. Add clean speech bubbles with short readable text: {}. \
         Add one small narrator caption box with fewer than 12 words. Crisp ink lines, polished \
         sequential art, expressive faces, clear character silhouettes.",
        panel.panel.scene,
        panel.choreo.character_positions,
        panel.choreo.action_beat,
        panel.cinema.shot_type,
        panel.cinema.composition,
        panel.cinema.colour_palette,
        dialogue
    )
}

async fn save_image_metadata(
    dir: &Path,
    panel_number: usize,
    attempt: usize,
    metadata: &Option<serde_json::Value>,
) -> Result<(), graph::GraphError> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    let path = dir.join(format!("panel_{panel_number:03}.attempt_{attempt}.json"));
    let data = serde_json::to_vec_pretty(metadata).map_err(|err| {
        graph::GraphError::Invalid(format!("failed to encode image metadata: {err}"))
    })?;
    tokio::fs::write(&path, data).await.map_err(|err| {
        graph::GraphError::Invalid(format!(
            "failed to write image metadata '{}': {err}",
            path.display()
        ))
    })?;
    Ok(())
}

fn validate_image_bytes(path: &Path, bytes_len: usize) -> Result<(), graph::GraphError> {
    if bytes_len < MIN_PANEL_IMAGE_BYTES {
        return Err(graph::GraphError::Invalid(format!(
            "generated image '{}' is only {} bytes; treating it as a likely blank/safety-filtered panel",
            path.display(),
            bytes_len
        )));
    }
    Ok(())
}

async fn save_image_data(
    dir: &Path,
    panel_number: usize,
    image: &ImageData,
) -> Result<PathBuf, graph::GraphError> {
    match image {
        ImageData::Url { url } => save_image_url(dir, panel_number, url).await,
        ImageData::Base64 { mime_type, data } => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|err| {
                    graph::GraphError::Invalid(format!("invalid base64 image data: {err}"))
                })?;
            let path = dir.join(format!(
                "panel_{panel_number:03}.{}",
                image_extension(mime_type)
            ));
            validate_image_bytes(&path, bytes.len())?;
            tokio::fs::write(&path, bytes).await.map_err(|err| {
                graph::GraphError::Invalid(format!(
                    "failed to write panel image '{}': {err}",
                    path.display()
                ))
            })?;
            Ok(path)
        }
    }
}

async fn save_image_url(
    dir: &Path,
    panel_number: usize,
    url: &str,
) -> Result<PathBuf, graph::GraphError> {
    let response = reqwest::get(url).await.map_err(|err| {
        graph::GraphError::Invalid(format!("failed to download panel image: {err}"))
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(graph::GraphError::Invalid(format!(
            "panel image download failed with status {status}"
        )));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("image/png")
        .to_string();
    let bytes = response.bytes().await.map_err(|err| {
        graph::GraphError::Invalid(format!("failed to read panel image bytes: {err}"))
    })?;
    let path = dir.join(format!(
        "panel_{panel_number:03}.{}",
        image_extension(&content_type)
    ));
    validate_image_bytes(&path, bytes.len())?;
    tokio::fs::write(&path, bytes).await.map_err(|err| {
        graph::GraphError::Invalid(format!(
            "failed to write panel image '{}': {err}",
            path.display()
        ))
    })?;
    Ok(path)
}

fn image_extension(content_type: &str) -> &'static str {
    match content_type.split(';').next().unwrap_or("").trim() {
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "png",
    }
}

async fn print_and_read(dp: DirectorPanel, _ctx: Context) -> Result<UserInput, graph::GraphError> {
    let rule = "─".repeat(62);
    println!("\n{rule}");
    println!("  PANEL {}", dp.panel_number);
    println!("{rule}");
    println!("SCENE: {}", dp.panel.scene);
    if !dp.panel.dialogues.is_empty() {
        println!();
        for d in &dp.panel.dialogues {
            println!("  [{:>14}]  \"{}\"", d.character, d.line);
        }
    }
    println!();
    println!("  [Choreographer]  {}", dp.choreo.action_beat);
    println!(
        "  [Cinematographer] {} · {}",
        dp.cinema.shot_type, dp.cinema.lighting
    );
    println!("{rule}");

    let direction = tokio::task::spawn_blocking(|| {
        print!("\nDirector's note for next panel (or 'exit'): ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        line.trim().to_string()
    })
    .await
    .map_err(|e| graph::GraphError::Invalid(format!("print_and_read failed: {e}")))?;

    Ok(UserInput {
        panel_number: dp.panel_number,
        recap: dp.recap,
        panel: dp.panel,
        choreo: dp.choreo,
        dialogue: dp.dialogue,
        cinema: dp.cinema,
        direction,
    })
}

fn route_input(input: UserInput) -> Either<StoryTurn, FinalSummary> {
    if input.direction.eq_ignore_ascii_case("exit") {
        return Either::Right(FinalSummary {
            panels_written: input.panel_number,
        });
    }

    let dialogue_text = if input.panel.dialogues.is_empty() {
        "  (silent panel)".to_string()
    } else {
        input
            .panel
            .dialogues
            .iter()
            .map(|d| format!("  [{}]: \"{}\"", d.character, d.line))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let panel_entry = format!(
        "=== Panel {} ===\n\
         SCENE:    {}\n\
         STAGING:  {} — {}\n\
         SHOT:     {} | {} | {}\n\
         PALETTE:  {}\n\
         SUBTEXT:  {}\n\
         {}",
        input.panel_number,
        input.panel.scene,
        input.choreo.character_positions,
        input.choreo.action_beat,
        input.cinema.shot_type,
        input.cinema.composition,
        input.cinema.lighting,
        input.cinema.colour_palette,
        input.dialogue.subtext,
        dialogue_text,
    );

    let recap = if input.recap.is_empty() {
        panel_entry
    } else {
        format!("{}\n\n{}", input.recap, panel_entry)
    };

    let direction = if input.direction.is_empty() {
        "Continue the story naturally, advancing plot and character.".to_string()
    } else {
        input.direction
    };

    Either::Left(StoryTurn {
        panel_number: input.panel_number + 1,
        recap,
        direction,
    })
}

fn story_turn(root: graph::Flow<StoryTurn>) -> graph::Flow<UserInput> {
    let (choreo, dialogue, cinema, carry) = root.split(split_crew);

    let (director_brief, director_carry) = choreo
        .agent(choreographer)
        .merge(
            (
                dialogue.agent(dialogue_writer),
                cinema.agent(cinematographer),
                carry,
            ),
            merge_all_notes,
        )
        .split(split_director);

    let (illustrator_brief, panel) = director_brief
        .agent(director)
        .merge(director_carry, merge_director)
        .split(split_illustrator);

    illustrator_brief
        .agent(illustrator)
        .merge(panel, merge_panel_render)
        .work(generate_panel_image)
        .work(print_and_read)
}

async fn run_story_turn(
    flow: &graph::CompiledFlow<StoryTurn, UserInput>,
    turn: StoryTurn,
    ctx: Context,
) -> Result<UserInput, ExampleError> {
    let mut runtime = flow.runtime(turn)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            graph::Step::Continue => {}
            graph::Step::Done(value) => return Ok(flow.decode_output(value)?),
            graph::Step::Suspend(_) => {
                return Err(ExampleError::unexpected(
                    "story example does not expect suspension",
                ));
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    dotenvy::dotenv().ok();

    let rule = "═".repeat(62);
    println!("{rule}");
    println!("   PRODUCTION ROOM  —  Multi-Agent Comic Writer");
    println!("   Crew: Choreographer · Dialogue Writer · Cinematographer · Director");
    println!("{rule}");
    println!();
    print!("Describe your story (genre, setting, characters, opening situation):\n> ");
    std::io::stdout().flush()?;

    let mut initial = String::new();
    std::io::stdin().read_line(&mut initial)?;
    let initial = initial.trim().to_string();

    if initial.is_empty() {
        eprintln!("No story provided. Exiting.");
        return Ok(());
    }

    println!("\nProduction starting — Panel 1 in progress...\n");

    let ctx = Context::new(FlowConf::default());
    let flow = graph::compile(story_turn)?;
    let mut turn = StoryTurn {
        panel_number: 1,
        recap: String::new(),
        direction: initial,
    };

    loop {
        let input = run_story_turn(&flow, turn, ctx.clone()).await?;
        match route_input(input) {
            Either::Left(next) => {
                println!(
                    "\nProduction continuing — Panel {} in progress...\n",
                    next.panel_number
                );
                turn = next;
            }
            Either::Right(summary) => {
                println!("\n{rule}");
                println!(
                    "  Story complete — {} panel(s) written.",
                    summary.panels_written
                );
                println!("{rule}");
                break;
            }
        }
    }

    Ok(())
}
