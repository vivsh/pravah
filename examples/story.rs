//! # Example — Production Room: Multi-Agent Comic Writer
//!
//! Each panel is produced by three independent specialist agents whose notes
//! are synthesised by a director. The story grows one panel at a time; the
//! user steers the plot after each panel. The entire loop lives inside a single
//! [`FlowRuntime`] — the `either` node loops back to `StoryTurn`.
//!
//! ```text
//! StoryTurn ─split─┬─ ChoreographerBrief ──agent──► ChoreoNotes ──┐
//!                  ├─ DialogueBrief ──────────agent──► DialogueNotes ├── merge ──► AllNotes
//!                  ├─ CinematographerBrief ───agent──► CinemaNote ───┘
//!                  └─ TurnCarry ─────────────────────────────────────┘
//!
//! AllNotes ─split─┬─ DirectorBrief ──agent──► ComicPanel ─┐
//!                 └─ DirectorCarry ─────────────────────────── merge ──► DirectorPanel
//!
//! DirectorPanel ──work (print + read stdin)──► UserInput
//!                                                  │
//!                          either ◄────────────────┘
//!                         /       \
//!          Left: StoryTurn          Right: FinalSummary (terminal)
//!          (loops back ↑)
//! ```
//!
//! The `recap` field in `StoryTurn` is a rich text log built after every panel:
//! scene, staging, cinematography, subtext, and dialogue. Each specialist
//! agent receives the full recap so continuity is maintained across turns.
//!
//! ## Running
//!
//! ```shell
//! GEMINI_API_KEY=<key> cargo run --example story
//! ```

use std::io::Write;

use either::Either;
use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Shared primitives ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Dialogue {
    character: String,
    line: String,
}

// ── Flow entry and loop-back type ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct StoryTurn {
    panel_number: usize,
    /// Rich text log of every prior panel's scene, staging, shot, subtext, and dialogue.
    recap: String,
    direction: String,
}

// ── Specialist briefs (produced by the 4-way split) ──────────────────────────

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

/// Carries panel_number and recap through all three agent calls untouched.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TurnCarry {
    panel_number: usize,
    recap: String,
}

// ── Specialist agent outputs ──────────────────────────────────────────────────

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

/// All three specialist outputs converged; ready to brief the director.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AllNotes {
    choreo: ChoreoNotes,
    dialogue: DialogueNotes,
    cinema: CinemaNote,
    panel_number: usize,
    recap: String,
}

// ── Director fork types ───────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DirectorBrief {
    panel_number: usize,
    recap: String,
    choreo: ChoreoNotes,
    dialogue: DialogueNotes,
    cinema: CinemaNote,
}

/// Carries specialist notes alongside the director agent so they survive to
/// the join and enrich the per-panel recap entry on the next turn.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DirectorCarry {
    panel_number: usize,
    recap: String,
    choreo: ChoreoNotes,
    dialogue: DialogueNotes,
    cinema: CinemaNote,
}

// ── Panel types ───────────────────────────────────────────────────────────────

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

// ── Split handlers ───────────────────────────────────────────────────────────

/// Fans out one `StoryTurn` into four independent branches in a single step.
fn split_crew(
    turn: StoryTurn,
) -> (ChoreographerBrief, DialogueBrief, CinematographerBrief, TurnCarry) {
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
            direction: turn.direction,
        },
        TurnCarry { panel_number: turn.panel_number, recap: turn.recap },
    )
}

fn split4(notes: AllNotes) -> (DirectorBrief, DirectorCarry) {
    (
        DirectorBrief {
            panel_number: notes.panel_number,
            recap: notes.recap.clone(),
            choreo: notes.choreo.clone(),
            dialogue: notes.dialogue.clone(),
            cinema: notes.cinema.clone(),
        },
        DirectorCarry {
            panel_number: notes.panel_number,
            recap: notes.recap,
            choreo: notes.choreo,
            dialogue: notes.dialogue,
            cinema: notes.cinema,
        },
    )
}

// ── Merge handlers ───────────────────────────────────────────────────────────

/// Collects all three specialist outputs and the carry in a single step.
fn merge_all_notes(
    (choreo, dialogue, cinema, carry): (ChoreoNotes, DialogueNotes, CinemaNote, TurnCarry),
) -> AllNotes {
    AllNotes {
        choreo,
        dialogue,
        cinema,
        panel_number: carry.panel_number,
        recap: carry.recap,
    }
}

fn merge_director(
    (panel, carry): (ComicPanel, DirectorCarry),
) -> DirectorPanel {
    DirectorPanel {
        panel_number: carry.panel_number,
        recap: carry.recap,
        panel,
        choreo: carry.choreo,
        dialogue: carry.dialogue,
        cinema: carry.cinema,
    }
}

// ── Specialist agents ─────────────────────────────────────────────────────────

impl Agent for ChoreographerBrief {
    type Output = ChoreoNotes;

    fn build() -> AgentConfig {
        AgentConfig::new(
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
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

impl Agent for DialogueBrief {
    type Output = DialogueNotes;

    fn build() -> AgentConfig {
        AgentConfig::new(
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
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

impl Agent for CinematographerBrief {
    type Output = CinemaNote;

    fn build() -> AgentConfig {
        AgentConfig::new(
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
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

impl Agent for DirectorBrief {
    type Output = ComicPanel;

    fn build() -> AgentConfig {
        AgentConfig::new(
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
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

// ── Work node ─────────────────────────────────────────────────────────────────

async fn print_and_read(dp: DirectorPanel, _ctx: Context) -> Result<UserInput, FlowError> {
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
    println!("  [Cinematographer] {} · {}", dp.cinema.shot_type, dp.cinema.lighting);
    println!("{rule}");

    let direction = tokio::task::spawn_blocking(|| {
        print!("\nDirector's note for next panel (or 'exit'): ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        line.trim().to_string()
    })
    .await
    .map_err(|e| FlowError::Internal {
        handler: "print_and_read",
        detail: e.to_string(),
    })?;

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

// ── Either handler ────────────────────────────────────────────────────────────

fn route_input(
    input: UserInput,
) -> Either<StoryTurn, FinalSummary> {
    if input.direction.eq_ignore_ascii_case("exit") {
        return Either::Right(FinalSummary { panels_written: input.panel_number });
    }

    let dialogue_text = if input.panel.dialogues.is_empty() {
        "  (silent panel)".to_string()
    } else {
        input.panel.dialogues
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

// ── Flow ──────────────────────────────────────────────────────────────────────

impl Flow for StoryTurn {
    type Output = FinalSummary;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .split(split_crew)
            .agent::<ChoreographerBrief>()
            .agent::<DialogueBrief>()
            .agent::<CinematographerBrief>()
            .merge(merge_all_notes)
            .split(split4)
            .agent::<DirectorBrief>()
            .merge(merge_director)
            .work(print_and_read)
            .either(route_input)
            .build()
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let start = StoryTurn { panel_number: 1, recap: String::new(), direction: initial };
    let mut runtime = FlowRuntime::new(start)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(summary) => {
                println!("\n{rule}");
                println!("  Story complete — {} panel(s) written.", summary.panels_written);
                println!("{rule}");
                break;
            }
            FlowStep::Suspend(_) => {
                eprintln!("Unexpected suspension");
                break;
            }
        }
    }

    Ok(())
}
