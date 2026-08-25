use std::path::PathBuf;

use argh::FromArgs;
use pravah_memory_eval::{
    DatasetKind, EvaluationRun, EvidenceGranularity, HnswComparator, HnswComparisonOptions,
    HnswQuery, datasets, score_retrieval,
};

/// Normalize and score Pravah memory evaluations.
#[derive(FromArgs)]
struct Args {
    #[argh(subcommand)]
    command: Command,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum Command {
    Normalize(NormalizeArgs),
    Score(ScoreArgs),
    Hnsw(HnswArgs),
}

/// Checksum-validate and normalize an official dataset JSON file.
#[derive(FromArgs)]
#[argh(subcommand, name = "normalize")]
struct NormalizeArgs {
    /// locomo or longmemeval
    #[argh(option)]
    dataset: String,
    /// session or turn
    #[argh(option, default = "String::from(\"session\")")]
    granularity: String,
    /// official source JSON file
    #[argh(option)]
    input: PathBuf,
    /// normalized output JSON file
    #[argh(option)]
    output: PathBuf,
}

/// Score an EvaluationRun JSON artifact by evidence provenance.
#[derive(FromArgs)]
#[argh(subcommand, name = "score")]
struct ScoreArgs {
    /// evaluation run JSON file
    #[argh(option)]
    input: PathBuf,
    /// retrieval score output JSON file
    #[argh(option)]
    output: PathBuf,
}

/// Compare forced exact and HNSW top-K over precomputed query vectors.
#[derive(FromArgs)]
#[argh(subcommand, name = "hnsw")]
struct HnswArgs {
    /// postgreSQL URL for an already migrated Pravah database
    #[argh(option)]
    database_url: String,
    /// JSON array of scoped HnswQuery values
    #[argh(option)]
    queries: PathBuf,
    /// comparison output JSON file
    #[argh(option)]
    output: PathBuf,
    /// top-K result count
    #[argh(option, default = "10")]
    k: u32,
    /// pgvector HNSW ef_search
    #[argh(option, default = "40")]
    ef_search: u32,
    /// strict iterative-scan tuple bound
    #[argh(option, default = "20_000")]
    max_scan_tuples: u32,
    /// unmeasured executions per path and query
    #[argh(option, default = "1")]
    warmups: u32,
    /// measured executions per path and query
    #[argh(option, default = "3")]
    repetitions: u32,
}

#[tokio::main]
async fn main() -> Result<(), pravah_memory_eval::EvaluationError> {
    match argh::from_env::<Args>().command {
        Command::Normalize(args) => normalize(args).await,
        Command::Score(args) => score(args).await,
        Command::Hnsw(args) => hnsw(args).await,
    }
}

async fn normalize(args: NormalizeArgs) -> Result<(), pravah_memory_eval::EvaluationError> {
    let kind = parse_dataset(&args.dataset)?;
    let granularity = parse_granularity(&args.granularity)?;
    let dataset = datasets::load_official(args.input, kind, granularity).await?;
    write_json(args.output, &dataset).await
}

async fn score(args: ScoreArgs) -> Result<(), pravah_memory_eval::EvaluationError> {
    let run: EvaluationRun = read_json(args.input).await?;
    write_json(args.output, &score_retrieval(&run.observations)).await
}

async fn hnsw(args: HnswArgs) -> Result<(), pravah_memory_eval::EvaluationError> {
    let queries: Vec<HnswQuery> = read_json(args.queries).await?;
    let pool = sqlx::PgPool::connect(&args.database_url).await?;
    let options = HnswComparisonOptions {
        k: args.k,
        ef_search: args.ef_search,
        max_scan_tuples: args.max_scan_tuples,
        warmups: args.warmups,
        repetitions: args.repetitions,
    };
    let comparison = HnswComparator::new(pool).compare(&queries, options).await?;
    write_json(args.output, &comparison).await
}

fn parse_dataset(value: &str) -> Result<DatasetKind, pravah_memory_eval::EvaluationError> {
    match value {
        "locomo" => Ok(DatasetKind::LoCoMo),
        "longmemeval" => Ok(DatasetKind::LongMemEval),
        _ => Err(configuration("dataset must be locomo or longmemeval")),
    }
}

fn parse_granularity(
    value: &str,
) -> Result<EvidenceGranularity, pravah_memory_eval::EvaluationError> {
    match value {
        "session" => Ok(EvidenceGranularity::Session),
        "turn" => Ok(EvidenceGranularity::Turn),
        _ => Err(configuration("granularity must be session or turn")),
    }
}

async fn read_json<T: serde::de::DeserializeOwned>(
    path: PathBuf,
) -> Result<T, pravah_memory_eval::EvaluationError> {
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|error| pravah_memory_eval::EvaluationError::io(&path, error))?;
    Ok(serde_json::from_slice(&bytes)?)
}

async fn write_json<T: serde::Serialize>(
    path: PathBuf,
    value: &T,
) -> Result<(), pravah_memory_eval::EvaluationError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| pravah_memory_eval::EvaluationError::io(parent, error))?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    tokio::fs::write(&path, bytes)
        .await
        .map_err(|error| pravah_memory_eval::EvaluationError::io(path, error))
}

fn configuration(message: impl Into<String>) -> pravah_memory_eval::EvaluationError {
    pravah_memory_eval::EvaluationError::InvalidConfiguration(message.into())
}
