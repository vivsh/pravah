use pravah_memory_eval::{HnswComparator, HnswComparisonOptions, HnswQuery};
use sqlx::{PgPool, postgres::PgPoolOptions};
use uuid::Uuid;

/// Verifies the comparator proves plan separation and reports high recall on a live pgvector index.
#[tokio::test]
async fn compares_forced_exact_and_hnsw_paths() {
    if std::env::var("PRAVAH_EVAL_DESTRUCTIVE_FIXTURE").as_deref() != Ok("1") {
        return;
    }
    let Some(database_url) = std::env::var("PRAVAH_EVAL_DATABASE_URL").ok() else {
        return;
    };
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&database_url)
        .await
        .expect("disposable PostgreSQL must accept connections");
    create_fixture(&pool).await;
    let queries = fixture_queries();
    let comparison = HnswComparator::new(pool)
        .compare(
            &queries,
            HnswComparisonOptions {
                repetitions: 2,
                ..HnswComparisonOptions::default()
            },
        )
        .await
        .expect("forced exact and HNSW plans should be comparable");
    assert_eq!(comparison.cases.len(), queries.len());
    assert!(comparison.mean_recall_at_k >= 0.9);
    assert!(
        comparison
            .hnsw_plan
            .to_string()
            .contains("pravah_memories_current_embedding_idx")
    );
}

async fn create_fixture(pool: &PgPool) {
    for statement in fixture_schema() {
        sqlx::query(statement)
            .execute(pool)
            .await
            .expect("fixture schema statement should succeed");
    }
    for index in 0..500_u32 {
        insert_memory(pool, index).await;
    }
    sqlx::query(
        "CREATE INDEX pravah_memories_current_embedding_idx ON pravah_memories \
         USING HNSW ((embedding::vector(3)) vector_cosine_ops) \
         WHERE stale = FALSE AND current_for_retrieval = TRUE",
    )
    .execute(pool)
    .await
    .expect("HNSW fixture index should be created");
    sqlx::query("ANALYZE pravah_memories")
        .execute(pool)
        .await
        .expect("fixture statistics should be refreshed");
}

fn fixture_schema() -> [&'static str; 5] {
    [
        "DROP TABLE IF EXISTS pravah_memories",
        "DROP TABLE IF EXISTS pravah_memory_profile",
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE TABLE pravah_memory_profile (id SMALLINT PRIMARY KEY, embedding_dimensions INTEGER NOT NULL)",
        "CREATE TABLE pravah_memories (id UUID PRIMARY KEY, user_key TEXT NOT NULL, agent_key TEXT NOT NULL, embedding vector(3) NOT NULL, stale BOOLEAN NOT NULL, current_for_retrieval BOOLEAN NOT NULL)",
    ]
}

async fn insert_memory(pool: &PgPool, index: u32) {
    if index == 0 {
        sqlx::query("INSERT INTO pravah_memory_profile (id, embedding_dimensions) VALUES (1, 3)")
            .execute(pool)
            .await
            .expect("profile fixture should be inserted");
    }
    let vector = format_vector(&fixture_vector(index));
    sqlx::query(
        "INSERT INTO pravah_memories \
         (id, user_key, agent_key, embedding, stale, current_for_retrieval) \
         VALUES ($1, 'eval-user', 'eval-agent', CAST($2 AS vector(3)), FALSE, TRUE)",
    )
    .bind(Uuid::now_v7())
    .bind(vector)
    .execute(pool)
    .await
    .expect("memory fixture should be inserted");
}

fn fixture_queries() -> Vec<HnswQuery> {
    [7_u32, 101, 249, 413]
        .into_iter()
        .map(|index| HnswQuery {
            query_id: format!("q-{index}"),
            user_key: "eval-user".to_owned(),
            agent_key: "eval-agent".to_owned(),
            embedding: fixture_vector(index),
        })
        .collect()
}

fn fixture_vector(index: u32) -> Vec<f32> {
    let angle = index as f32 * 0.017_453_292;
    vec![angle.cos(), angle.sin(), (index % 17) as f32 / 17.0]
}

fn format_vector(values: &[f32]) -> String {
    let values = values
        .iter()
        .map(f32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}
