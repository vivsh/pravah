use std::str::FromStr;

use mool as db;
use serde_json::Value as JsonValue;

use super::*;

const FIXTURE_USER: &str = "relation-work-user";
const FIXTURE_AGENT: &str = "relation-work-agent";

/// Verifies a reverse-ordered chain reaches the edge sentinel without closing the component.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with PostgreSQL 17"]
async fn recursive_edge_sentinel_bounds_executor_work() {
    let (sqlx_pool, schema_name) = relation_test_pool().await;
    create_relation_tables(&sqlx_pool).await;
    seed_reverse_chain(&sqlx_pool, 5_000).await;
    let mut pool = db::DbPool::from_pool(sqlx_pool.clone());
    let edge_limit = relation_query_limit(1);
    let ids = vec![Uuid::from_u128(1)];
    let relations = bound_relation_query(RELATION_NEIGHBOURHOOD_SQL, ids.clone(), edge_limit)
        .all::<MemoryRelationRow>(&mut pool)
        .await
        .unwrap();
    assert_eq!(relations.len(), edge_limit as usize);

    let explain =
        format!("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING OFF) {RELATION_NEIGHBOURHOOD_SQL}");
    let plan = bound_relation_query(&explain, ids, edge_limit)
        .scalar::<JsonValue>(&mut pool)
        .await
        .unwrap();
    let recursive_rows = recursive_union_rows(&plan).unwrap();
    assert!(
        recursive_rows <= edge_limit as u64 * 2 + 1,
        "recursive union produced {recursive_rows} rows for edge sentinel {edge_limit}"
    );
    drop(pool);
    execute_sql(&sqlx_pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Binds the production relation statement for behavior and plan inspection.
fn bound_relation_query(sql: &str, ids: Vec<Uuid>, edge_limit: i64) -> db::RawQuery {
    db::query(sql)
        .bind("memory_ids", ids)
        .bind("user_key", FIXTURE_USER.to_owned())
        .bind("agent_key", FIXTURE_AGENT.to_owned())
        .bind("known_at_bounded", false)
        .bind("known_at", chrono::Utc::now())
        .bind("include_supersedes", true)
        .bind("supersession_at", chrono::Utc::now())
        .bind("edge_limit", edge_limit)
}

/// Finds the recursive executor node and returns its emitted row count.
fn recursive_union_rows(plan: &JsonValue) -> Option<u64> {
    find_plan_node(plan.get(0)?.get("Plan")?, "Recursive Union")?
        .get("Actual Rows")?
        .as_u64()
}

/// Recursively locates one PostgreSQL plan node by its stable node type.
fn find_plan_node<'a>(plan: &'a JsonValue, node_type: &str) -> Option<&'a JsonValue> {
    if plan.get("Node Type")?.as_str()? == node_type {
        return Some(plan);
    }
    plan.get("Plans")?
        .as_array()?
        .iter()
        .find_map(|child| find_plan_node(child, node_type))
}

/// Creates an isolated schema so the executor fixture cannot affect application data.
async fn relation_test_pool() -> (sqlx::PgPool, String) {
    let database_url = std::env::var("MEMORY_POSTGRES_DATABASE_URL")
        .expect("MEMORY_POSTGRES_DATABASE_URL must select an isolated test database");
    let admin = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(&database_url)
        .await
        .unwrap();
    let schema = format!("pravah_relation_work_{}", Uuid::now_v7().simple());
    execute_sql(&admin, &format!("CREATE SCHEMA {schema}")).await;
    drop(admin);
    let options = sqlx::postgres::PgConnectOptions::from_str(&database_url)
        .unwrap()
        .options([("search_path", format!("{schema},public"))]);
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
        .unwrap();
    (pool, schema)
}

/// Creates only the columns and indexes used by recursive relation hydration.
async fn create_relation_tables(pool: &sqlx::PgPool) {
    execute_sql(
        pool,
        "CREATE TABLE pravah_memories (\
         id UUID PRIMARY KEY, user_key TEXT NOT NULL, agent_key TEXT NOT NULL, \
         stale BOOLEAN NOT NULL)",
    )
    .await;
    execute_sql(
        pool,
        "CREATE TABLE pravah_memory_relations (\
         from_memory_id UUID NOT NULL, to_memory_id UUID NOT NULL, \
         user_key TEXT NOT NULL, agent_key TEXT NOT NULL, origin_evidence_id UUID NOT NULL, \
         kind TEXT NOT NULL, effective_at TIMESTAMPTZ, reconciler_revision TEXT NOT NULL, \
         created_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (from_memory_id, to_memory_id))",
    )
    .await;
    execute_sql(
        pool,
        "CREATE INDEX relation_work_to_idx ON pravah_memory_relations \
         (to_memory_id, kind, from_memory_id)",
    )
    .await;
}

/// Seeds a long chain with the farthest relations first in physical order.
async fn seed_reverse_chain(pool: &sqlx::PgPool, edge_count: i32) {
    sqlx::query(
        "INSERT INTO pravah_memories (id, user_key, agent_key, stale) \
         SELECT ('00000000-0000-0000-0000-' || lpad(node::TEXT, 12, '0'))::UUID, \
                $2, $3, FALSE \
         FROM generate_series(1, $1 + 1) AS node",
    )
    .bind(edge_count)
    .bind(FIXTURE_USER)
    .bind(FIXTURE_AGENT)
    .execute(pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO pravah_memory_relations \
         (from_memory_id, to_memory_id, user_key, agent_key, origin_evidence_id, \
          kind, effective_at, reconciler_revision, created_at) \
         SELECT ('00000000-0000-0000-0000-' || lpad(edge::TEXT, 12, '0'))::UUID, \
                ('00000000-0000-0000-0000-' || lpad((edge + 1)::TEXT, 12, '0'))::UUID, \
                $2, $3, '00000000-0000-0000-0000-000000000001'::UUID, \
                'corroborates', NULL, 'fixture', now() \
         FROM generate_series($1, 1, -1) AS edge",
    )
    .bind(edge_count)
    .bind(FIXTURE_USER)
    .bind(FIXTURE_AGENT)
    .execute(pool)
    .await
    .unwrap();
}

/// Executes one fixture DDL statement.
async fn execute_sql(pool: &sqlx::PgPool, sql: &str) {
    sqlx::query(sql).execute(pool).await.unwrap();
}
