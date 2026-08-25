# Pravah memory evaluations

This isolated package normalizes official LoCoMo and cleaned LongMemEval v1 data, runs the histories through an application-configured `MemoryManager`, scores evidence retrieval, and compares scoped pgvector HNSW results with a forced exact scan. It does not choose an LLM, prompt, embedding model, or answer judge for you.

## Fetch and normalize

Dataset files are downloaded from pinned upstream revisions, verified by SHA-256, and kept under the ignored `data/` directory.

```bash
cd evaluations/memory
scripts/fetch-datasets.sh data

cargo run -- normalize \
  --dataset locomo \
  --granularity session \
  --input data/locomo10.json \
  --output results/locomo-session.json

cargo run -- normalize \
  --dataset longmemeval \
  --granularity session \
  --input data/longmemeval_s_cleaned.json \
  --output results/longmemeval-session.json
```

Use `--granularity turn` to measure the impact of application chunking separately. Dataset timestamps do not include timezones, so normalization interprets them as UTC and records that decision in the typed contract.

## Run Pravah retrieval

Construct providers and `MemoryManager` exactly as in the application, then pass the normalized dataset to the library runner:

```rust,no_run
# async fn run(
#     manager: pravah_memory::MemoryManager,
#     dataset: pravah_memory_eval::EvaluationDataset,
# ) -> Result<(), pravah_memory_eval::EvaluationError> {
use pravah_memory_eval::EvaluationRunner;

let runner = EvaluationRunner::builder(manager)
    .system_label("extractor=x; embedder=y; prompts=git-sha")
    .search_limits(10, 50)
    .group_concurrency(4)
    .build()?;

let run = runner.run(&dataset).await?;
let score = pravah_memory_eval::score_retrieval(&run.observations);
# let _ = score;
# Ok(())
# }
```

The runner creates a unique `(user_key, agent_key)` scope per benchmark history and run. Evidence inside a history is submitted in source order. Reconciliation is off by default; call `.reconcile(batch_limit, max_batches)` to measure it explicitly.

`score_retrieval` reports evidence recall, hit rate, and MRR overall and by upstream category. LongMemEval abstention items are reported but excluded from retrieval recall because they have no ground-truth evidence location. Full benchmark answer accuracy still requires a separately documented answer model and the benchmark's official judge.

## Compare exact and HNSW recall

Create a JSON array of `HnswQuery` values using the same embedding provider as the database profile:

```json
[
  {
    "query_id": "question-1",
    "user_key": "eval-user",
    "agent_key": "eval-agent",
    "embedding": [0.1, 0.2, 0.3]
  }
]
```

Then run:

```bash
cargo run -- hnsw \
  --database-url "$DATABASE_URL" \
  --queries results/hnsw-queries.json \
  --output results/hnsw-ef40.json \
  --k 10 \
  --ef-search 40 \
  --max-scan-tuples 20000 \
  --warmups 1 \
  --repetitions 3
```

The comparator uses identical scoped/current filters. It disables index scans for the exact oracle, disables sequential scans for ANN, enables strict iterative HNSW scans, and rejects the run unless PostgreSQL's JSON plan proves that the ANN path selected `pravah_memories_current_embedding_idx`. Run several `ef_search` values and preserve each JSON artifact; do not mix this database-only latency with end-to-end retrieval latency.

Run the live comparator contract on the supported PostgreSQL images through the shared database harness:

```bash
cd evaluations/memory
scripts/test-postgres-matrix.sh
```

Pass explicit versions, for example `scripts/test-postgres-matrix.sh 16 17`, to run a subset. The live test requires the script's destructive-fixture opt-in and is otherwise skipped, preventing accidental table replacement on an arbitrary database.

## Reproducibility rules

- Never edit or commit downloaded datasets.
- Keep provider, prompt, model, dimension, and code revisions in `system_label` and experiment notes.
- Compare session and turn evidence as separate runs.
- Use a fresh evaluation scope or database for every run.
- Report dataset variant, checksum, top-K, candidate bounds, reconciliation mode, HNSW settings, hardware, PostgreSQL/pgvector versions, and error bars.
- Do not claim official LoCoMo or LongMemEval answer quality from retrieval-only scores.
