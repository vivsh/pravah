#!/usr/bin/env bash
set -euo pipefail

dbharness="${DBHARNESS:-$HOME/Projects/dbharness/bin/dbharness}"
target_dir="${TMPDIR:-/tmp}/pravah-memory-eval-target"
versions=("$@")
if [[ ${#versions[@]} -eq 0 ]]; then
    versions=(16 17 18)
fi

for version in "${versions[@]}"; do
    target="postgres:${version}"
    namespace="pravah-memory-eval-pg${version}"
    if ! "$dbharness" up "$target" --namespace "$namespace"; then
        "$dbharness" down "$target" --namespace "$namespace" || true
        exit 1
    fi
    eval "$("$dbharness" env "$target" --namespace "$namespace" --prefix PRAVAH_EVAL)"
    if ! PRAVAH_EVAL_DESTRUCTIVE_FIXTURE=1 CARGO_TARGET_DIR="$target_dir" \
        cargo test --offline --test hnsw_live; then
        "$dbharness" down "$target" --namespace "$namespace"
        exit 1
    fi
    "$dbharness" down "$target" --namespace "$namespace"
done
