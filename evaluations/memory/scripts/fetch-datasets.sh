#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-data}"
mkdir -p "$output_dir"

locomo_revision="3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
locomo_sha="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
locomo_url="https://raw.githubusercontent.com/snap-research/locomo/${locomo_revision}/data/locomo10.json"

long_revision="98d7416c24c778c2fee6e6f3006e7a073259d48f"
long_sha="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
long_url="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/${long_revision}/longmemeval_s_cleaned.json?download=true"

fetch_verified() {
    local url="$1"
    local expected="$2"
    local destination="$3"
    local temporary="${destination}.part"
    curl --fail --location --retry 3 --output "$temporary" "$url"
    local actual
    actual="$(shasum -a 256 "$temporary" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "checksum mismatch for $destination: expected $expected, received $actual" >&2
        return 1
    fi
    mv "$temporary" "$destination"
}

fetch_verified "$locomo_url" "$locomo_sha" "$output_dir/locomo10.json"
fetch_verified "$long_url" "$long_sha" "$output_dir/longmemeval_s_cleaned.json"
