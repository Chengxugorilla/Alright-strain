#!/usr/bin/env bash
set -euo pipefail

dir=${1:-.}
output="$dir/merged.fasta"
shopt -s nullglob
inputs=()
for file in "$dir"/*.fasta; do [[ "$file" != "$output" ]] && inputs+=("$file"); done
((${#inputs[@]})) || { echo "没有可合并的 FASTA 文件。" >&2; exit 1; }

tmp=$(mktemp "$dir/.merged.fasta.XXXXXX")
trap 'rm -f "$tmp"' EXIT
for file in "${inputs[@]}"; do cat -- "$file"; printf '\n'; done > "$tmp"
mv -- "$tmp" "$output"
echo "已合并 ${#inputs[@]} 个文件：$output"
