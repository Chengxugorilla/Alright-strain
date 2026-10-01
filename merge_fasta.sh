#!/usr/bin/env bash

set -euo pipefail

output="Victoria_DNA.fasta"
temp_file="$(mktemp "./.${output}.tmp.XXXXXX")"
trap 'rm -f "$temp_file"' EXIT

shopt -s nullglob
fasta_files=(*.fasta)
input_count=0

for fasta_file in "${fasta_files[@]}"; do
    [[ "$fasta_file" == "$output" ]] && continue

    cat -- "$fasta_file" >> "$temp_file"

    # 防止缺少行尾换行符的文件与下一个 FASTA 文件粘连。
    if [[ -s "$fasta_file" ]] && [[ -n "$(tail -c 1 -- "$fasta_file")" ]]; then
        printf '\n' >> "$temp_file"
    fi

    ((input_count += 1))
done

if ((input_count == 0)); then
    echo "错误：当前目录中没有可合并的 .fasta 文件。" >&2
    exit 1
fi

mv -- "$temp_file" "$output"
trap - EXIT

echo "已将 ${input_count} 个 FASTA 文件合并为 ${output}。"
