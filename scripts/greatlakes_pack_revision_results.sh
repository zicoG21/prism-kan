#!/usr/bin/env bash
set -euo pipefail

# Pack Great Lakes revision outputs for transfer back to the local workstation.
# Run from the repository root on Great Lakes after jobs finish:
#
#   bash scripts/greatlakes_pack_revision_results.sh
#
# The script packs lightweight result/artifact files and Slurm stdout/stderr.
# It does not pack model checkpoints.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="artifacts/greatlakes"
mkdir -p "$out_dir"
manifest="$out_dir/revision_results_manifest_${stamp}.txt"
filelist="$out_dir/revision_results_filelist_${stamp}.txt"
tarball="$out_dir/revision_results_${stamp}.tar.gz"

collect_files() {
  for root in "$@"; do
    [[ -e "$root" ]] || continue
    find "$root" -type f \
      \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' -o -name '*.out' -o -name '*.err' \)
  done
}

{
  collect_files results/revision logs/greatlakes
  collect_files claim_records score_reports task_cards adapters scorers
  for f in README_WORKSHOP.md requirements.txt; do
    [[ -f "$f" ]] && printf '%s\n' "$f"
  done
} | sort -u > "$filelist"

{
  echo "# Great Lakes Revision Results Manifest"
  echo "created_at=$(date -Is)"
  echo "host=$(hostname)"
  echo
  echo "## Packed file count"
  wc -l < "$filelist"
  echo
  echo "## Packed roots"
  cut -d/ -f1 "$filelist" | sort | uniq -c
  echo
  echo "## Packed files"
  cat "$filelist"
} > "$manifest"

tar -czf "$tarball" "$manifest" "$filelist" --files-from "$filelist"

echo "Wrote manifest: $manifest"
echo "Wrote filelist: $filelist"
echo "Wrote tarball:  $tarball"
