#!/usr/bin/env bash
set -euo pipefail

# Pack Great Lakes revision outputs for transfer back to the local workstation.
# Run from the repository root on Great Lakes after jobs finish:
#
#   bash scripts/greatlakes_pack_revision_results.sh
#
# The script only packs CSV/MD logs and Slurm stdout/stderr. It does not pack
# model checkpoints.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="artifacts/greatlakes"
mkdir -p "$out_dir"
manifest="$out_dir/revision_results_manifest_${stamp}.txt"
tarball="$out_dir/revision_results_${stamp}.tar.gz"

{
  echo "# Great Lakes Revision Results Manifest"
  echo "created_at=$(date -Is)"
  echo "host=$(hostname)"
  echo
  echo "## Result files"
  find results/revision \
    -type f \
    \( -name '*.csv' -o -name '*.md' -o -name '*.json' \) \
    | sort
  echo
  echo "## Great Lakes logs"
  find logs/greatlakes \
    -type f \
    \( -name '*.out' -o -name '*.err' \) \
    | sort
} > "$manifest"

tar -czf "$tarball" \
  "$manifest" \
  $(find results/revision -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' \) | sort) \
  $(find logs/greatlakes -type f \( -name '*.out' -o -name '*.err' \) | sort)

echo "Wrote manifest: $manifest"
echo "Wrote tarball:  $tarball"
