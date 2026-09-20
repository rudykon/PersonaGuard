#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="MER-PS/MER-PS-trainval"
DEST_DIR="${1:-"$ROOT_DIR/data/download"}"
TARGET_ARCHIVE="$DEST_DIR/MER_PS_trainval.zip"

mkdir -p "$DEST_DIR"

if [[ -e "$TARGET_ARCHIVE" ]]; then
  echo "Archive already exists: $TARGET_ARCHIVE"
  exit 0
fi

if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli download)
else
  cat >&2 <<'EOF'
The Hugging Face CLI is required.
Install it with:
  pip install -U huggingface_hub
Then authenticate with:
  hf auth login
EOF
  exit 127
fi

STAGING_DIR="$(mktemp -d "$DEST_DIR/.merps_download.XXXXXX")"
cleanup() {
  rm -rf -- "$STAGING_DIR"
}
trap cleanup EXIT

echo "Downloading the approved MER-PS training/validation archive..."
"${HF_CLI[@]}" "$REPO_ID"   --repo-type dataset   --include '*.zip'   --local-dir "$STAGING_DIR"

mapfile -d '' ARCHIVES < <(find "$STAGING_DIR" -type f -name '*.zip' -print0)
if [[ "${#ARCHIVES[@]}" -ne 1 ]]; then
  echo "Expected one dataset archive, found ${#ARCHIVES[@]}." >&2
  exit 1
fi

mv -- "${ARCHIVES[0]}" "$TARGET_ARCHIVE"

if command -v unzip >/dev/null 2>&1; then
  echo "Checking archive integrity..."
  unzip -tq "$TARGET_ARCHIVE"
fi

echo "Saved: $TARGET_ARCHIVE"
