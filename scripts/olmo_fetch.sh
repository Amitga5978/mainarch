#!/usr/bin/env bash
# Fetch allenai/OLMo-2-0425-1B.
#
# OLMo 2 is here rather than a more familiar model because it is actually open
# source, not merely open weights. AI2 publishes the weights, the Dolma training
# corpus, the training code, and every intermediate checkpoint, so nothing
# between the training data and a token coming out of this stack is a black box.
#
# ~6 GB. The weights are fp32 in the checkpoint and are converted to f16 once at
# load time.
set -euo pipefail

DIR="${1:-./olmo-2-1b}"
REPO="${OLMO_REPO:-allenai/OLMo-2-0425-1B}"
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DIR"
cd "$DIR"

FILES=(
  config.json
  generation_config.json
  model.safetensors.index.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
)

echo "fetching ${REPO} into $(pwd)"
for f in "${FILES[@]}"; do
  if [ -s "$f" ]; then
    echo "  have $f ($(stat -c%s "$f") bytes)"
    continue
  fi
  echo "  get  $f"
  curl -fL --progress-bar -o "$f" "${BASE}/${f}"
done

echo
echo "done. next:"
echo "  just olmo-preflight $DIR      # CPU-only, no GPU needed"
echo "  just olmo 'The capital of France is' $DIR"
echo "  just olmo-serve $DIR"
