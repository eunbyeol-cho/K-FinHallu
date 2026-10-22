#!/usr/bin/env bash
# Run every detector of the paper on the 808 test samples.
#
#   bash scripts/run_paper_models.sh [binary|fourclass|multiclass] [api|vllm|all]
#
# Run from benchmark/ after `python -m src.data.preprocess`. Closed models need OPENAI_API_KEY /
# GOOGLE_API_KEY. Open models are served one at a time with vLLM on CUDA_VISIBLE_DEVICES
# (the paper used 4x RTX A6000 48GB per model, tensor parallel 4).
set -euo pipefail

TASK="${1:-fourclass}"
WHICH="${2:-all}"
OUT="../results/inference"
PORT=8000
TP="${TP:-4}"

API_MODELS=(gpt-4o gpt-5 gemini-2.5-flash gemini-2.5-pro)
VLLM_MODELS=(
  kakaocorp/kanana-2-30b-a3b-instruct
  LGAI-EXAONE/EXAONE-4.0-32B
  meta-llama/Llama-3.1-8B-Instruct
  meta-llama/Llama-3.1-70B-Instruct
  meta-llama/Llama-3.3-70B-Instruct
  Qwen/Qwen3-8B
  Qwen/Qwen3-14B
  Qwen/Qwen3-32B
)

if [[ "$WHICH" == api || "$WHICH" == all ]]; then
  for m in "${API_MODELS[@]}"; do
    python -m src.inference.run_inference --model_name "$m" --classification_type "$TASK" --output_dir "$OUT"
  done
fi

if [[ "$WHICH" == vllm || "$WHICH" == all ]]; then
  for m in "${VLLM_MODELS[@]}"; do
    python -m vllm.entrypoints.openai.api_server --model "$m" --port "$PORT" \
      --tensor-parallel-size "$TP" --max-model-len 9216 --gpu-memory-utilization 0.85 --enforce-eager &
    SERVER=$!
    until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do
      kill -0 "$SERVER" 2> /dev/null || { echo "vLLM failed to start for $m" >&2; exit 1; }
      sleep 10
    done
    python -m src.inference.run_inference --model_name "$m" --port "$PORT" --classification_type "$TASK" \
      --output_dir "$OUT"
    kill "$SERVER"; wait "$SERVER" 2> /dev/null || true
  done
fi

python -m src.evaluate_results --results_dir "$OUT"
