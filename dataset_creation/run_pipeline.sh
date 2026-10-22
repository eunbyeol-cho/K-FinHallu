#!/usr/bin/env bash
# End-to-end construction of a K-FinHallu split from the AI-Hub corpus.
#
#   bash run_pipeline.sh --aihub_dir /path/to/aihub_download [options]
#
# Options
#   --split train|test   split to build (default: train)
#   --work_dir DIR       where every intermediate file goes (default: dataset_creation/work)
#   --from N / --to N    run only stages N..M (default: 0..10)
#   --llm_seeds          select seed questions with the LLM instead of the paper's seed list
#   --gen_model M        model that writes dialogues and injects hallucinations (default: gpt-4o)
#   --judge_model M      model that selects seeds, scores quality and picks hard negatives
#                        (default: gpt-5-mini)
#   --provider P         openai | google (default: openai)
#   --python PY          python interpreter (default: python)
#
# Stages (each is followed by `validate.py <stage>`; the pipeline stops on a failed check)
#    0 aihub            extract + checksum the AI-Hub files                      (deterministic)
#    1 processed        filter documents/paragraphs, embed with KURE-v1          (deterministic*)
#    2 seeds            one seed question per document                           (paper list / LLM)
#    3 dialogues        faithful dialogue simulation, user + assistant LLM       (LLM: gen_model)
#    4 dialogue_eval    5-criterion dialogue scoring                             (LLM: judge_model)
#    5 dialogue_filter  drop dialogues with lowest scores                        (deterministic)
#    6 injection        hallucination injection                                  (LLM: gen_model)
#    7 hallu_eval       appropriateness / naturalness scoring                    (LLM: judge_model)
#    8 hallu_filter     drop injections scoring < 2                              (deterministic)
#    9 benchmark        truncate + positive / hard-negative contexts             (LLM: judge_model)
#   10 sft              benchmark preprocessing + SFT jsonl                      (deterministic)
#   * up to GPU floating-point ties in the embedding neighbours (see README).
#
# The test split was human-revised after stage 5 and after injection (paper, Sec. 4), so for
# --split test only stages 0-5 are automated; the released test set is the revised result.
#
# The paper used gpt-4o / gpt-5-mini. Requires OPENAI_API_KEY (or GEMINI_API_KEY with
# --provider google) for stages 2 (--llm_seeds), 3, 4, 6, 7 and 9, in the environment or .env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SRC="$HERE/src"

AIHUB_DIR=""; SPLIT="train"; WORK="$HERE/work"; FROM=0; TO=10; LLM_SEEDS=0; PY="python"
GEN_MODEL="gpt-4o"; JUDGE_MODEL="gpt-5-mini"; PROVIDER="openai"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aihub_dir) AIHUB_DIR="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --work_dir) WORK="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --llm_seeds) LLM_SEEDS=1; shift ;;
    --gen_model) GEN_MODEL="$2"; shift 2 ;;
    --judge_model) JUDGE_MODEL="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ "$SPLIT" == "test" && "$TO" -gt 5 ]]; then
  echo "test split: stages after 5 involve human revision; running stages $FROM..5 only"
  TO=5
fi
mkdir -p "$WORK"/{data,dialogues,filtered,injection,preprocessed}
WORK="$(cd "$WORK" && pwd)"
cd "$WORK"   # src/*.py resolve data/ and vector_db_cache/ relative to the working directory

validate() { "$PY" "$HERE/validate.py" "$1" --work_dir "$WORK" --split "$SPLIT"; }
want() { [[ "$1" -ge "$FROM" && "$1" -le "$TO" ]]; }
banner() { echo; echo "######## stage $1: $2 ($SPLIT)"; }

if want 0; then
  banner 0 "AI-Hub files"
  [[ -n "$AIHUB_DIR" ]] || { echo "--aihub_dir is required for stage 0" >&2; exit 2; }
  "$PY" "$HERE/prepare_aihub.py" --aihub_dir "$AIHUB_DIR" --out_dir "$WORK/data/aihub_data"
  validate aihub
fi

if want 1; then
  banner 1 "preprocessing"
  "$PY" "$SRC/preprocess.py" --split "$SPLIT" --output "data/processed_data_$SPLIT.json"
  "$PY" "$SRC/build_context_mapping.py" --source "data/processed_data_$SPLIT.json" \
    --output "context_mapping_$SPLIT.json"
  validate processed
fi

if want 2; then
  banner 2 "seed questions"
  if [[ "$LLM_SEEDS" == 1 ]]; then
    "$PY" "$SRC/seed_question_extractor.py" --input_data "data/processed_data_$SPLIT.json" \
      --output "seed_questions_$SPLIT.json" --provider "$PROVIDER" --model "$JUDGE_MODEL"
  else
    "$PY" "$HERE/make_seed_file.py" --split "$SPLIT" --processed "data/processed_data_$SPLIT.json" \
      --output "seed_questions_$SPLIT.json"
  fi
  validate seeds
fi

if want 3; then
  banner 3 "dialogue simulation"
  # main.py skips seeds already present in --output and then overwrites it with the new
  # dialogues only, so always start this stage from an empty output.
  rm -f "dialogues/dialogue_results_$SPLIT.json" "dialogues/created_data_$SPLIT.json"
  "$PY" "$SRC/main.py" --split "$SPLIT" \
    --processed-data "data/processed_data_$SPLIT.json" --seed-questions "seed_questions_$SPLIT.json" \
    --min-turns 4 --max-turns 8 --provider "$PROVIDER" --model "$GEN_MODEL" --num-workers 4 \
    --output "dialogues/dialogue_results_$SPLIT.json" --created-data "dialogues/created_data_$SPLIT.json"
  validate dialogues
fi

if want 4; then
  banner 4 "dialogue quality scoring"
  "$PY" "$SRC/evaluate_dialogue.py" --input "dialogues/dialogue_results_$SPLIT.json" \
    --output "dialogues/dialogue_results_evaluated_$SPLIT.json" \
    --provider "$PROVIDER" --model "$JUDGE_MODEL" --parallel-conversations --num-workers 8
  validate dialogue_eval
fi

if want 5; then
  banner 5 "dialogue filtering"
  "$PY" "$SRC/filter_dialogue_quality.py" --split "$SPLIT" --output-dir filtered \
    --eval-input "dialogues/dialogue_results_evaluated_$SPLIT.json" \
    --original-input "dialogues/dialogue_results_$SPLIT.json" \
    --created-data-input "dialogues/created_data_$SPLIT.json"
  validate dialogue_filter
fi

if want 6; then
  banner 6 "hallucination injection"
  "$PY" "$SRC/inject_hallucinations.py" --input "filtered/dialogue_results_filtered_$SPLIT.json" \
    --output "injection/hallucination_injected_$SPLIT.json" --provider "$PROVIDER" --model "$GEN_MODEL" --seed 42
  validate injection
fi

if want 7; then
  banner 7 "injection quality scoring"
  "$PY" "$SRC/evaluate_hallucination.py" --input "injection/hallucination_injected_$SPLIT.json" \
    --output "injection/hallucination_evaluated_$SPLIT.json" \
    --provider "$PROVIDER" --model "$JUDGE_MODEL" --parallel-conversations --num-workers 8
  validate hallu_eval
fi

if want 8; then
  banner 8 "injection filtering"
  "$PY" "$SRC/filter_hallucination_quality.py" --eval "injection/hallucination_evaluated_$SPLIT.json" \
    --original "injection/hallucination_injected_$SPLIT.json" \
    --output "injection/hallucination_filtered_$SPLIT.json" --min-appropriateness 2 --min-naturalness 2
  validate hallu_filter
fi

if want 9; then
  banner 9 "benchmark file"
  "$PY" "$SRC/build_train_benchmark.py" --input "injection/hallucination_filtered_$SPLIT.json" \
    --original-dialogue "dialogues/dialogue_results_$SPLIT.json" \
    --processed-data "data/processed_data_$SPLIT.json" --created-data "filtered/created_data_$SPLIT.json" \
    --output "kfinhallu_$SPLIT.json" --embedding-cache-dir "vector_db_cache/$SPLIT" \
    --provider "$PROVIDER" --model "$JUDGE_MODEL" --num-workers 8 --num-turn-workers 8 \
    --cache-path "vector_db_cache/$SPLIT/classification_cache.json" --seed 42
  validate benchmark
fi

if want 10; then
  banner 10 "benchmark preprocessing + SFT data"
  pushd "$REPO/benchmark" > /dev/null
  "$PY" -m src.data.preprocess --dialogues_path "$WORK/kfinhallu_$SPLIT.json" \
    --context_mapping_path "$WORK/context_mapping_$SPLIT.json" --output_dir "$WORK/preprocessed/$SPLIT"
  "$PY" -m src.data.prepare_sft_data --input_dir "$WORK/preprocessed/$SPLIT" --output_dir "$WORK/sft" \
    --classification_type fourclass
  "$PY" -m src.data.prepare_sft_data_v2 --input_dir "$WORK/preprocessed/$SPLIT" --output_dir "$WORK/sft"
  popd > /dev/null
  validate sft
fi

echo; echo "Done: stages $FROM..$TO of the $SPLIT split in $WORK"
