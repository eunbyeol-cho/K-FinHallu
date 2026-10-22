# Benchmark

Run from `benchmark/` after restoring the data (`python ../scripts/restore_data.py ...`).

## Evaluation

```bash
python -m src.data.preprocess                      # -> ../data/preprocessed/ (404 + 404 samples)

python -m src.inference.run_inference --model_name gemini-2.5-pro --classification_type fourclass
python -m src.inference.run_inference --model_name Qwen/Qwen3-32B --port 8000 --classification_type fourclass  # vLLM server on :8000

python -m src.evaluate_results --results_dir ../results/inference
```

- `--classification_type`: `binary` (Table 3), `fourclass` (Tables 4, 8), `multiclass` (Table 9)
- All 12 detectors of the paper: `bash scripts/run_paper_models.sh fourclass`
- Prompts: `src/prompts.py`

## Fine-tuning (Qwen3-8B SFT-R)

SFT-R is trained to give a short rationale (answerability of the documents, then the response)
before the label.

```bash
# training data
python -m src.data.preprocess --dialogues_path ../data/full/kfinhallu_train.json \
  --context_mapping_path ../data/full/context_mapping_train.json --output_dir ../data/preprocessed_train
python -m src.data.prepare_sft_data_v2 --input_dir ../data/preprocessed_train   # -> ../data/sft/{train,val}_fourclass_v2.jsonl

# training (8 GPUs)
NPROC_PER_NODE=8 python -m src.sft.train --config configs/sft/qwen3_8b_fourclass_v2.yaml

# evaluation: merge the checkpoint with the lowest validation loss and serve it with vLLM
swift export --adapters <checkpoint> --merge_lora true
python -m vllm.entrypoints.openai.api_server --model <checkpoint>-merged --served-model-name sft-r --port 8000
python -m src.sft.evaluate --model_name sft-r --port 8000 --classification_type fourclass_v2
```

Ablation without the rationale (Qwen3-8B SFT): use `src.data.prepare_sft_data --classification_type fourclass`,
`configs/sft/qwen3_8b_fourclass.yaml` and `--classification_type fourclass`.
