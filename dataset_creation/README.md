# Dataset construction

```bash
bash run_pipeline.sh --aihub_dir /path/to/aihub_download --split train
bash run_pipeline.sh --split train --from 5 --to 8      # run a range of stages
python validate.py <stage> --split train                # check a stage's output
```

Requires a GPU (KURE-v1 embeddings) and `OPENAI_API_KEY` (or `GEMINI_API_KEY` with
`--provider google`). Outputs go to `work/`. Each stage is checked by `validate.py` before the
next one runs, so a new run keeps the structure of the paper's data.

Models: `--gen_model` writes the dialogues and injects the hallucinations (paper: `gpt-4o`);
`--judge_model` selects seeds (with `--llm_seeds`), scores quality and confirms hard negatives
(paper: `gpt-5-mini`). Use them to build a new split with newer models, e.g. a harder one.

| # | stage | model |
|---|-------|-------|
| 0 | extract AI-Hub files (`prepare_aihub.py`) | – |
| 1 | preprocessing, passage index | KURE-v1 |
| 2 | seed questions (paper seeds in `configs/`; `--llm_seeds` to re-select) | judge |
| 3 | dialogue simulation | gen |
| 4 | dialogue quality scoring | judge |
| 5 | dialogue filtering | – |
| 6 | hallucination injection | gen |
| 7 | injection quality scoring | judge |
| 8 | injection filtering | – |
| 9 | truncation, positive / negative contexts → `kfinhallu_{split}.json` | KURE-v1, judge |
| 10 | benchmark preprocessing, SFT data | – |

LLM stages (3, 4, 6, 7, 9) give different text on every run. For `--split test` only stages 0–5
run; the released test set was additionally revised by human annotators.
