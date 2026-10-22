# Data

| file | content |
|------|---------|
| `kfinhallu_{test,train}.json` | benchmark records (404 / 1,312) |
| `context_index_{test,train}.json` | retrieval passages (2,108 / 33,368) |
| `manifest.json` | checksums of the paper's files |

AI-Hub text fields (`seed_question`, `original_question`, `original_answer`, `clue_text`,
`context`) are stored as `{"__restore__": [...]}` references. Run
`python scripts/restore_data.py --aihub_dir ...` to get the full files in `data/full/`.

Each record is a dialogue truncated at its hallucinated (last) turn. That turn has
`hallucination_type`, `hallucinated_answer`, and the retrieval contexts `positive_context_ids` /
`negative_context_ids`.
