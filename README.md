# K-FinHallu

**A Hallucination Detection Benchmark for Multi-Turn RAG in Korean Finance** (Findings of EMNLP 2026)

[[Paper]](https://arxiv.org/abs/2605.29523)

K-FinHallu is a benchmark for detecting hallucinations in multi-turn RAG dialogues grounded in
authentic Korean financial documents. Hallucinations are injected under a taxonomy based on context
answerability: *Contradictory*, *Unverifiable*, *Irrelevance* and *False Refusal* when the retrieved
documents contain the answer, and *False Acceptance* when they do not. A detector reads the dialogue
history, the retrieved documents and the current response, and classifies the response.

| Split | Base dialogues | Records | Evaluation samples |
|-------|---------------:|--------:|-------------------:|
| Test  | 202 | 404 | 808 |
| Train | 1,312 | 1,312 | 2,624 |

We first generate faithful (hallucination-free) dialogues of 4–8 turns and then inject hallucinations
into them. For the test split, GPT-4o and Gemini-2.5-Flash each inject into every one of the 202 base
dialogues (202 x 2 = 404 hallucinated samples). Each sample is truncated at its injection turn, so the
hallucination is always on the final turn and the preceding history is faithful. Pairing each
hallucinated sample with its unmodified counterpart gives 404 faithful samples, 808 in total. The
training split follows the same scheme with GPT-4o only (1,312 base dialogues, 2,624 samples).

## Repository structure

The repository holds the benchmark data, the code to evaluate detectors on it, and the code that
built it:

```
K-FinHallu/
├── data/                  benchmark files without AI-Hub source text (see step 1)
├── scripts/
│   └── restore_data.py    fills in the source text from your AI-Hub download
├── benchmark/             test-sample preprocessing, detector inference, metrics, SFT-R training
└── dataset_creation/      construction pipeline from the AI-Hub corpus to the training data
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # API keys for GPT / Gemini
```

## 1. Restore the data

Following AI-Hub's terms, the released files contain no source text. Passages, source questions,
source answers and clue sentences are stored as references to the AI-Hub corpus (ids and character
offsets) and are filled in from your own copy of the corpus.

1. Download [AI-Hub dataset 71610](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71610)
   (금융, 법률 문서 기계독해 데이터; AI-Hub account required). All 20 zip files of `Training` and
   `Validation` (`raw_data` TS_/VS_ and `labeled_data` TL_/VL_, types 1–5) are needed. They can stay
   zipped:

   ```
   aihub_download/
   ├── Training/raw_data/TS_1.정답경계 추출형.zip ... TS_5.절차(방법)형.zip
   ├── Training/labeled_data/TL_1.정답경계 추출형.zip ... TL_5.절차(방법)형.zip
   ├── Validation/raw_data/VS_1 ... VS_5 (.zip)
   └── Validation/labeled_data/VL_1 ... VL_5 (.zip)
   ```

2. Restore (Python standard library only):

   ```bash
   python scripts/restore_data.py --aihub_dir /path/to/aihub_download
   ```

   This writes `data/full/kfinhallu_{test,train}.json` and `data/full/context_mapping_{test,train}.json`
   and checks each file against `data/manifest.json`; `[ OK ]` means it is identical to the file used
   in the paper. The benchmark and fine-tuning steps read `data/full/`.

## 2. Evaluate a detector

```bash
cd benchmark
python -m src.data.preprocess                                   # 808 test samples

# API model
python -m src.inference.run_inference --model_name gpt-5 --classification_type fourclass

# open model: start a local vLLM server (in another terminal), then run inference against it
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --port 8000 --tensor-parallel-size 4 --max-model-len 9216
python -m src.inference.run_inference --model_name Qwen/Qwen3-8B --port 8000 --classification_type fourclass

python -m src.evaluate_results --results_dir ../results/inference
```

`--classification_type`: `binary`, `fourclass`, `multiclass`.
All models of the paper: `bash scripts/run_paper_models.sh fourclass`.

## 3. Fine-tune (SFT-R)

See [`benchmark/README.md`](benchmark/README.md).

## 4. Build your own training data

The construction pipeline generates a training split from the AI-Hub corpus. You can run it with
newer LLMs than the GPT-4o / GPT-5-mini used in the paper, e.g. to build a harder split:

```bash
cd dataset_creation
bash run_pipeline.sh --aihub_dir /path/to/aihub_download --split train \
  --gen_model <dialogue / injection model> --judge_model <scoring / hard-negative model>
```

The default (`gpt-4o`, `gpt-5-mini`) follows the paper. See
[`dataset_creation/README.md`](dataset_creation/README.md).

## License

Code and annotations: [CC BY-NC 4.0](LICENSE). AI-Hub source data: AI-Hub terms of use
([`NOTICE.md`](NOTICE.md)).

> This research (paper) used datasets from 'The Open AI Dataset Project (AI-Hub, S. Korea)'. All
> data information can be accessed through 'AI-Hub (www.aihub.or.kr)'.

## Citation

```bibtex
@article{cho2026kfinhallu,
  title   = {K-FinHallu: A Hallucination Detection Benchmark for Multi-Turn RAG in Korean Finance},
  author  = {Cho, Eunbyeol and Lee, Yunseung and Kim, Mirae and Yang, Jeewon and Kwak, Youngjun and Choi, Edward},
  journal = {arXiv preprint arXiv:2605.29523},
  year    = {2026}
}
```
