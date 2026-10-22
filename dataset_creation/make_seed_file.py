#!/usr/bin/env python3
"""Stage 2 (default): build the seed-question file from the paper's seed ids.

    python make_seed_file.py --split train --processed work/data/processed_data_train.json \
        --output work/seed_questions_train.json

The paper's dialogues were generated from the seed ids in
configs/seed_qids_{split}.json (in dialogue_id order). This script looks the
questions up in processed_data, so the dialogue simulator starts from exactly
the same seeds. To select new seeds with an LLM instead, run
src/seed_question_extractor.py (run_pipeline.sh --llm_seeds).
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', required=True, choices=['train', 'test'])
    ap.add_argument('--processed', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    with open(os.path.join(HERE, 'configs', f'seed_qids_{args.split}.json'), encoding='utf-8') as f:
        seed_ids = json.load(f)['seed_qids']
    with open(args.processed, encoding='utf-8') as f:
        qid_data = json.load(f)['qid_data']

    missing = [q for q in seed_ids if q not in qid_data]
    if missing:
        raise SystemExit(f'{len(missing)} seed ids not in {args.processed}, e.g. {missing[:3]}')
    seeds = [{'doc_id': qid_data[q]['doc_id'], 'question_id': q, 'question': qid_data[q]['question']}
             for q in seed_ids]
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)
    print(f'{len(seeds)} seed questions -> {args.output}')


if __name__ == '__main__':
    main()
