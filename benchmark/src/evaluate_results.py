"""Compute the paper's metrics from inference result files.

    python -m src.evaluate_results --results_dir ../results/inference

Reads every test_results_*_{binary,fourclass,multiclass}.json (written by
src.inference.run_inference) and sft_eval_*.json (src.sft.evaluate) in
--results_dir and prints the tables of the paper:

  binary     Table 3   Acc / Prec / Rec / F1 (positive class = hallucination)
  fourclass  Table 4   Faithful / Hallucinated / Refusal (False + True) / Overall (also Table 12)
             Table 8   full four-class view
  multiclass Table 9   per-type F1 and their macro average
  binary     Table 11  recall per hallucination type and injection source

Conventions (these reproduce the paper's numbers exactly):
  * An output that could not be parsed has prediction None.
    - Accuracy-type summary numbers (binary Acc/Prec/Rec/F1, four-class
      Overall) are computed over parsed outputs only.
    - Per-class / per-type numbers (four-class class accuracies, multi-class
      F1, recall by injection source) count an unparsed output as wrong.
  * `inconsistency` (contradiction with the dialogue history) is a subtype of
    Contradictory and is reported together with it.
  * Overall of the multi-class table is the macro average of the five per-type
    F1 scores; Overall of the injection-source table is the mean of the five
    per-type recalls as printed (rounded to 2 decimals).
"""

import argparse
import glob
import json
import os
from collections import defaultdict

TYPE_GROUP = {
    'contradictory': 'contradictory',
    'inconsistency': 'contradictory',
    'unverifiable': 'unverifiable',
    'irrelevance': 'irrelevance',
    'false_acceptance': 'false_acceptance',
    'false_refusal': 'false_refusal',
}
TYPES = ['contradictory', 'unverifiable', 'irrelevance', 'false_acceptance', 'false_refusal']
# multiclass label ids (src/prompts.py): 0 none, 1 contradictory, 2 unverifiable,
# 3 irrelevance, 4 false refusal, 5 false acceptance
MULTICLASS_LABEL = {'contradictory': 1, 'unverifiable': 2, 'irrelevance': 3,
                    'false_refusal': 4, 'false_acceptance': 5}


def load_rows(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for mode in ('hallucination_mode', 'faithful_mode'):
        block = data.get(mode)
        if block is not None:
            rows += block['results'] if isinstance(block, dict) else block
    if not rows and 'results' in data:
        rows = data['results']
    # Label by file name: the metadata model name is not unique (a fine-tuned
    # model served as "Qwen/Qwen3-8B" would collide with the base model).
    name = os.path.basename(path)[:-len('.json')]
    for prefix in ('test_results_', 'sft_eval_'):
        if name.startswith(prefix):
            name = ('sft:' if prefix == 'sft_eval_' else '') + name[len(prefix):]
    for suffix in ('_binary', '_fourclass_v2', '_fourclass', '_multiclass'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name, rows


def parsed(rows):
    return [r for r in rows if r.get('prediction') is not None]


def safe_div(a, b):
    return a / b if b else 0.0


def binary_metrics(rows):
    rows = parsed(rows)
    tp = sum(r['ground_truth'] == 1 and r['prediction'] == 1 for r in rows)
    fp = sum(r['ground_truth'] == 0 and r['prediction'] == 1 for r in rows)
    fn = sum(r['ground_truth'] == 1 and r['prediction'] != 1 for r in rows)
    acc = safe_div(sum(r['prediction'] == r['ground_truth'] for r in rows), len(rows))
    prec, rec = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': safe_div(2 * prec * rec, prec + rec)}


def fourclass_metrics(rows):
    per = defaultdict(lambda: [0, 0])
    for r in rows:                      # unparsed outputs count as wrong
        per[r['ground_truth']][1] += 1
        per[r['ground_truth']][0] += r['prediction'] == r['ground_truth']
    acc = lambda *cs: safe_div(sum(per[c][0] for c in cs), sum(per[c][1] for c in cs))
    ok = parsed(rows)                   # overall: parsed outputs only
    overall = safe_div(sum(r['prediction'] == r['ground_truth'] for r in ok), len(ok))
    return {'faithful': acc(0), 'hallucinated': acc(1), 'false_refusal': acc(2),
            'true_refusal': acc(3), 'refusal': acc(2, 3), 'overall': overall}


def multiclass_metrics(rows):
    out = {}                            # unparsed outputs count as wrong
    for t in TYPES:
        lab = MULTICLASS_LABEL[t]
        tp = sum(r['ground_truth'] == lab and r['prediction'] == lab for r in rows)
        fp = sum(r['ground_truth'] != lab and r['prediction'] == lab for r in rows)
        fn = sum(r['ground_truth'] == lab and r['prediction'] != lab for r in rows)
        p, rc = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        out[t] = safe_div(2 * p * rc, p + rc)
    out['overall'] = sum(out[t] for t in TYPES) / len(TYPES)
    return out


def recall_by_source(rows):
    """Binary recall on hallucinated samples, per type and injection source."""
    cnt = defaultdict(lambda: [0, 0])
    for r in rows:                      # unparsed outputs count as missed
        if r.get('mode') != 'hallucination':
            continue
        key = (TYPE_GROUP[r['hallucination_type']], r.get('injection_model'))
        cnt[key][1] += 1
        cnt[key][0] += r['prediction'] == 1
    out = {k: safe_div(*v) for k, v in cnt.items()}
    for src in {k[1] for k in out}:
        # The paper averages the per-type values as printed (2 decimals).
        out[('overall', src)] = sum(round(out.get((t, src), 0.0), 2) for t in TYPES) / len(TYPES)
    return out


def table(title, header, body):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]
    line = lambda cells: '  '.join(str(c).ljust(w) for c, w in zip(cells, widths))
    print(f'\n{title}\n{line(header)}')
    for row in body:
        print(line(row))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--json_out', default=None, help='also write all metrics as json')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, '*.json')))
    metrics = {'binary': {}, 'fourclass': {}, 'multiclass': {}, 'source': {}}
    for path in files:
        base = os.path.basename(path)
        if not (base.startswith('test_results_') or base.startswith('sft_eval_')):
            continue
        name, rows = load_rows(path)
        if base.endswith('_binary.json'):
            metrics['binary'][name] = binary_metrics(rows)
            metrics['source'][name] = recall_by_source(rows)
        elif '_fourclass' in base:
            metrics['fourclass'][name] = fourclass_metrics(rows)
        elif base.endswith('_multiclass.json'):
            metrics['multiclass'][name] = multiclass_metrics(rows)

    f3 = lambda x: f'{x:.3f}'
    if metrics['binary']:
        table('Binary detection (Table 3)', ['Model', 'Acc.', 'Prec.', 'Rec.', 'F1'],
              [[m, *(f3(v[k]) for k in ('acc', 'prec', 'rec', 'f1'))] for m, v in metrics['binary'].items()])
    if metrics['fourclass']:
        table('Four-class, refusal collapsed (Table 4)', ['Model', 'Faith.', 'Hallu.', 'Ref.', 'Overall'],
              [[m, *(f3(v[k]) for k in ('faithful', 'hallucinated', 'refusal', 'overall'))]
               for m, v in metrics['fourclass'].items()])
        table('Four-class, full view (Table 8)', ['Model', 'Faith.', 'Hallu.', 'F.Ref.', 'T.Ref.', 'Overall'],
              [[m, *(f3(v[k]) for k in ('faithful', 'hallucinated', 'false_refusal', 'true_refusal', 'overall'))]
               for m, v in metrics['fourclass'].items()])
    if metrics['multiclass']:
        table('Multi-class F1 (Table 9)', ['Model', 'Contra.', 'Unver.', 'Irrel.', 'F.Acc.', 'F.Ref.', 'Overall'],
              [[m, *(f3(v[k]) for k in TYPES + ['overall'])] for m, v in metrics['multiclass'].items()])
    if metrics['source']:
        cols = TYPES + ['overall']
        table('Binary recall by type and injection source, gpt-4o / gemini (Table 11)',
              ['Model'] + [f'{c[:6]}-{s[:3]}' for c in cols for s in ('gpt-4o', 'gemini')],
              [[m] + [f'{v.get((c, s), 0):.2f}' for c in cols for s in ('gpt-4o', 'gemini')]
               for m, v in metrics['source'].items()])

    if args.json_out:
        serial = {k: {m: {('|'.join(map(str, kk)) if isinstance(kk, tuple) else kk): vv
                          for kk, vv in v.items()} for m, v in d.items()} for k, d in metrics.items()}
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(serial, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
