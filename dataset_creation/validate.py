#!/usr/bin/env python3
"""Validate the output of each dataset-construction stage.

    python validate.py <stage> --work_dir work --split train

Stages (see run_pipeline.sh):

  aihub            0  extracted AI-Hub files
  processed        1  processed_data_{split}.json + context_mapping_{split}.json
  seeds            2  seed_questions_{split}.json
  dialogues        3  faithful dialogue simulation
  dialogue_eval    4  LLM dialogue-quality scores
  dialogue_filter  5  low-quality dialogue removal
  injection        6  hallucination injection
  hallu_eval       7  LLM injection-quality scores
  hallu_filter     8  low-quality injection removal
  benchmark        9  final kfinhallu_{split}.json (truncated, with retrieval ids)
  sft             10  benchmark preprocessing + SFT jsonl

Every check prints PASS / FAIL (WARN for known, tolerated issues); the exit
code is non-zero if any check fails.
Checks marked [paper] compare against the released paper data and run only
when that data is available (restore it first with scripts/restore_data.py).

Deterministic stages (processed, dialogue_filter, hallu_filter) are checked by
recomputing them independently. LLM stages cannot reproduce the paper's text,
so they are checked for schema, value ranges and the invariants of the paper
data.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

ACTIONS = {'seed_question', 'select_question', 'new_question', 'reconfirm'}
HALLU_TYPES = {'contradictory', 'inconsistency', 'unverifiable', 'irrelevance',
               'false_refusal', 'false_acceptance'}
NEGATIVE_TYPES = {'false_refusal', 'false_acceptance'}
# Paper numbers (Section 4 / Table 2); used for [paper] count checks.
PAPER = {
    'test': {'docs': 272, 'queries': 2064, 'contexts': 2108, 'dialogues': 272, 'filtered': 202},
    'train': {'docs': 2732, 'queries': 42364, 'contexts': 33368, 'dialogues': 2000, 'filtered': 1540,
              'hallu_filtered': 1320, 'benchmark': 1312},
}
# Dialogue-quality rule (filter_dialogue_quality.py): lowest score on any metric removes the
# dialogue, except correctness, which needs at least two lowest-score turns.
MIN_SCORE = {'answerability': 0, 'plausibility': 1, 'correctness': 0, 'diversity': 1, 'coherency': 1}
SCORE_RANGE = {'answerability': {0, 1}, 'plausibility': {1, 2, 3}, 'correctness': {0, 1},
               'diversity': {1, 2, 3}, 'coherency': {1, 2, 3}}


class Checker:
    def __init__(self, title):
        self.title = title
        self.failed = 0
        print(f'== {title}')

    def check(self, ok, msg, detail=None):
        print(f'  [{"PASS" if ok else "FAIL"}] {msg}' + (f'  ({detail})' if detail and not ok else ''))
        self.failed += not ok
        return ok

    def info(self, msg):
        print(f'  [info] {msg}')

    def warn(self, msg):
        print(f'  [WARN] {msg}')

    def done(self):
        print(f'== {self.title}: {"OK" if not self.failed else f"{self.failed} check(s) FAILED"}\n')
        return self.failed


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def convs_of(data):
    return data['conversations'] if isinstance(data, dict) else data


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def first_bad(items, pred):
    for x in items:
        if not pred(x):
            return x
    return None


class Paths:
    def __init__(self, work, split):
        self.work, self.split = work, split
        d = lambda *p: os.path.join(work, *p)
        self.aihub = d('data', 'aihub_data')
        self.processed = d('data', f'processed_data_{split}.json')
        self.context_mapping = d(f'context_mapping_{split}.json')
        self.seeds = d(f'seed_questions_{split}.json')
        self.dialogues = d('dialogues', f'dialogue_results_{split}.json')
        self.created = d('dialogues', f'created_data_{split}.json')
        self.dialogue_eval = d('dialogues', f'dialogue_results_evaluated_{split}.json')
        self.dialogue_filtered = d('filtered', f'dialogue_results_filtered_{split}.json')
        self.injected = d('injection', f'hallucination_injected_{split}.json')
        self.hallu_eval = d('injection', f'hallucination_evaluated_{split}.json')
        self.hallu_filtered = d('injection', f'hallucination_filtered_{split}.json')
        self.benchmark = d(f'kfinhallu_{split}.json')
        self.preprocessed = d('preprocessed', split)
        self.sft = d('sft')


def paper_file(name):
    path = os.path.join(REPO_ROOT, 'data', 'full', name)
    return path if os.path.exists(path) else None


def manifest():
    return load(os.path.join(REPO_ROOT, 'data', 'manifest.json'))


# ---------------------------------------------------------------------------
# Stage 0-2
# ---------------------------------------------------------------------------

def v_aihub(p, args):
    c = Checker('stage 0: AI-Hub files')
    sys.path.insert(0, HERE)
    from prepare_aihub import EXPECTED
    for rel, sha in EXPECTED.items():
        path = os.path.join(p.aihub, rel)
        c.check(os.path.exists(path) and sha256_file(path) == sha, f'{rel} matches the paper copy')
    extra = [os.path.relpath(os.path.join(r, f), p.aihub) for r, _, fs in os.walk(p.aihub)
             for f in fs if f.endswith('.json')]
    c.check(sorted(extra) == sorted(EXPECTED), 'no other json files (they would change preprocessing)')
    return c.done()


def v_processed(p, args):
    c = Checker(f'stage 1: preprocessing ({p.split})')
    data = load(p.processed)
    qd, cm = data['qid_data'], data['context_mapping']
    c.info(f'{len(qd):,} queries, {len(cm):,} contexts')
    # Paper's document count: documents with at least one (kept) query.
    docs = {v['doc_id'] for v in cm.values() if v.get('q_id')}
    exp = PAPER[p.split]
    c.check(len(qd) == exp['queries'], f'{exp["queries"]:,} queries (paper)', len(qd))
    c.check(len(cm) == exp['contexts'], f'{exp["contexts"]:,} contexts', len(cm))
    c.check(len(docs) == exp['docs'], f'{exp["docs"]:,} documents (paper)', len(docs))
    bad = first_bad(qd.items(), lambda kv: kv[1]['context_id'] in cm)
    c.check(bad is None, 'every query points to an existing context', bad and bad[0])
    bad = first_bad(qd.items(), lambda kv: int(kv[0]) % 2 == 1)
    c.check(bad is None, 'only odd question ids are kept (AI-Hub convention used by the paper)', bad and bad[0])
    bad = first_bad(qd.items(), lambda kv: len(kv[1]['similar_context_id']) == 10
                    and all(x in cm for x in kv[1]['similar_context_id']))
    c.check(bad is None, 'every query has 10 similar contexts from the same split', bad and bad[0])
    bad = first_bad(cm.items(), lambda kv: all(q in qd for q in kv[1].get('q_id', [])))
    c.check(bad is None, 'context -> query links are consistent', bad and bad[0])
    texts = collections.Counter(v['context'] for v in cm.values())
    c.check(max(texts.values()) == 1, 'no duplicated context text (dedup by content)')
    c.check(os.path.exists(p.context_mapping), f'{os.path.basename(p.context_mapping)} written')
    if os.path.exists(p.context_mapping):
        cmap = load(p.context_mapping)
        c.check(list(cmap) == list(cm) and all('q_id' not in v for v in cmap.values()),
                'context_mapping keeps processed_data order without q_id')
        expected = manifest()['files'][f'context_mapping_{p.split}.json']['sha256']
        c.check(sha256_file(p.context_mapping) == expected,
                f'[paper] context_mapping_{p.split}.json is byte-identical to the paper file')
    paper_proc = args.paper_processed
    if paper_proc:
        # Optional: compare with a processed_data file from the original run.
        ref = load(paper_proc)
        same_ctx = sum(qd[q]['context_id'] == ref['qid_data'].get(q, {}).get('context_id') for q in qd)
        c.check(same_ctx == len(qd), '[paper] query -> context assignment identical', f'{same_ctx}/{len(qd)}')
        sim = sum(qd[q]['similar_context_id'] == ref['qid_data'][q]['similar_context_id'] for q in qd)
        c.info(f'similar_context_id identical for {sim}/{len(qd)} queries '
               '(GPU embedding ties may reorder a few; only used as LLM candidates)')
    return c.done()


def v_seeds(p, args):
    c = Checker(f'stage 2: seed questions ({p.split})')
    seeds = load(p.seeds)
    qd = load(p.processed)['qid_data']
    ids = [s.get('question_id') for s in seeds]
    c.check(all(s.get('question_id') and s.get('question') and s.get('doc_id') for s in seeds),
            'every seed has question_id, question, doc_id')
    c.check(len(set(ids)) == len(ids), 'seed ids are unique')
    bad = first_bad(seeds, lambda s: s['question_id'] in qd and qd[s['question_id']]['doc_id'] == s['doc_id']
                    and qd[s['question_id']]['question'] == s['question'])
    c.check(bad is None, 'seeds match processed_data (id, text, document)', bad and bad['question_id'])
    docs = [s['doc_id'] for s in seeds]
    c.check(len(set(docs)) == len(docs), 'at most one seed per document')
    paper = load(os.path.join(HERE, 'configs', f'seed_qids_{p.split}.json'))['seed_qids']
    if ids == paper:
        c.info(f'paper seed list ({len(paper)} seeds, configs/seed_qids_{p.split}.json)')
    else:
        c.info(f'{len(ids)} seeds, {len(set(ids) & set(paper))} shared with the paper seed list '
               '(expected when seeds are re-extracted with the LLM)')
    return c.done()


# ---------------------------------------------------------------------------
# Stage 3-5
# ---------------------------------------------------------------------------

def check_dialogue_structure(c, convs, qid_ok, min_turns, max_turns):
    ids = [x['dialogue_id'] for x in convs]
    c.check(len(set(ids)) == len(ids), 'dialogue ids are unique')
    bad = first_bad(convs, lambda x: x['num_turns'] == len(x['dialogue'])
                    and min_turns <= len(x['dialogue']) <= max_turns)
    c.check(bad is None, f'{min_turns}-{max_turns} turns and num_turns == len(dialogue)', bad and bad['dialogue_id'])
    bad = first_bad(convs, lambda x: [t['turn'] for t in x['dialogue']] == list(range(1, len(x['dialogue']) + 1)))
    c.check(bad is None, 'turns are numbered 1..n', bad and bad['dialogue_id'])
    bad = first_bad(convs, lambda x: x['dialogue'][0]['action'] == 'seed_question'
                    and x['dialogue'][0]['qid'] == x['seed_qid'])
    c.check(bad is None, 'turn 1 asks the seed question', bad and bad['dialogue_id'])
    turns = [(x['dialogue_id'], t) for x in convs for t in x['dialogue']]
    need = ('qid', 'original_question', 'original_answer', 'user_question', 'assistant_answer',
            'passages', 'action', 'clue_text')
    bad = first_bad(turns, lambda dt: all(k in dt[1] for k in need))
    c.check(bad is None, 'every turn has ' + ', '.join(need), bad and bad[0])
    bad = first_bad(turns, lambda dt: dt[1]['action'] in ACTIONS)
    c.check(bad is None, f'actions in {sorted(ACTIONS)}', bad and (bad[0], bad[1]['action']))
    bad = first_bad(turns, lambda dt: qid_ok(dt[1]['qid']))
    c.check(bad is None, 'every qid is an AI-Hub query or a created query', bad and (bad[0], bad[1]['qid']))
    bad = first_bad(turns, lambda dt: dt[1]['passages'] and all(ps.get('context_id') for ps in dt[1]['passages']))
    c.check(bad is None, 'every turn has a retrieved passage', bad and bad[0])
    bad = first_bad(turns, lambda dt: dt[1]['user_question'].strip() and dt[1]['assistant_answer'].strip())
    c.check(bad is None, 'no empty question / answer', bad and bad[0])
    c.info('actions: ' + str(dict(collections.Counter(t['action'] for _, t in turns))))
    c.info('turns per dialogue: ' + str(dict(sorted(collections.Counter(len(x['dialogue']) for x in convs).items()))))


def v_dialogues(p, args):
    c = Checker(f'stage 3: dialogue simulation ({p.split})')
    convs = convs_of(load(p.dialogues))
    qd = load(p.processed)['qid_data']
    created = load(p.created)['qid_data'] if os.path.exists(p.created) else {}
    seeds = [s['question_id'] for s in load(p.seeds)]
    c.info(f'{len(convs)} dialogues from {len(seeds)} seeds, {len(created)} created queries')
    c.check([x['seed_qid'] for x in convs] == [s for s in seeds if s in {x['seed_qid'] for x in convs}],
            'dialogues follow the seed order')
    c.check(len(convs) == len(seeds), 'one dialogue per seed', f'{len(convs)} vs {len(seeds)}')
    max_qid = max(int(q) for q in qd)
    c.check(all(int(q) > max_qid for q in created), 'created query ids do not collide with AI-Hub ids')
    check_dialogue_structure(c, convs, lambda q: q in qd or q in created, args.min_turns, args.max_turns)
    return c.done()


def v_dialogue_eval(p, args):
    c = Checker(f'stage 4: dialogue quality scores ({p.split})')
    ev = convs_of(load(p.dialogue_eval))
    orig = convs_of(load(p.dialogues))
    c.check([x['dialogue_id'] for x in ev] == [x['dialogue_id'] for x in orig], 'every dialogue was scored')
    turns = [t for x in ev for t in x['dialogue']]
    for m in ('answerability', 'plausibility', 'correctness'):
        bad = first_bad(turns, lambda t: t.get('evaluations', {}).get(m) in SCORE_RANGE[m])
        c.check(bad is None, f'turn score {m} in {sorted(SCORE_RANGE[m])}', bad and bad.get('evaluations'))
    for m in ('diversity', 'coherency'):
        bad = first_bad(ev, lambda x: x.get('average_evaluations', {}).get(m) in SCORE_RANGE[m])
        c.check(bad is None, f'dialogue score {m} in {sorted(SCORE_RANGE[m])}', bad and bad['dialogue_id'])
    same_text = all(a['dialogue'][i]['assistant_answer'] == t['assistant_answer']
                    for a, b in zip(orig, ev) for i, t in enumerate(b['dialogue']))
    c.check(same_text, 'scoring did not modify the dialogues')
    return c.done()


def expected_good_dialogues(ev):
    good = []
    for x in ev:
        avg = x.get('average_evaluations', {}) or {}
        bad = any(avg.get(m) == MIN_SCORE[m] for m in ('diversity', 'coherency'))
        for m in ('answerability', 'plausibility'):
            bad |= any((t.get('evaluations') or {}).get(m) == MIN_SCORE[m] for t in x['dialogue'])
        bad |= sum((t.get('evaluations') or {}).get('correctness') == 0 for t in x['dialogue']) >= 2
        if not bad:
            good.append(x['dialogue_id'])
    return good


def v_dialogue_filter(p, args):
    c = Checker(f'stage 5: dialogue filtering ({p.split})')
    ev = convs_of(load(p.dialogue_eval))
    kept = convs_of(load(p.dialogue_filtered))
    expected = expected_good_dialogues(ev)
    c.check([x['dialogue_id'] for x in kept] == expected,
            'kept dialogues == independent recomputation of the quality rule',
            f'{len(kept)} kept vs {len(expected)} expected')
    rate = len(kept) / len(ev)
    paper = {'train': 0.770, 'test': 0.743}[p.split]
    c.info(f'retention {len(kept)}/{len(ev)} = {rate:.1%} (paper: {paper:.1%})')
    if len(ev) == PAPER[p.split]['dialogues'] and p.split == 'train':
        c.check(len(kept) == PAPER['train']['filtered'], '[paper] 1,540 of 2,000 train dialogues kept', len(kept))
    return c.done()


# ---------------------------------------------------------------------------
# Stage 6-8
# ---------------------------------------------------------------------------

def hallu_turns(conv):
    return [t for t in conv['dialogue'] if t.get('hallucination_type')]


def v_injection(p, args):
    c = Checker(f'stage 6: hallucination injection ({p.split})')
    inj = convs_of(load(p.injected))
    src = {x['dialogue_id']: x for x in convs_of(load(p.dialogue_filtered))}
    c.check(all(x['dialogue_id'] in src for x in inj), 'injected dialogues come from the filtered set')
    c.info(f'{len(inj)}/{len(src)} dialogues kept (dialogues refusing on a reconfirm turn are dropped)')
    types = collections.Counter(t['hallucination_type'] for x in inj for t in hallu_turns(x))
    c.check(set(types) <= HALLU_TYPES, f'hallucination types in {sorted(HALLU_TYPES)}', dict(types))
    c.info('types: ' + str(dict(types)))
    # Turns whose original answer is already a refusal are tagged false_refusal automatically;
    # every other dialogue gets at most one assigned hallucination.
    assigned = collections.Counter(sum(t['hallucination_type'] != 'false_refusal' for t in hallu_turns(x)) for x in inj)
    c.check(set(assigned) <= {0, 1}, 'at most one assigned (non-refusal) hallucination per dialogue', dict(assigned))
    per = collections.Counter(len(hallu_turns(x)) for x in inj)
    c.info(f'hallucinated turns per dialogue: {dict(sorted(per.items()))} '
           '(stage 9 keeps the first one; dialogues without one are dropped)')
    edited = [(x['dialogue_id'], t) for x in inj for t in hallu_turns(x)
              if t['hallucination_type'] not in ('false_refusal', 'false_acceptance')]
    bad = first_bad(edited, lambda dt: dt[1].get('hallucinated_answer'))
    c.check(bad is None, 'every content-type injection has a hallucinated_answer', bad and bad[0])
    unchanged = [d for d, t in edited if t.get('hallucinated_answer') == t['assistant_answer']]
    if unchanged:
        c.warn(f'{len(unchanged)} content-type injections left the answer unchanged (dialogues {unchanged[:5]}); '
               'stage 7/8 should remove them')
    fa = [(x['dialogue_id'], t) for x in inj for t in hallu_turns(x) if t['hallucination_type'] == 'false_acceptance']
    bad = first_bad(fa, lambda dt: dt[1].get('hallucinated_answer') == dt[1]['assistant_answer'])
    c.check(bad is None, 'false_acceptance keeps the answer (only the evidence is replaced)', bad and bad[0])
    bad = first_bad([(x['dialogue_id'], t) for x in inj for t in hallu_turns(x)],
                    lambda dt: dt[1].get('change_description'))
    c.check(bad is None, 'every hallucinated turn records a change_description', bad and bad[0])

    def untouched(x):
        s = src[x['dialogue_id']]
        return all(a['assistant_answer'] == b['assistant_answer'] and a['user_question'] == b['user_question']
                   for a, b in zip(s['dialogue'], x['dialogue']))
    bad = first_bad(inj, untouched)
    c.check(bad is None, 'faithful questions/answers are left unchanged', bad and bad['dialogue_id'])
    return c.done()


def v_hallu_eval(p, args):
    c = Checker(f'stage 7: injection quality scores ({p.split})')
    ev = convs_of(load(p.hallu_eval))
    inj = {x['dialogue_id'] for x in convs_of(load(p.injected))}
    c.check(all(x['dialogue_id'] in inj for x in ev), 'scored dialogues come from stage 6')
    scored = [t for x in ev for t in x['dialogue'] if 'hallucination_evaluations' in t]
    bad = first_bad(scored, lambda t: t['hallucination_evaluations'].get('appropriateness') in (1, 2, 3)
                    and t['hallucination_evaluations'].get('naturalness') in (1, 2, 3))
    c.check(bad is None, 'appropriateness / naturalness in {1,2,3}', bad and bad.get('hallucination_evaluations'))
    c.info(f'{len(scored)} scored turns in {len(ev)} dialogues')
    return c.done()


def v_hallu_filter(p, args):
    c = Checker(f'stage 8: injection filtering ({p.split})')
    ev = convs_of(load(p.hallu_eval))
    inj = convs_of(load(p.injected))
    kept = convs_of(load(p.hallu_filtered))
    failed = {x['dialogue_id'] for x in ev for t in x['dialogue'] if 'hallucination_evaluations' in t
              and (t['hallucination_evaluations'].get('appropriateness', 0) < args.min_appropriateness
                   or t['hallucination_evaluations'].get('naturalness', 0) < args.min_naturalness)}
    expected = [x['dialogue_id'] for x in inj if x['dialogue_id'] not in failed]
    c.check([x['dialogue_id'] for x in kept] == expected,
            f'kept == recomputed rule (appropriateness >= {args.min_appropriateness}, '
            f'naturalness >= {args.min_naturalness})', f'{len(kept)} vs {len(expected)}')
    c.check(all(a == b for a, b in zip(kept, [x for x in inj if x['dialogue_id'] not in failed])),
            'kept dialogues are copied unchanged from stage 6')
    if p.split == 'train' and len(inj) == PAPER['train']['filtered']:
        c.check(len(kept) == PAPER['train']['hallu_filtered'], '[paper] 1,320 dialogues kept', len(kept))
    return c.done()


# ---------------------------------------------------------------------------
# Stage 9-10
# ---------------------------------------------------------------------------

def check_benchmark_records(c, data, cmap, split):
    keys = {'dialogue_id', 'seed_qid', 'seed_question', 'doc_id', 'num_turns', 'dialogue'}
    if split == 'test':
        keys.add('injection_model')
    bad = first_bad(data, lambda r: set(r) == keys)
    c.check(bad is None, f'record keys == {sorted(keys)}', bad and sorted(bad))
    bad = first_bad(data, lambda r: r['num_turns'] == len(r['dialogue']) >= 1)
    c.check(bad is None, 'num_turns == len(dialogue)', bad and bad['dialogue_id'])
    bad = first_bad(data, lambda r: len(hallu_turns(r)) == 1 and r['dialogue'][-1].get('hallucination_type'))
    c.check(bad is None, 'truncated at the hallucinated turn: exactly one hallucination, on the last turn',
            bad and bad['dialogue_id'])
    bad = first_bad(data, lambda r: all('positive_context_ids' not in t and 'negative_context_ids' not in t
                                        for t in r['dialogue'][:-1]))
    c.check(bad is None, 'retrieval ids only on the evaluated (last) turn', bad and bad['dialogue_id'])
    last = [(r['dialogue_id'], r['dialogue'][-1]) for r in data]
    bad = first_bad(last, lambda dt: dt[1]['hallucination_type'] in HALLU_TYPES)
    c.check(bad is None, 'hallucination types are valid', bad and bad[1]['hallucination_type'])
    bad = first_bad(last, lambda dt: len(dt[1]['positive_context_ids']) == 10
                    and len(set(dt[1]['positive_context_ids'])) == 10)
    c.check(bad is None, '10 distinct positive_context_ids', bad and bad[0])
    bad = first_bad(last, lambda dt: dt[1]['passages'][0]['context_id'] in dt[1]['positive_context_ids'])
    c.check(bad is None, 'the gold passage is among the positive contexts', bad and bad[0])
    bad = first_bad(last, lambda dt: len(dt[1]['negative_context_ids']) == (10 if dt[1]['hallucination_type'] in NEGATIVE_TYPES else 0))
    c.check(bad is None, '10 negative_context_ids for false_refusal / false_acceptance, none otherwise',
            bad and (bad[0], bad[1]['hallucination_type'], len(bad[1]['negative_context_ids'])))
    placeholder = re.compile(r'^\d+_similar_placeholder$')
    ids = [i for _, t in last for i in t['positive_context_ids'] + t['negative_context_ids']]
    missing = [i for i in ids if i not in cmap and not placeholder.match(i)]
    c.check(not missing, 'every retrieval id is in context_mapping (or a false_acceptance placeholder)',
            missing[:3])
    gold_neg = first_bad(last, lambda dt: dt[1]['passages'][0]['context_id'] not in dt[1]['negative_context_ids'])
    c.check(gold_neg is None, 'the gold passage is never a negative context', gold_neg and gold_neg[0])
    types = collections.Counter(t['hallucination_type'] for _, t in last)
    c.info('types: ' + str(dict(sorted(types.items()))))


def v_benchmark(p, args):
    c = Checker(f'stage 9: benchmark file ({p.split})')
    data = load(p.benchmark)
    cmap = load(p.context_mapping)
    filtered = {x['dialogue_id']: x for x in convs_of(load(p.hallu_filtered))}
    c.info(f'{len(data)} records')
    c.check(all(r['dialogue_id'] in filtered for r in data), 'records come from stage 8')
    c.check(len(data) == sum(1 for x in filtered.values() if hallu_turns(x)),
            'every stage-8 dialogue with a hallucination is kept')

    def prefix_ok(r):
        src = filtered[r['dialogue_id']]['dialogue']
        return all(a.get('user_question') == b.get('user_question') and a.get('assistant_answer') == b.get('assistant_answer')
                   for a, b in zip(r['dialogue'], src))
    bad = first_bad(data, prefix_ok)
    c.check(bad is None, 'dialogues are prefixes of the stage-8 dialogues', bad and bad['dialogue_id'])
    check_benchmark_records(c, data, cmap, p.split)
    if p.split == 'train' and len(filtered) == PAPER['train']['hallu_filtered']:
        c.check(len(data) == PAPER['train']['benchmark'], '[paper] 1,312 train records', len(data))
    return c.done()


def v_paper_data(p, args):
    """Run the stage-9 invariants on the released (restored) paper data."""
    failed = 0
    for split in ('test', 'train'):
        data_path, cmap_path = paper_file(f'kfinhallu_{split}.json'), paper_file(f'context_mapping_{split}.json')
        c = Checker(f'paper data: kfinhallu_{split}.json')
        if not data_path:
            c.check(False, 'data/full not found; run scripts/restore_data.py first')
            failed += c.done()
            continue
        check_benchmark_records(c, load(data_path), load(cmap_path), split)
        failed += c.done()
    return failed


def v_sft(p, args):
    c = Checker(f'stage 10: preprocessing + SFT data ({p.split})')
    pre = {m: load(os.path.join(p.preprocessed, f'preprocessed_{m}_mode.json')) for m in ('hallucination', 'faithful')}
    n = len(load(p.benchmark))
    c.check(len(pre['hallucination']) == len(pre['faithful']) == n,
            f'{n} hallucination-mode + {n} faithful-mode samples ({2 * n} total)')
    bad = first_bad(pre['hallucination'] + pre['faithful'], lambda s: s['contexts'])
    c.check(bad is None, 'every sample has contexts', bad and bad['dialogue_id'])
    bad = first_bad(pre['faithful'], lambda s: s['current_answer'])
    c.check(bad is None, 'every faithful-mode sample has an answer', bad and bad['dialogue_id'])
    empty = [s for s in pre['hallucination'] if not s['current_answer']]
    bad = first_bad(empty, lambda s: s['hallucination_type'] == 'false_refusal')
    c.check(bad is None, 'only false_refusal samples may lack a hallucinated answer', bad and bad['dialogue_id'])
    if empty:
        c.warn(f'{len(empty)} hallucination-mode false_refusal samples have no answer (the original answer '
               'was already a refusal, so no hallucinated_answer was written); the prompt shows "None". '
               'This is also the case in the paper\'s SFT data (61 of 2,624).')
    if p.split == 'train' and os.path.isdir(p.sft):
        rows = {}
        for name in ('train_fourclass', 'val_fourclass', 'train_fourclass_v2', 'val_fourclass_v2'):
            path = os.path.join(p.sft, f'{name}.jsonl')
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    rows[name] = sum(1 for _ in f)
        for v in ('', '_v2'):
            if f'train_fourclass{v}' in rows:
                total = rows[f'train_fourclass{v}'] + rows.get(f'val_fourclass{v}', 0)
                c.check(total == 2 * n, f'fourclass{v}: train + val == {2 * n}', rows)
        # [paper] byte-level comparison is meaningful only for the paper's train file.
        down = manifest().get('downstream', {})
        if n == PAPER['train']['benchmark']:
            for name in rows:
                path = os.path.join(p.sft, f'{name}.jsonl')
                exp = down.get(f'sft/{name}.jsonl')
                if exp and sha256_file(path) == exp:
                    c.info(f'{name}.jsonl is byte-identical to the paper SFT data')
    return c.done()


STAGES = {
    'aihub': v_aihub, 'processed': v_processed, 'seeds': v_seeds, 'dialogues': v_dialogues,
    'dialogue_eval': v_dialogue_eval, 'dialogue_filter': v_dialogue_filter, 'injection': v_injection,
    'hallu_eval': v_hallu_eval, 'hallu_filter': v_hallu_filter, 'benchmark': v_benchmark,
    'sft': v_sft, 'paper_data': v_paper_data,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('stage', choices=list(STAGES) + ['all'])
    ap.add_argument('--work_dir', default=os.path.join(HERE, 'work'))
    ap.add_argument('--split', default='train', choices=['train', 'test'])
    ap.add_argument('--min_turns', type=int, default=4)
    ap.add_argument('--max_turns', type=int, default=8)
    ap.add_argument('--min_appropriateness', type=int, default=2)
    ap.add_argument('--min_naturalness', type=int, default=2)
    ap.add_argument('--paper_processed', default=None,
                    help='processed_data_{split}.json of the original run, if available')
    args = ap.parse_args()

    p = Paths(args.work_dir, args.split)
    stages = [s for s in STAGES if s != 'paper_data'] if args.stage == 'all' else [args.stage]
    failed = 0
    for s in stages:
        failed += STAGES[s](p, args)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
