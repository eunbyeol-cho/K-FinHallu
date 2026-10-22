#!/usr/bin/env python3
"""Restore the full K-FinHallu benchmark from the released files and AI-Hub.

The released files under data/ contain no AI-Hub source text. Every field that
was derived from the source corpus is stored as a list of segments:

    {"__restore__": [segment, ...]}

where each segment is either a literal string (text written by an LLM or an
annotator, not copied from AI-Hub) or a reference

    [source_ref, start, end]

meaning source_text[start:end]. A source_ref names one AI-Hub string together
with a hash of its content, e.g. "c:2415385:3f1a9c0e5b7d" (paragraph context),
"q:6467623:..." (question) or "a:6467623:..." (answer text). The hash makes the
reference unambiguous even when AI-Hub reuses an id for different text.

After decoding, the output files are compared byte for byte (SHA-256) against
data/manifest.json, which was computed from the exact files used for the
paper's experiments.

Usage:
    python scripts/restore_data.py --aihub_dir /path/to/aihub_download

`--aihub_dir` is the folder downloaded from AI-Hub dataset 71610
("금융, 법률 문서 기계독해 데이터"). It is searched recursively for the
Training/Validation `.zip` archives or the `.json` files extracted from them.
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HASH_LEN = 12

RELEASED_FILES = {
    # released file -> restored file
    'kfinhallu_test.json': 'kfinhallu_test.json',
    'kfinhallu_train.json': 'kfinhallu_train.json',
    'context_index_test.json': 'context_mapping_test.json',
    'context_index_train.json': 'context_mapping_train.json',
}


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:HASH_LEN]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# AI-Hub loading
# ---------------------------------------------------------------------------

def iter_aihub_json(aihub_dir):
    """Yield (name, parsed_json) for every AI-Hub json, reading inside zips.

    The same json can exist both inside a zip and extracted next to it; each
    basename is loaded once.
    """
    seen = set()
    candidates = []
    for root, _, files in os.walk(aihub_dir):
        for name in files:
            candidates.append(os.path.join(root, name))
    # Prefer extracted json over zip members, then a fixed order.
    candidates.sort(key=lambda p: (not p.endswith('.json'), p))

    for path in candidates:
        if path.endswith('.json'):
            base = os.path.basename(path)
            if base in seen:
                continue
            with open(path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            if not isinstance(data, dict) or 'data' not in data:
                continue
            seen.add(base)
            yield base, data
        elif path.endswith('.zip'):
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    base = os.path.basename(member)
                    if not base.endswith('.json') or base in seen:
                        continue
                    data = json.loads(zf.read(member).decode('utf-8-sig'))
                    if not isinstance(data, dict) or 'data' not in data:
                        continue
                    seen.add(base)
                    yield base, data


def build_source_index(aihub_dir):
    """Map source_ref -> text for every paragraph, question and answer."""
    sources = {}
    n_files = 0
    for _, data in iter_aihub_json(aihub_dir):
        n_files += 1
        for doc in data['data']:
            for para in doc.get('paragraphs') or []:
                cid = para.get('context_id')
                context = para.get('context')
                if cid and context:
                    sources[f'c:{cid}:{text_hash(context)}'] = context
                for qa in para.get('qas') or []:
                    qid = qa.get('question_id')
                    if not qid:
                        continue
                    question = qa.get('question')
                    if question:
                        sources[f'q:{qid}:{text_hash(question)}'] = question
                    answer = (qa.get('answer') or {}).get('text')
                    if answer:
                        sources[f'a:{qid}:{text_hash(answer)}'] = answer
    return sources, n_files


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class MissingSource(Exception):
    pass


def decode_segments(segments, sources):
    parts = []
    for seg in segments:
        if isinstance(seg, str):
            parts.append(seg)
            continue
        ref, start, end = seg
        if ref not in sources:
            raise MissingSource(ref)
        parts.append(sources[ref][start:end])
    return ''.join(parts)


def restore(obj, sources, missing):
    if isinstance(obj, dict):
        if set(obj) == {'__restore__'}:
            try:
                return decode_segments(obj['__restore__'], sources)
            except MissingSource as e:
                missing.add(str(e))
                return None
        return {k: restore(v, sources, missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [restore(v, sources, missing) for v in obj]
    return obj


def dump_json(obj, path):
    # Same serialization as the original files (indent=2, no trailing newline).
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--aihub_dir', required=True,
                        help='AI-Hub dataset 71610 download folder (zips or extracted json)')
    parser.add_argument('--data_dir', default=os.path.join(REPO_ROOT, 'data'),
                        help='Folder with the released files (default: data/)')
    parser.add_argument('--out_dir', default=None,
                        help='Output folder (default: <data_dir>/full)')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, 'full')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)

    print(f'Indexing AI-Hub files under {args.aihub_dir} ...')
    sources, n_files = build_source_index(args.aihub_dir)
    print(f'  {n_files} json files, {len(sources):,} source strings')
    if n_files == 0:
        sys.exit('No AI-Hub json found. Check --aihub_dir.')

    ok = True
    for released, restored in RELEASED_FILES.items():
        with open(os.path.join(args.data_dir, released), encoding='utf-8') as f:
            data = json.load(f)
        missing = set()
        full = restore(data, sources, missing)
        out_path = os.path.join(out_dir, restored)
        dump_json(full, out_path)

        expected = manifest['files'][restored]['sha256']
        actual = file_sha256(out_path)
        if missing:
            ok = False
            print(f'[FAIL] {restored}: {len(missing)} source strings not found in AI-Hub '
                  f'(e.g. {sorted(missing)[:3]})')
        elif actual != expected:
            ok = False
            print(f'[FAIL] {restored}: sha256 {actual[:16]} != expected {expected[:16]}')
        else:
            print(f'[ OK ] {restored}: identical to the paper version (sha256 {actual[:16]})')

    if not ok:
        sys.exit('\nRestoration did not reproduce the paper files. See messages above.')
    print(f'\nAll files restored to {out_dir}')


if __name__ == '__main__':
    main()
