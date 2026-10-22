#!/usr/bin/env python3
"""Stage 0: extract the AI-Hub source files used to build K-FinHallu.

    python prepare_aihub.py --aihub_dir /path/to/aihub_download --out_dir work/data/aihub_data

Extracts exactly the 17 json files the paper's data was built from (four
question types for raw/labeled data in Training and Validation, plus
VL_tableqa.json) into the layout expected by src/preprocess.py:

    <out_dir>/Training/{raw_data,labeled_data}/*.json
    <out_dir>/Validation/{raw_data,labeled_data}/*.json

Each file is checked against the sha256 of the copy used for the paper, so a
changed AI-Hub release is detected before any processing starts.
"""

import argparse
import hashlib
import os
import sys
import zipfile

# relative path -> sha256 of the file used for the paper
EXPECTED = {
    'Training/labeled_data/TL_multiple_choice.json': 'ad6930a3147162f0df1a386c20eb788c79dadef99b0df0569bebccefcf5c10b3',
    'Training/labeled_data/TL_span_extraction_how.json': '22f7ec340e0f7aa8f48c5d581c513907d81f6776287d9e9a30d1f96c3506b2e2',
    'Training/labeled_data/TL_span_extraction.json': 'd0b2fd8b0214060a5a4844db27e93965f46fba6c539c32e272e38ac3637d87ac',
    'Training/labeled_data/TL_text_entailment.json': 'f96d3d41721dc217edcfd5ab580d4a33b0fbaa578e4812e8d06467837f9768d7',
    'Training/raw_data/TS_multiple_choice.json': '418a7295f4c2eae95a6abe3f4afbc9fd8db1bb2b864231e75f0645f548503601',
    'Training/raw_data/TS_span_extraction_how.json': 'd13faa9933347fc0ae9f8a97f3ca1f296ea2fe48620cd157caefd71149063271',
    'Training/raw_data/TS_span_extraction.json': '7efca03d0beaf5fcf1cbb4d674cbecb703d7449b7d2d3865d5bb145c9d201530',
    'Training/raw_data/TS_text_entailment.json': '83486187b9ca0843da4c41630d8bd4d2662b60f914da658bd9708467e0152761',
    'Validation/labeled_data/VL_multiple_choice.json': 'b8605d72ed9ede49349736c8a5a3226c31b7bf33cdf39b34b9910cb6233adc31',
    'Validation/labeled_data/VL_span_extraction_how.json': 'cc8f94e4f71701a507e96e4edd38e94170cfe4e725d69db00cc432b242dda288',
    'Validation/labeled_data/VL_span_extraction.json': 'e19ed1ee7d4ac54cb7628a83895e49e2f4022515bcb66b43d9b31c5f1129af7d',
    'Validation/labeled_data/VL_tableqa.json': '1c332e38dac085946321fc620244da4df5e1567cd7999d95bd5cd16914c6edbb',
    'Validation/labeled_data/VL_text_entailment.json': 'f681782130532db2c5cf942bbde0f0dd9f7ef49848a3ba989e4e190526c25464',
    'Validation/raw_data/VS_multiple_choice.json': 'd32e4a4ac48b6c06bd4e243135942355851d9b9d5ad61913ff5993b24cb69794',
    'Validation/raw_data/VS_span_extraction_how.json': '4b08df69ee687115d7ab7d0cf323b9fdc124c171d8dd121daa272437dd0a5aa1',
    'Validation/raw_data/VS_span_extraction.json': '4ff815b7dec622db54d222b6855985d5b9d447c90e3eff4ed05727816b1ed1c3',
    'Validation/raw_data/VS_text_entailment.json': '146b6cd930d612e75d6a6e142066d93790716ab3415e299b2b5e090cba436239',
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def find_sources(aihub_dir):
    """basename -> bytes reader, from extracted json or from zip members."""
    found = {}
    for root, _, files in os.walk(aihub_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            if name.endswith('.json'):
                found.setdefault(name, lambda p=path: open(p, 'rb').read())
            elif name.endswith('.zip'):
                with zipfile.ZipFile(path) as zf:
                    for member in zf.namelist():
                        base = os.path.basename(member)
                        if base.endswith('.json'):
                            found.setdefault(base, lambda p=path, m=member: zipfile.ZipFile(p).read(m))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--aihub_dir', required=True, help='AI-Hub dataset 71610 download folder')
    ap.add_argument('--out_dir', required=True, help='e.g. work/data/aihub_data')
    args = ap.parse_args()

    sources = find_sources(args.aihub_dir)
    failed = []
    for rel, expected in EXPECTED.items():
        out = os.path.join(args.out_dir, rel)
        if os.path.exists(out) and sha256_bytes(open(out, 'rb').read()) == expected:
            print(f'[ OK ] {rel} (already extracted)')
            continue
        base = os.path.basename(rel)
        if base not in sources:
            failed.append(f'{rel}: not found under {args.aihub_dir}')
            continue
        data = sources[base]()
        if sha256_bytes(data) != expected:
            failed.append(f'{rel}: sha256 differs from the file used for the paper')
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(data)
        print(f'[ OK ] {rel}')

    # Any extra json in out_dir would change preprocessing; refuse it.
    extra = []
    for root, _, files in os.walk(args.out_dir):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), args.out_dir)
            if name.endswith('.json') and rel not in EXPECTED:
                extra.append(rel)
    if extra:
        failed.append(f'unexpected json files in {args.out_dir} (remove them): {extra}')

    if failed:
        print('\n'.join(['', 'FAILED:'] + failed))
        sys.exit(1)
    print(f'\nAll {len(EXPECTED)} AI-Hub files verified in {args.out_dir}')


if __name__ == '__main__':
    main()
