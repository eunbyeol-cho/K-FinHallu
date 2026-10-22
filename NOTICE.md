# NOTICE: source data and terms

## Source dataset

K-FinHallu is built on **금융, 법률 문서 기계독해 데이터** (Korean Financial and Legal Document Machine
Reading Comprehension), distributed by AI-Hub, the Open AI Dataset Project operated by Korea's
National Information Society Agency (NIA).

- AI-Hub: https://aihub.or.kr
- Source dataset: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71610

## Redistribution

The release of this derived dataset was reviewed by AI-Hub / NIA with the consent of the data
builder, and approved on the conditions that it contains no source text and that the source is
attributed.

This repository **does** distribute, under CC BY-NC 4.0:
- the dialogues, injected hallucinations, labels and annotations created for K-FinHallu;
- references to the source corpus (`context_id`, `question_id`, character offsets, content hashes);
- the code for restoring, building and evaluating the benchmark.

This repository **does not** distribute any AI-Hub source text (passages, source questions, source
answers, clue sentences). Users obtain the source corpus from AI-Hub themselves and rebuild the full
benchmark with `scripts/restore_data.py`.

## Required attribution

Publications that use K-FinHallu must also acknowledge AI-Hub:

> This research (paper) used datasets from 'The Open AI Dataset Project (AI-Hub, S. Korea)'. All
> data information can be accessed through 'AI-Hub (www.aihub.or.kr)'.

> 이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "금융, 법률 문서
> 기계독해 데이터"를 활용하여 수행된 연구입니다. 본 연구에 활용된 데이터는 AI 허브(aihub.or.kr)에서
> 다운로드 받으실 수 있습니다.

## Licenses

- K-FinHallu code, dialogues and annotations: CC BY-NC 4.0 (see `LICENSE`).
- AI-Hub source data: AI-Hub terms of use (https://aihub.or.kr). Users are responsible for complying
  with both.
