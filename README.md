# K-FinHallu

**A Hallucination Detection Benchmark for Multi-Turn RAG in Korean Finance**

## Abstract

Large Language Models (LLMs) have advanced financial automation through Retrieval-Augmented Generation (RAG), yet hallucinations remain a critical barrier to deployment in high-stakes environments. Existing benchmarks focus on single-turn, English-centric tasks, leaving the multi-turn dynamics and linguistic-regulatory nuances of the Korean financial domain unaddressed. We introduce K-FinHallu, the first benchmark for hallucination detection in multi-turn Korean financial RAG, designed as a controlled evaluation of source faithfulness and answerability. We construct multi-turn dialogues from authentic Korean financial documents and inject hallucinations under a proposed hierarchical taxonomy based on context answerability that explicitly accounts for justified abstention. Benchmarking frontier and open-source LLMs as hallucination detectors, we find that even the strongest models struggle with fine-grained financial diagnostics and refusal behavior. While fine-tuning an 8B model on our training split yields performance competitive with frontier LLMs, justified abstention remains the weakest axis across all evaluated models.

---

The benchmark and evaluation code will be released here.
