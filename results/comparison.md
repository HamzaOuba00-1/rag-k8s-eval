## Generation and RAGAS results

Means with 95% bootstrap CIs over questions. Δ = paired bootstrap difference; * = CI excludes 0.

### qwen3.8-27b on the evaluated configurations

| Chunking | Retrieval | Faithfulness | Answer relevancy | Context precision | Context recall | Refusal (out-of-corpus) | False refusal | Cited sentences |
|---|---|---|---|---|---|---|---|---|
| fixed | hybrid_rerank | 0.98 [0.95, 1.00] | 0.88 [0.83, 0.91] | 0.89 [0.82, 0.95] | 0.96 [0.89, 1.00] | 1.00 [1.00, 1.00] | 0.02 [0.00, 0.07] | 0.91 [0.84, 0.97] |
| structure | hybrid_rerank | 0.99 [0.98, 1.00] | 0.87 [0.83, 0.91] | 0.92 [0.87, 0.96] | 0.97 [0.91, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.92 [0.85, 0.98] |
| semantic | hybrid_rerank | 0.98 [0.96, 1.00] | 0.90 [0.86, 0.93] | 0.72 [0.63, 0.81] | 0.90 [0.81, 0.98] | 1.00 [1.00, 1.00] | 0.04 [0.00, 0.11] | 0.92 [0.84, 0.98] |
| fixed | bm25 | 0.99 [0.98, 1.00] | 0.89 [0.85, 0.92] | 0.80 [0.70, 0.88] | 0.94 [0.88, 1.00] | 1.00 [1.00, 1.00] | 0.04 [0.00, 0.11] | 0.92 [0.84, 0.98] |
| fixed | dense | 0.99 [0.96, 1.00] | 0.88 [0.84, 0.92] | 0.65 [0.56, 0.74] | 0.93 [0.84, 1.00] | 1.00 [1.00, 1.00] | 0.07 [0.00, 0.16] | 0.92 [0.84, 0.98] |
| fixed | hybrid | 0.98 [0.95, 1.00] | 0.87 [0.83, 0.91] | 0.84 [0.76, 0.91] | 0.96 [0.89, 1.00] | 1.00 [1.00, 1.00] | 0.04 [0.00, 0.11] | 0.89 [0.81, 0.96] |

### Paired differences vs `fixed_hybrid_rerank` (qwen3.8-27b)

| Configuration | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| structure_hybrid_rerank | +0.011 [+0.000, +0.034] | -0.003 [-0.022, +0.016] | +0.026 [-0.048, +0.109] | +0.011 [-0.067, +0.089] |
| semantic_hybrid_rerank | +0.002 [-0.030, +0.039] | +0.015 [-0.028, +0.060] | -0.166 [-0.243, -0.098] * | -0.059 [-0.141, +0.007] |
| fixed_bm25 | +0.014 [-0.012, +0.047] | +0.009 [-0.021, +0.036] | -0.094 [-0.172, -0.025] * | -0.011 [-0.078, +0.056] |
| fixed_dense | +0.000 [+0.000, +0.000] | +0.006 [-0.013, +0.027] | -0.240 [-0.334, -0.146] * | -0.022 [-0.111, +0.044] |
| fixed_hybrid | -0.011 [-0.025, +0.000] | -0.003 [-0.023, +0.018] | -0.047 [-0.109, +0.008] | +0.000 [-0.067, +0.067] |

### qwen3.8-27b vs qwen2.5-7b-instruct on `fixed_hybrid_rerank` (same contexts, same judge)

| Metric | qwen3.8-27b | qwen2.5-7b-instruct | Δ (app - reference) |
|---|---|---|---|
| faithfulness | 0.98 [0.95, 1.00] | 0.94 [0.89, 0.99] | +0.039 [-0.008, +0.093] |
| answer_relevancy | 0.88 [0.83, 0.91] | 0.86 [0.80, 0.91] | -0.012 [-0.063, +0.036] |
| context_precision | 0.89 [0.82, 0.95] | 0.89 [0.82, 0.95] | +0.000 [+0.000, +0.000] |
| context_recall | 0.96 [0.89, 1.00] | 0.96 [0.89, 1.00] | +0.000 [+0.000, +0.000] |
| citation_coverage | 0.91 [0.84, 0.97] | 0.68 [0.56, 0.78] | +0.229 [+0.129, +0.338] * |

Context precision and recall depend only on retrieval: both generators share them.
