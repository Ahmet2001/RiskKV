---
license: mit
language:
- en
task_categories:
- text-generation
- question-answering
pretty_name: Risk-Routed KV Exact-Recall Benchmark
size_categories:
- n<1K
---

# Risk-Routed KV Exact-Recall Benchmark

This dataset contains controlled synthetic exact-recall examples used to evaluate **risk-routed heterogeneous KV memory** policies for long-context Transformer inference.

The benchmark is designed for testing whether a model can retrieve exact strings from long contexts under different KV-cache policies:

- **Full KV**
- **Uniform low-bit Quantized KV**
- **Risk-routed heterogeneous KV**, where exact-critical spans stay in Full KV and background context is quantized

## Tasks

1. `single_needle`: retrieve a secret marker from a long filler context.
2. `multi_needle`: retrieve the marker associated with a queried label such as BLUE.
3. `kv_retrieval`: retrieve a key-value record such as a city access code.
4. `code_string`: retrieve the exact string returned by a small code block.
5. `date_negation`: retrieve the approved final date while ignoring a negated distractor.

## Schema

Each JSONL row has:

```json
{
  "id": "single_needle-8192w-0",
  "task": "single_needle",
  "context_words_target": 8192,
  "context": "...long context...",
  "query": "Which marker was labeled as the secret marker?",
  "gold": "MK53-4897X",
  "distractors": ["ZX17-2044Q", "ALPHA-77K2"],
  "choices": ["MK53-4897X", "ZX17-2044Q", "ALPHA-77K2"],
  "exact_spans": ["MK53-4897X"],
  "source": "synthetic_controlled_exact_recall"
}
```

## Intended evaluation

For forced-choice evaluation, compute answer sequence NLL for each candidate in `choices` after the context and query. The prediction is correct if the `gold` answer has the lowest NLL.

## Related code

https://github.com/Ahmet2001/-risk-routed-heterogeneous-kv-memory
