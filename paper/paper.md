# Risk-Routed Heterogeneous KV Memory for Long-Context Transformer Inference

## Abstract

Long-context Transformer inference is often limited by KV-cache storage and memory bandwidth. Existing cache compression methods usually frame the problem as token eviction or uniform compression. We argue that exact-recall workloads need a different framing: **fidelity routing**. Exact-critical spans such as identifiers, dates, code strings, quoted values, and negations should remain in Full KV, while low-risk background context can be stored in low-bit Quantized KV. In clean forced-choice experiments, 4-bit risk-routed fidelity routing reaches 96.7% accuracy at 26.4% effective memory, nearly matching Full KV at 97.8%. In long-context tests up to 16K, 3-bit routing improves accuracy over uniform 3-bit quantization from 64.4% to 82.2% at nearly identical effective memory. Larger-model validation shows that 4-bit quantization is a strong baseline and that the current heuristic router needs improvement. The current implementation is a proxy and does not yet implement packed low-bit KV storage.

## Core thesis

KV-cache compression should be formulated as a token-level memory-fidelity assignment problem, not only as keep/drop or uniform quantization.

## Method

1. Detect exact-critical spans.
2. Expand them by a local window.
3. Store those tokens in Full KV.
4. Store all remaining background tokens in low-bit Quantized KV.

## Limitations

- No packed 3-bit/4-bit KV storage implementation yet.
- Heuristic detector only.
- Main results are controlled synthetic exact-recall tasks.
- Stronger baselines such as SnapKV/H2O/RULER subsets remain future work.
