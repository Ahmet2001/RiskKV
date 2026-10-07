# RiskKV: Risk-Aware Heterogeneous KV Cache Compression

**Status:** early research prototype / paper draft  
**Core idea:** KV-cache compression should be treated as **risk-aware fidelity routing**, not only token keep/drop.

Long-context Transformer inference is limited by KV-cache memory and bandwidth. Many KV-cache compression methods decide which tokens to keep and which to evict, or apply one uniform compression level to every token.

**RiskKV** explores a different policy:

> Preserve exact-critical spans in **Full KV**, while storing lower-risk background context in **low-bit Quantized KV**.

Exact-critical spans include identifiers, dates, code strings, quoted values, negations, secret markers, and key-value facts. Background context is approximated with 3-bit or 4-bit KV.

## Method in one sentence

```text
exact-critical tokens ± local window -> Full KV
all remaining background tokens       -> low-bit Quantized KV
```

RiskKV is a **risk-aware heterogeneous KV-cache compression** method: different parts of the context are stored at different fidelity levels according to their estimated risk of exact-recall failure.

## Motivation

Classic KV-cache reduction often asks:

> **Which tokens should we keep?**

RiskKV asks a different question:

> **Which KV fidelity should each token use?**

This distinction matters because a token that appears unimportant under a local attention signal can later become exact-answer critical. If an identifier, date, code string, or negation is evicted or degraded too aggressively, exact recall can fail.

Uniform quantization keeps all tokens, but applies the same precision everywhere. RiskKV instead reserves full-fidelity KV for brittle, high-risk spans and compresses the lower-risk background more aggressively.

## Why heterogeneous KV?

RiskKV separates context into two memory classes:

- **High-risk / exact-critical tokens:** stored in Full KV.
- **Lower-risk background tokens:** stored in low-bit Quantized KV.

The goal is to preserve exact information where precision matters most while reducing the effective KV-cache footprint for the rest of the context.

## Current strongest results

### HM-16: clean forced-choice evaluation up to 2K

Qwen2.5-0.5B-Instruct, contexts 512/1024/2048.

| Policy | Effective memory | Accuracy | Gold NLL | Margin |
|---|---:|---:|---:|---:|
| Full KV | 1.000 | 0.978 | 4.184 | 15.474 |
| 4-bit RiskKV | 0.264 | 0.967 | 3.345 | 15.288 |
| 4-bit quant_only | 0.250 | 0.922 | 3.993 | 14.493 |
| 3-bit RiskKV | 0.203 | 0.889 | 2.792 | 15.095 |
| 3-bit quant_only | 0.188 | 0.811 | 4.980 | 13.247 |

### HM-17: long-context clean evaluation up to 16K

Qwen2.5-0.5B-Instruct, contexts 4096/8192/16384.

| Policy | Effective memory | Accuracy | Gold NLL | Margin |
|---|---:|---:|---:|---:|
| Full KV | 1.000 | 0.933 | 3.775 | 16.278 |
| 3-bit RiskKV | 0.189 | 0.822 | 6.086 | 15.065 |
| 3-bit quant_only | 0.188 | 0.644 | 9.528 | 12.144 |
| 4-bit RiskKV | 0.252 | 0.800 | 7.307 | 13.226 |
| 4-bit quant_only | 0.250 | 0.867 | 7.455 | 12.962 |

At 16K specifically:

| Policy | Accuracy |
|---|---:|
| Full KV | 1.000 |
| 3-bit RiskKV | 0.933 |
| 3-bit quant_only | 0.733 |

### HM-18: focused 16K / 3-bit system proxy

| Policy | Accuracy | Gold NLL | Margin | Effective KV MB |
|---|---:|---:|---:|---:|
| Full KV | 1.000 | 4.424 | 16.138 | 192.000 |
| 3-bit RiskKV | 0.800 | 5.740 | 17.048 | 36.157 |
| 3-bit quant_only | 0.800 | 8.483 | 14.753 | 36.000 |

### HM-19/HM-20: Qwen2.5-1.5B at 8K

| Model | Setting | Policy | Accuracy | Gold NLL | Margin | Effective KV MB |
|---|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 8K / 3-bit | Full KV | 1.000 | 1.292 | 19.102 | 224.000 |
| Qwen2.5-1.5B | 8K / 3-bit | quant_only | 0.360 | 71.199 | -5.083 | 42.000 |
| Qwen2.5-1.5B | 8K / 3-bit | RiskKV | 0.240 | 72.010 | -5.041 | 42.365 |
| Qwen2.5-1.5B | 8K / 4-bit | Full KV | 1.000 | 1.341 | 18.706 | 224.000 |
| Qwen2.5-1.5B | 8K / 4-bit | quant_only | 0.800 | 7.979 | 10.964 | 56.000 |
| Qwen2.5-1.5B | 8K / 4-bit | RiskKV | 0.760 | 7.973 | 10.677 | 56.337 |

**Takeaway:** the current results suggest that heterogeneous precision can protect exact-recall-critical information under aggressive compression, especially in the smaller-model experiments. Larger-model validation also shows an important limitation: 3-bit quantization is too aggressive for the current proxy on Qwen2.5-1.5B at 8K, while uniform 4-bit quantization remains a strong baseline.

RiskKV is therefore presented as an early research direction rather than a universally superior compression policy.

## Important caveat

This repository contains an **inference-time proxy implementation**.

It simulates low-bit KV fidelity by quantizing and dequantizing FP16 tensors. It does **not** currently implement packed 3-bit or 4-bit KV-storage kernels.

Therefore, measured GPU peak memory still includes FP16 cache tensors and proxy overhead. The reported `effective_kv_mb` metric should be interpreted as an estimate of the intended packed-cache footprint rather than directly measured packed-kernel memory usage.

## Quick start

```bash
pip install -r requirements.txt

python experiments/forced_choice_proxy.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --context 2048 \
  --bits 4 \
  --cases 2 \
  --out outputs
```

## Repository layout

```text
experiments/forced_choice_proxy.py     Main proxy evaluation script
results/aggregate_results.csv          Summary of HM-16/HM-17/HM-18
results/hm19_hm20_bigger_model.csv     Qwen2.5-1.5B validation results
paper/paper.md                         Paper draft summary
docs/experiments_summary.md            Experiment log and interpretation
hf_dataset/                            Dataset card and examples
```

## Research direction

RiskKV is built around a simple hypothesis:

> KV-cache compression should be selective not only about **which tokens remain**, but also about **how faithfully different tokens are represented**.

Future work can extend the current proxy toward:

- learned or model-derived risk scoring,
- more than two fidelity tiers,
- token- and layer-aware precision routing,
- real packed low-bit KV kernels,
- broader long-context benchmarks,
- stronger comparisons against modern KV eviction and quantization baselines.

## Citation

If you use this code, please cite the repository or the associated paper draft once available.
