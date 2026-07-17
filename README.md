# Risk-Routed Heterogeneous KV Memory

**Status:** early research prototype / paper draft  
**Core idea:** KV-cache compression should be treated as **fidelity routing**, not only token keep/drop.

Long-context Transformer inference is limited by KV-cache memory and bandwidth. Many cache compression methods decide which tokens to keep and which to evict, or apply one uniform compression level to every token. This project tests a different policy:

> Keep exact-critical spans in **Full KV**, and store low-risk background context in **low-bit Quantized KV**.

Exact-critical spans include identifiers, dates, code strings, quoted values, negations, secret markers, and key-value facts. Background text is approximated with 3-bit or 4-bit KV.

## Method in one sentence

```text
exact-critical tokens ± local window -> Full KV
all remaining background tokens       -> low-bit Quantized KV
```

This is called **risk-routed heterogeneous KV memory** or **risk-aware fidelity routing**.

## Why this matters

Classic KV-cache reduction often asks: **Which tokens should we keep?**

This project asks: **Which memory fidelity should each token use?**

This matters because a low-attention token can later become exact-answer critical. If an identifier or date is evicted, exact recall fails. Uniform quantization keeps all tokens, but treats exact-critical spans and filler background with the same fidelity. Our approach preserves brittle exact spans while compressing the rest.

## Current strongest results

### HM-16: clean forced-choice evaluation up to 2K

Qwen2.5-0.5B-Instruct, contexts 512/1024/2048.

| Policy | Effective memory | Accuracy | Gold NLL | Margin |
|---|---:|---:|---:|---:|
| Full KV | 1.000 | 0.978 | 4.184 | 15.474 |
| 4-bit ours_core | 0.264 | 0.967 | 3.345 | 15.288 |
| 4-bit quant_only | 0.250 | 0.922 | 3.993 | 14.493 |
| 3-bit ours_core | 0.203 | 0.889 | 2.792 | 15.095 |
| 3-bit quant_only | 0.188 | 0.811 | 4.980 | 13.247 |

### HM-17: long-context clean evaluation up to 16K

Qwen2.5-0.5B-Instruct, contexts 4096/8192/16384.

| Policy | Effective memory | Accuracy | Gold NLL | Margin |
|---|---:|---:|---:|---:|
| Full KV | 1.000 | 0.933 | 3.775 | 16.278 |
| 3-bit ours_core | 0.189 | 0.822 | 6.086 | 15.065 |
| 3-bit quant_only | 0.188 | 0.644 | 9.528 | 12.144 |
| 4-bit ours_core | 0.252 | 0.800 | 7.307 | 13.226 |
| 4-bit quant_only | 0.250 | 0.867 | 7.455 | 12.962 |

At 16K specifically:

| Policy | Accuracy |
|---|---:|
| Full KV | 1.000 |
| 3-bit ours_core | 0.933 |
| 3-bit quant_only | 0.733 |

### HM-18: focused 16K / 3-bit system proxy

| Policy | Accuracy | Gold NLL | Margin | Effective KV MB |
|---|---:|---:|---:|---:|
| Full KV | 1.000 | 4.424 | 16.138 | 192.000 |
| 3-bit ours_core | 0.800 | 5.740 | 17.048 | 36.157 |
| 3-bit quant_only | 0.800 | 8.483 | 14.753 | 36.000 |

### HM-19/HM-20: Qwen2.5-1.5B at 8K

| Model | Setting | Policy | Accuracy | Gold NLL | Margin | Effective KV MB |
|---|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 8K / 3-bit | Full KV | 1.000 | 1.292 | 19.102 | 224.000 |
| Qwen2.5-1.5B | 8K / 3-bit | quant_only | 0.360 | 71.199 | -5.083 | 42.000 |
| Qwen2.5-1.5B | 8K / 3-bit | ours_core | 0.240 | 72.010 | -5.041 | 42.365 |
| Qwen2.5-1.5B | 8K / 4-bit | Full KV | 1.000 | 1.341 | 18.706 | 224.000 |
| Qwen2.5-1.5B | 8K / 4-bit | quant_only | 0.800 | 7.979 | 10.964 | 56.000 |
| Qwen2.5-1.5B | 8K / 4-bit | ours_core | 0.760 | 7.973 | 10.677 | 56.337 |

**Takeaway:** larger-model validation shows 3-bit is too aggressive for the current proxy on Qwen2.5-1.5B at 8K. Uniform 4-bit quantization is a strong baseline; the current heuristic router matches gold NLL but does not improve accuracy.

## Important caveat

This repository contains an **inference-time proxy implementation**. It simulates low-bit KV fidelity by quantizing/dequantizing FP16 tensors. It does **not** implement packed 3-bit or 4-bit KV storage kernels. Therefore measured GPU peak memory includes FP16 cache tensors and proxy overhead. We report `effective_kv_mb` as an intended packed-cache estimate.

## Quick start

```bash
pip install -r requirements.txt
python experiments/forced_choice_proxy.py --model Qwen/Qwen2.5-0.5B-Instruct --context 2048 --bits 4 --cases 2 --out outputs
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

## Citation

If you use this code, cite the repository or the associated paper draft once available.
