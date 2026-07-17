# Project Story

The project started from the idea that low-attention tokens are not always low value. Exact identifiers may receive little attention before a query is known, but later become the required answer. The core design evolved from token eviction to heterogeneous memory fidelity: Full KV for exact-critical spans, Quantized KV for background.
