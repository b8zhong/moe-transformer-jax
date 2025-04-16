# JAX MoE

JAX MoE is a modular implementation of a Mixture-of-Experts transformer using JAX and Haiku. It is designed for flexibility, extensibility, and performance across multiple devices. The architecture focuses on sparse expert routing, efficient attention, and scalable tokenization.

## Features

- Sparse expert activation with top-k routing per token
- Rotary embeddings and grouped-query attention
- Attention with KV caching for autoregressive use cases
- Expert capacity limits with residual fallback on overflow
- Parallelized BPE tokenizer with deterministic merges
- Modular Haiku components for experimentation
- Multi-device expert execution via `shard_map`

## Components

**MoEBlock**  
Implements token-to-expert routing. Each token selects k experts with highest router scores. Experts are capped in capacity; excess tokens are handled by residual pass-through.

**TransformerBlock**  
Layer block combining rotary attention with either dense FF or sparse MoE. Supports per-head configuration for queries and keys/values.

**MoeTransformer**  
Stacked transformer with shared embeddings. Exposes encoder and decoder functions through tied parameters.

**Tokenizer**  
Multiprocess BPE vocabulary builder. Compatible with GPT-style byte-level encoding. Outputs human-readable vocab and merge rules.

## Usage

### Tokenizer

```bash
python tokenizer/parallel_bpe.py -w 8 -n 300 -d data_cache/
```

Saves `merges.txt` and `vocabs.txt` under `data_cache/`.

### Install

```bash
pip install -r requirements.txt
```

## Notes

- Top-k expert selection is differentiable and batch-aware
- Attention uses rotary position encoding on keys and queries
- Grouped-query attention reduces KV head duplication
- Sharding logic can assign different experts to different devices
- BPE uses regex-based segmentation compatible with GPT-4 patterns

## Work in Progress

This repo is not complete for end-to-end training. Missing features include:

- Dataset pipeline and loss function
- Optimizer and training loop
- Generation/sampling logic
- Multi-host training support
- Flash attention or fused kernels

## License

MIT
