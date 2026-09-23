# TinyK3

A decoder-only transformer language model built from scratch, combining **MLA (Multi-head Latent Attention)**, **Stable LatentMoE with Quantile-Balancing routing**, and **Attention Residuals** in a single architecture. Fully compatible with HuggingFace `AutoModel` (`model_type="tiny_k3"`).

> **⚠️ Training Status: ~25% complete.** The latest checkpoint on [HuggingFace](https://huggingface.co/AbdelrhmanEbied/Tiny-K3) is from step 2000 of a 10000-step run — I ran out of GPU credits. Resume training by downloading the checkpoint (see [Resuming Training](#resuming-training)).

---

## Architecture

### High-Level Model Flow

```
Input IDs [batch, seq_len]
        │
        ▼
┌─────────────────┐
│  Embedding      │  embed_tokens: vocab_size × hidden_size
│  (tied lm_head) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    Transformer Blocks                    │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  AttentionResidual (depth attention over blocks)  │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  RMSNorm → MLA (Multi-head Latent Attention)      │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  AttentionResidual (depth attention over blocks)  │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  RMSNorm → MoE (Stable LatentMoE) or DenseFFN    │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          │                              │
│  × 20 layers             │  Block boundary every 4      │
│                          │  layers commits partial      │
│                          │  block to blocks list        │
└──────────────────────────┼──────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  AttentionResidual     │  final_res
              │  (final depth mix)     │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │  RMSNorm               │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │  lm_head               │  tied to embed_tokens
              │  [batch, seq, vocab]   │
              └───────────┬────────────┘
                          ▼
                  Causal LM Loss
```

### MLA (Multi-head Latent Attention)

Instead of storing full K/V matrices, MLA compresses them into a low-rank latent representation:

```
Input x [B, T, D]
        │
        ▼
┌──────────────────────────────────────────────────┐
│  w_in (fused projection)                         │
│  [c_kv | c_q | k_rope] = x @ w_in               │
│                                                   │
│  c_kv      : [B, T, kv_lora_rank=256]  ← latent │
│  c_q       : [B, T, kv_lora_rank=256]           │
│  k_rope_raw: [B, T, qk_rope_dim=32]              │
└──────────┬───────────────────┬────────────────────┘
           │                   │
     ┌─────▼─────┐      ┌─────▼─────┐
     │  w_uq_nope│      │  w_qr_rope│
     │  → q_nope │      │  → q_rope │
     │ [B,H,T,64]│      │ [B,H,T,32]│
     └─────┬─────┘      └─────┬─────┘
           │            ┌─────▼─────┐
           │            │   RoPE    │
           │            └─────┬─────┘
     ┌─────▼──────────────────▼─────┐
     │  q = [q_nope | q_rope]       │
     │  [B, H, T, 96]               │
     └──────────────┬───────────────┘
                    │
     ┌──────────────▼───────────────┐
     │  k = c_kv @ w_uk^T           │  ← keys from latent
     │  v = c_kv @ w_uv             │  ← values from latent
     │  [B, H, T, 96] / [B,H,T,96]  │
     └──────────────┬───────────────┘
                    │
     ┌──────────────▼───────────────┐
     │  SDPA (flash / sdpa backend) │
     │  is_causal=True              │
     │  scale = softmax_scale       │
     └──────────────┬───────────────┘
                    │
     ┌──────────────▼───────────────┐
     │  w_o → output [B, T, D]      │
     └──────────────────────────────┘

Weight Absorption (inference only):
  w_q_absorbed   = w_uq_nope^T @ w_uk    → fold Q projection
  w_out_absorbed = w_o @ w_uv            → fold output projection
  → decode runs entirely in latent space (kv_lora_rank=256 dims)
  → single GEMM at the end: [B,T,H*256] @ [H*256, D]
```

**Key parameters:**
- `kv_lora_rank = 256` — latent compression dimension
- `qk_nope_dim = 64` — non-positional query/key dimension per head
- `qk_rope_dim = 32` — positional (RoPE) dimension per head
- `head_dim = 96` (64 + 32), `num_heads = 16`, total = 1536 = hidden_size

### Stable LatentMoE

Each MoE layer routes tokens through a small subset of experts, operating in a compressed latent space:

```
Input x [B, T, D]
        │
        ▼
    ┌───┴───────────────────┐
    │                       │
    ▼                       ▼
┌─────────┐          ┌──────────────┐
│ Shared  │          │  Router      │
│ Expert  │          │  (linear)    │
│         │          └──────┬───────┘
│ x@w13_s │                 ▼
│ → SITU  │          ┌──────────────┐
│ → w2_s  │          │ Top-2 select │
│ → D     │          │ + QB bias    │
└────┬────┘          └──────┬───────┘
     │                      │
     │         ┌────────────▼────────────┐
     │         │  down_proj: D → L (384) │  ← latent bottleneck
     │         └────────────┬────────────┘
     │                      │
     │    ┌─────────────────▼─────────────────┐
     │    │  Packed dispatch [E, C, L]        │
     │    │  sort by expert → batched GEMM    │
     │    │  w13: L → 2I  (gate+up)          │
     │    │  SITU activation                  │
     │    │  w2: I → L                       │
     │    │  index_add_ back                  │
     │    └─────────────────┬─────────────────┘
     │                      │
     │         ┌────────────▼────────────┐
     │         │  RMSNorm on latent     │
     │         │  up_proj: L → D         │
     │         └────────────┬────────────┘
     │                      │
     ▼                      ▼
    ┌─────────────────────────┐
    │  shared_out + routed_out│
    │  [B, T, D]              │
    └─────────────────────────┘
```

**Quantile Balancing (QB) — aux-loss-free load balancing:**
1. During forward pass, record margin = `router_score - (K+1)th_score` for each token
2. After each step, compute `expert_bias = -quantile(margins, 1 - K/E)`
3. Next forward pass adds bias to router scores before top-K selection
4. Result: expert loads converge to uniform without an auxiliary loss term

**SITU-GLU activation:** `softcap(gate, β₁) · sigmoid(gate) · softcap(up, β₂)` where β₁=4.0, β₂=25.0

### Attention Residuals (AttnRes)

Instead of uniform residual summation (`h = Σ all previous`), each sub-layer learns to weight prior outputs via softmax over the depth axis:

```
blocks = [block₀, block₁, ..., blockₙ]   ← committed every 4 layers
partial = attn_output + ffn_output         ← accumulating within block

Sources = [blocks..., partial]
           │
           ▼
    ┌──────────────┐
    │  RMSNorm(s)  │
    └──────┬───────┘
           ▼
    ┌──────────────────────────┐
    │  logits = pq · norm(s)   │  pq = learnable pseudo_query
    │  α = softmax(logits)     │
    └──────┬───────────────────┘
           ▼
    ┌──────────────────────────┐
    │  output = Σ αᵢ · sᵢ     │
    └──────────────────────────┘

Every 4 layers: partial → blocks (new block committed)
At model end:   final_res(blocks, partial) → final output
```

**Why:** Residual stream dilution — each layer's contribution shrinks as 1/L, and magnitude grows unbounded. AttnRes lets the model learn which depths to attend to.

---

## Default Configuration

| Parameter | Value |
|---|---|
| **Total parameters** | 969M |
| **Active parameters** | 297M (per token, top-2 MoE) |
| Layers | 20 |
| Hidden size | 1536 |
| Attention heads | 16 × 96 head_dim |
| First K dense layers | 1 (layer 0 = dense FFN) |
| AttnRes block boundary | every 4 layers |
| Vocab size | 32,000 (Llama tokenizer) |
| Max seq len | 1024 (train) / 4096 (YaRN) |
| MoE experts | 32 routed + 1 shared |
| MoE top-K | 2 |
| MoE latent dim | 384 |
| MoE intermediate | 1024 |
| MLA kv_lora_rank | 256 |
| MLA qk_nope / qk_rope | 64 / 32 |
| Optimizer | 8-bit AdamW |
| Learning rate | 3e-4 (cosine, 500 warmup) |
| Batch | 16 micro × 32 accum = 512 seqs |
| Precision | bf16 (H100) / fp16 (T4) |

---

## Quick Start

### Install

```bash
uv sync
# or
pip install -e .
```

### Train

```bash
# Single GPU (H100/B200)
accelerate launch --num_processes=1 --num_machines=1 \
    --mixed_precision=bf16 --dynamo_backend=no \
    -m training.train

# 2 GPUs (T4/Kaggle)
accelerate launch --multi_gpu --num_processes=2 \
    --mixed_precision=fp16 --dynamo_backend=no \
    -m training.train
```

Override configs via `overrides.json` (trainer) and `model_overrides.json` (model) in the working directory.

### Load a checkpoint

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "AbdelrhmanEbied/Tiny-K3",
    trust_remote_code=True,
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "hf-internal-testing/llama-tokenizer"
)
```

### Resuming Training

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    "AbdelrhmanEbied/Tiny-K3",
    local_dir="runs/pretrain/step_2000",
)
# Set in overrides.json: "checkpoint_path": "runs/pretrain/step_2000"
```

Or use `archive/train.sh` which automates clone → install → auth → download checkpoint → launch.

---

## Training Pipeline

### Data Flow

```
HuggingFace Datasets (streaming)
        │
        ▼
┌─────────────────────┐
│  DatasetSpec        │  declarative: path/name/split/streaming/prob
│  (multi-source mix) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Buffer shuffle     │  50k sample buffer
│  Batched tokenize   │  chunks of 64
│  Append EOS         │
│  Pack to fixed len  │  no padding waste
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Strided sharding   │  rank × num_workers
│  (distributed)      │  unique slice per worker
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  DataLoader         │  pin_memory, prefetch
│  collate: clone      │
│  input_ids → labels  │
└──────────┬──────────┘
           ▼
┌─────────────────────────────────────────────────┐
│  Trainer (HF Accelerate + DeepSpeed ZeRO-1)    │
│                                                 │
│  for step in range(num_train_steps):            │
│    for micro_step in range(grad_accum):         │
│      forward → loss / grad_accum                │
│      backward                                   │
│    optimizer.step (cosine LR schedule)          │
│    QB bias sync (every qb_update_interval)      │
│    eval (every eval_interval)                   │
│    checkpoint (every save_interval)             │
│    log to W&B (every log_interval)              │
└─────────────────────────────────────────────────┘
```

### Default Dataset Mix

| Source | Probability |
|---|---|
| FineWeb-edu (sample-10BT) | 60% |
| Cosmopedia-v2 (SmolLM) | 25% |
| Open-web-math | 15% |

---

## Project Structure

```
model/
  model.py            # TinyK3Model, TransformerBlock, DenseFFN
  mla.py              # Multi-head Latent Attention
  moe.py              # Stable LatentMoE (packed dispatch)
  attn_residuals.py   # Attention Residuals (depth attention)
  rms_norm.py         # RMSNorm
  rope.py             # RoPE + YaRN
  sampling.py         # top-k/top-p/min-p/penalties sampler

training/
  train.py            # entrypoint: python -m training.train
  trainer.py          # Accelerate/DeepSpeed training loop
  data.py             # Packed streaming data pipeline
  helpers.py          # DeepSpeed config, Accelerator builder
  tokenizer.py        # TokenizerManager wrapper

configs/
  model_config.py     # ModelConfig (PretrainedConfig)
  trainer_config.py   # TrainConfig dataclass

tests/                # 179 tests (10 files)
Reports/              # Benchmark reports (3)
archive/              # Retired code & docs
  train.sh            # One-shot training launcher
baselines/            # Correctness oracles (old MoE)
benchmarks/           # Performance benchmarks
```

---

## Benchmarks

See `Reports/` for detailed reports:

| Report | Key Finding |
|---|---|
| `moe_benchmark_report.md` | Packed dispatch **5.6×–22× faster** than gather on T4, avoids OOM |
| `mla_optimization_report.md` | Split Q + fused GEMM: **1.27× prefill, 1.05–1.11× decode** on T4 |
| `kda_vs_mla_report.md` | **KDA rejected** — 3× slower than MLA on T4 (FLA kernels need sm80+) |

---

## Testing

```bash
pytest tests/ -q        # 179 tests
ruff check .            # lint
ruff format .           # format
```

CI runs lint + format + tests on push/PR (`.github/workflows/ci-cd.yaml`).

---

## License

Apache 2.0
