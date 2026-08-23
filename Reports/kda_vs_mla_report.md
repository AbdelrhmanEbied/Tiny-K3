# KDA vs MLA — Head-to-Head Benchmark Report

**Harness:** `archive/kda_vs_mla_benchmark.py` · **Status: CLOSED — hybrid KDA rejected**
**Date:** August 2026 · **Branch:** `feat/attention` · **Companion:** `kda_report.md`, `mla_optimization_report.md`

> **DECISION:** After running the benchmarks we decided **not to use a hybrid KDA
> architecture** — it is not worth it for this project. As a result, `model/kda.py` and
> `tests/test_kda.py` were moved to `archive/` and all KDA fields were removed from
> `ModelConfig`. The model proceeds with pure MLA attention.
>
> The main reason is hardware: training happens on **T4 GPUs (sm75), which are not
> supported by the FLA Triton kernels** KDA's performance depends on. Without fused
> kernels, KDA runs as plain PyTorch and loses to MLA by ~3x across every phase — so its
> only remaining advantage would be memory, at an unacceptable speed cost.

---

## 1. What was compared

Both attention layers received **the exact same workload** — identical hidden size, identical
head count, identical per-head QK/V width, identical sequences. The only variable is the
architecture inside the layer.

| | MLA | KDA |
|---|---|---|
| Hidden size | 256 | 256 |
| Heads × head dim | 8 × 64 (`qk_nope=56 + rope=8`) | 8 × 64 |
| Memory strategy | compressed KV latent (`kv_lora_rank=128`) + growing `kv/pe` cache + weight absorption | fixed recurrent state `[B,H,K,V]` + conv tails + learned per-channel decay |
| Implementation | optimized torch + SDPA (+ absorption path) | pure PyTorch chunkwise / recurrent |

Three phases measured with interleaved min-of-trials timing and CUDA sync:

1. **TRAIN** — forward+backward on a full teacher-forced sequence
2. **PREFILL** — single eval pass over a prompt
3. **DECODE** — autoregressive single-token steps with state carried across calls

Modes via `MLA_BENCH_MODE=short|medium|long` control batch size, sequence lengths, and step
counts (short: 100 steps / medium: 1000 / long: 3000 at B=4).

## 2. Results — local CPU (i7-3840QM, loaded dev machine), short mode

```
=== KDA vs MLA | device=cpu | mode=short ===
TRAIN  fwd+bwd (B=1,T=128):  MLA    13644 us | KDA    47040 us | 0.29x
PREFILL (T=64):              MLA     3577 us | KDA    11968 us | 0.30x
DECODE (100 steps):          MLA      792 us/step | KDA     2094 us/step | 0.38x
CACHE memory (B=1):          MLA 528 KiB growing | KDA 137 KiB CONSTANT
```

## 3. Interpretation

**Speed: MLA wins ~3x across every phase right now.** This is an *implementation* gap, not an
architecture verdict. SDPA and dense GEMMs are among the most heavily optimized kernels in
existence; KDA currently runs as pure PyTorch — batched einsums plus a triangular solve that
torch neither fuses nor autotunes.

The critical hardware constraint: KDA only reaches competitive speed through the FLA project's
fused Triton kernels, which officially target **sm80+ GPUs — our T4 training cards (sm75) are
not supported**. Without them there is no path to closing this gap on our training hardware,
which makes hybrid-KDA a memory optimization purchased at ~3x training-speed cost. Not worth it.

**Memory: KDA wins by design.** 137 KiB constant regardless of context vs 528 KiB at only
512 max positions for MLA — but at our model scale and context lengths, MLA's cache is
affordable, and the constant-vs-growing distinction doesn't outweigh the speed loss.

