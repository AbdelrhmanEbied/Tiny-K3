# MLA Optimization Report

**Scope:** `model/mla.py` rewrite for inference/training efficiency, verified against the previous implementation.
**Date:** August 2026 · **Branch:** `feat/attention`

---

## 1. What changed

| # | Change | Before | After | Status |
|---|--------|--------|-------|--------|
| 1 | Q projection split | Single `w_uq_qr` Linear producing `H·(Dn+Dr)`; absorbed path computed the nope half and discarded it | Two Linears: `w_uq_nope` (H·Dn), `w_qr_rope` (H·Dr); absorbed path only computes rope | ✅ kept |
| 2 | Fused input GEMM | `w_in` (2·Dc) plus a separate `w_kr` GEMM over the same input | One `w_in` producing `[c_kv \| c_q \| k_rope]` in a single matmul | ✅ kept |
| 3 | Output projection (absorbed path) | Per-head einsum `"bhtk,dhk->btd"` | Pre-flattened buffer `w_out_absorbed_flat [H·Dc, D]`, single wide GEMM after head merge | ✅ kept |
| 4 | Causal attention fast path | Eval always built a bool mask → blocked SDPA's fused kernels | `is_causal=True` whenever `start == 0`; bool mask only for chunked/cached decode | ✅ kept |
| 5 | Grouped-Query Attention (`enable_gqa=True`) | n/a | Tried to avoid materializing head-shared K/V | ❌ reverted |

### Why #5 was reverted

Measured ~2× slower than zero-copy `.expand()` views on the CPU SDPA kernel. Since `expand()` is a stride trick that flash-style GPU kernels also handle efficiently, the portable choice is expand. Revisit only if profiling on the target hardware shows the expanded copies matter.

## 2. Correctness verification

Three independent layers of proof, all passing (64 tests):

1. **Reference math** (`tests/test_mla.py::test_mla_matches_reference_math`) — output equals a from-scratch `softmax(QKᵀ·scale)V` with explicit causal masking, no SDPA, no cache. Also: softmax rows sum to 1, no future-token leakage, gradients reach every projection.
2. **Absorption equivalence** (`test_mla_absorbed_vs_unabsorbed_equivalence`) — identical weights through the absorbed and unabsorbed paths match to 1e-5 across prefill *and* step-by-step decoding. This cross-checks two algebraically different derivations of the same computation.
3. **Old-vs-new layout equivalence** (`archive/mla_benchmark.py::test_new_matches_old_outputs`) — old weights are transferred into the new parameter layout (including the head-major `w_uq_qr` split) and outputs must match to 1e-4 across training, eval prefill, absorbed prefill, and incremental decode.

## 3. Benchmark results

Benchmark harness: interleaved old/new measurement, min-of-trials, `cuda.synchronize()` around every window, weights transferred between layouts so both run identical math. Modes via `MLA_BENCH_MODE=short|medium|long`.

Model config used: hidden=256, heads=8, kv_lora_rank=128, qk_nope_dim=64, qk_rope_dim=32.

### NVIDIA T4 (Kaggle), long mode — the authoritative numbers

| Scenario | Shape | Old | New | Speedup |
|---|---|---:|---:|---:|
| Train fwd+bwd | B=4, T=512 | 6485 µs/iter | 6501 µs/iter | 1.00x |
| Prefill (unabsorbed) | T=256 | 740 µs | 582 µs | **1.27x** |
| Decode step (unabsorbed) | 3000 steps, S→3256 | 756 µs | 723 µs | 1.05x |
| Decode step (absorbed) | 3000 steps | 756 µs | 684 µs | **1.11x** |

### Development laptop CPU (i7-3840QM, loaded machine) — smoke signals only

| Scenario | short mode | medium mode |
|---|---|---|
| Train fwd+bwd | 1.18x | — |
| Prefill | 0.94–1.26x (noisy) | 0.78–1.26x (noisy) |
| Decode unabsorbed | 1.11x | 0.94–1.01x |
| Decode absorbed | 1.10–1.51x | 1.05–1.08x |

CPU numbers carry ±30% run-to-run variance (machine load 5–7 on 8 threads) and should be treated as direction-only.

## 4. Analysis

- **Prefill gains most (1.27x)** because it is compute-bound: fewer kernel launches (fused input GEMM) plus access to SDPA's fused causal kernel (`is_causal=True` instead of a bool mask).
- **Decode at batch=1 is launch-bound on GPU**: ~700µs of dispatch overhead dominates either implementation's arithmetic, capping visible wins at ~5–11%. The gap between old and new should widen with larger decode batch sizes as GEMMs become compute-bound.
- **Absorption vs unabsorbed decode flipped between devices**: on CPU absorption was 2–3× faster (memory-bandwidth-bound recomputation of K/V over the whole past each step); on GPU at batch=1 they're nearly tied because launch overhead masks the FLOP difference. Absorption remains architecturally essential — its advantage grows with context length and batch size.
- **Training is unchanged (1.00x)** as expected: the training path was already a single causal SDPA call; the split projections perform identical FLOPs.
