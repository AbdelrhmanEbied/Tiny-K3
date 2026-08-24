"""Benchmark: gather-based dispatch vs packed dispatch for the MoE layer.

Detects CUDA and falls back to CPU. Generates Reports/moe_benchmark_report.md.
"""

import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.model_config import ModelConfig
from model.moe import MoE

REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "Reports" / "moe_benchmark_report.md"
)


def pick_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"device: cuda ({name})")
    else:
        print("device: cpu (no gpu found)")
    return "cuda" if torch.cuda.is_available() else "cpu"


def gather_forward(moe: MoE, x: torch.Tensor):
    bsz, seq_len, hidden = x.shape
    x_flat = x.reshape(-1, hidden)
    N = x_flat.shape[0]
    K = moe.topk

    gate_up_shared = torch.einsum("nd,sdi->nsi", x_flat, moe.w13_shared)
    gate_s, up_s = gate_up_shared.chunk(2, dim=-1)
    shared_h = F.silu(gate_s) * up_s
    shared_out = torch.einsum("nsi,sid->nd", shared_h, moe.w2_shared)

    topk_idx, weights = moe.route(x_flat)
    z = moe.down_proj(x_flat)

    w13_sel = moe.w13[topk_idx]  # [N,K,L,2I]
    w2_sel = moe.w2[topk_idx]  # [N,K,I,L]

    z_per_slot = z.unsqueeze(1).expand(N, K, moe.moe_latent_dim)
    gate_up = torch.einsum("nkl,nkli->nki", z_per_slot, w13_sel)
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate) * up
    expert_out = torch.einsum("nki,nkil->nkl", act, w2_sel)
    u = (expert_out * weights.to(expert_out.dtype).unsqueeze(-1)).sum(dim=1)

    routed_out = moe.up_proj(moe.norm(u))
    output = (shared_out + routed_out).view(bsz, seq_len, hidden)
    return output


def packed_forward(moe: MoE, x: torch.Tensor):
    bsz, seq_len, hidden = x.shape
    x_flat = x.reshape(-1, hidden)
    N = x_flat.shape[0]
    E = moe.num_experts
    K = moe.topk

    gate_up_shared = torch.einsum("nd,sdi->nsi", x_flat, moe.w13_shared)
    gate_s, up_s = gate_up_shared.chunk(2, dim=-1)
    shared_h = F.silu(gate_s) * up_s
    shared_out = torch.einsum("nsi,sid->nd", shared_h, moe.w2_shared)

    topk_idx, weights = moe.route(x_flat)
    z = moe.down_proj(x_flat)

    expert_ids = topk_idx.reshape(-1)
    token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, K).reshape(-1)
    gates = weights.reshape(-1).to(z.dtype)

    expert_ids, perm = torch.sort(expert_ids)
    token_ids = token_ids[perm]
    gates = gates[perm]

    counts = torch.bincount(expert_ids, minlength=E)
    capacity = int(counts.max().item())
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    slot_ids = torch.arange(N * K, device=x.device) - starts[expert_ids]

    packed = z.new_zeros(E, capacity, moe.moe_latent_dim)
    packed[expert_ids, slot_ids] = z[token_ids]

    proj = torch.matmul(packed, moe.w13)
    gate, up = proj.chunk(2, dim=-1)
    act = F.silu(gate) * up
    expert_out = torch.matmul(act, moe.w2)

    valid = expert_out[expert_ids, slot_ids]
    u = z.new_zeros(N, moe.moe_latent_dim)
    u.index_add_(0, token_ids, valid * gates.unsqueeze(-1))

    routed_out = moe.up_proj(moe.norm(u))
    output = (shared_out + routed_out).view(bsz, seq_len, hidden)
    return output


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, warmup: int, iters: int, label: str, device: str) -> float:
    for i in range(warmup):
        fn()
        _sync(device)
        print(f"\r  {label}: warmup {i + 1}/{warmup}", end="", flush=True)
    start = perf_counter()
    for i in range(iters):
        fn()
        _sync(device)
        elapsed = (perf_counter() - start) / (i + 1) * 1000.0
        print(
            f"\r  {label}: iter {i + 1}/{iters} ({elapsed:.1f} ms/iter)",
            end="",
            flush=True,
        )
    total_ms = (perf_counter() - start) / iters * 1000.0
    print(f"\r  {label}: done ({total_ms:.2f} ms/iter)")
    return total_ms


def is_oom(err: RuntimeError) -> bool:
    msg = str(err).lower()
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return "allocate" in msg or "memory" in msg


def run_case(
    name: str, cfg_kwargs: dict, batch: int, seq: int, iters: int, device: str
):
    torch.manual_seed(0)
    cfg = ModelConfig(**cfg_kwargs)
    moe = MoE(cfg).to(device)
    moe.train()
    x = torch.randn(batch, seq, cfg.hidden_size, device=device, requires_grad=True)

    with torch.inference_mode():
        base = moe(x)[0]
        assert torch.allclose(base, gather_forward(moe, x), atol=1e-4), (
            f"{name}: gather != module"
        )
        packed = packed_forward(moe, x)
    assert torch.allclose(base, packed, atol=1e-4), f"{name}: packed != module"

    variants = {
        "gather fwd": lambda: gather_forward(moe, x),
        "packed fwd": lambda: packed_forward(moe, x),
        "gather fwd+bwd": lambda: (
            moe.zero_grad(set_to_none=False),
            gather_forward(moe, x).square().mean().backward(),
        ),
        "packed fwd+bwd": lambda: (
            moe.zero_grad(set_to_none=False),
            packed_forward(moe, x).square().mean().backward(),
        ),
    }

    n_tokens = batch * seq
    rows = {}
    header = (
        f"=== {name}: B={batch} T={seq} ({n_tokens} tokens) | "
        f"D={cfg.hidden_size} L={cfg.moe_latent_dim} E={cfg.num_experts} "
        f"K={cfg.num_experts_per_token} I={cfg.moe_intermediate_size} ==="
    )
    print(f"\n{header}", flush=True)

    peak_mem_mb = None
    for label, fn in variants.items():
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ms = timed(fn, warmup=3, iters=iters, label=label, device=device)
            rows[label] = (ms, n_tokens / (ms / 1000.0))
            if device == "cuda":
                peak_mem_mb = max(
                    peak_mem_mb or 0, torch.cuda.max_memory_allocated() / 1024**2
                )
        except RuntimeError as e:
            if is_oom(e):
                print(f"\r  {label}: OOM" + " " * 40)
                rows[label] = None
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

    print(f"\n{'variant':<18} {'ms/iter':>10} {'tokens/s':>12}")
    for label, r in rows.items():
        if r is None:
            print(f"{label:<18} {'OOM':>10}")
            continue
        ms, tps = r
        mem = (
            f"   [peak {peak_mem_mb:.0f} MB]"
            if (label.endswith("fwd+bwd") and peak_mem_mb)
            else ""
        )
        print(f"{label:<18} {ms:>10.2f} {tps:>12,.0f}{mem}")

    speedup = None
    if rows["gather fwd+bwd"] and rows["packed fwd+bwd"]:
        speedup = rows["gather fwd+bwd"][0] / rows["packed fwd+bwd"][0]
        print(f"--> training speedup (packed/gather): {speedup:.2f}x")

    return {
        "name": name,
        "batch": batch,
        "seq": seq,
        "n_tokens": n_tokens,
        "cfg": cfg,
        "rows": rows,
        "speedup": speedup,
        "peak_mem_mb": peak_mem_mb,
    }


def write_report(results: list[dict], device: str):
    lines = [
        "# MoE Dispatch Benchmark: Gather vs Packed",
        "",
        f"- device: `{device}`"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""),
        f"- torch: {torch.__version__}",
        f"- threads: {torch.get_num_threads()}",
        "- correctness: packed output verified against gather baseline (`atol=1e-4`) before timing",
        "",
        "Dispatch strategies compared:",
        "",
        "- **gather**: `w13[topk_idx]` copies each selected expert's weights per token-slot -> `[N,K,L,2I]` intermediate",
        "- **packed**: tokens sorted by expert into `[E,C,L]`, one batched GEMM per expert weight, `index_add_` combine",
        "",
    ]
    for r in results:
        cfg = r["cfg"]
        lines += [
            f"## {r['name']} — B={r['batch']} T={r['seq']} ({r['n_tokens']} tokens)",
            "",
            (
                f"D={cfg.hidden_size}, L={cfg.moe_latent_dim}, E={cfg.num_experts}, "
                f"K={cfg.num_experts_per_token}, I={cfg.moe_intermediate_size}"
            ),
            "",
            "| variant | ms/iter | tokens/s |",
            "|---|---:|---:|",
        ]
        for label, val in r["rows"].items():
            if val is None:
                lines.append(f"| {label} | OOM | — |")
            else:
                lines.append(f"| {label} | {val[0]:.2f} | {val[1]:,.0f} |")
        if r["speedup"]:
            lines.append("")
            lines.append(f"**training speedup (packed/gather): {r['speedup']:.2f}x**")
        if r["peak_mem_mb"]:
            lines.append(
                f"*peak gpu memory across variants: {r['peak_mem_mb']:.0f} MB*"
            )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nreport written to {REPORT_PATH}")


def main():
    device = pick_device()
    print(f"threads: {torch.get_num_threads()}", flush=True)

    results = [
        run_case("short", {}, batch=1, seq=64, iters=20, device=device),
        run_case("medium", {}, batch=4, seq=128, iters=10, device=device),
        run_case("long", {}, batch=4, seq=256, iters=5, device=device),
        run_case(
            "wide",
            {
                "num_experts": 64,
                "num_experts_per_token": 6,
                "moe_latent_dim": 512,
                "moe_intermediate_size": 1024,
            },
            batch=116,
            seq=512,
            iters=10,
            device=device,
        ),
    ]

    write_report(results, device)


if __name__ == "__main__":
    main()
