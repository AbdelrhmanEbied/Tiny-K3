from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainConfig:
    lr: float = 3e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    num_train_steps: int = 9500
    warmup_steps: int = 500
    optimizer_type: Literal["AdamW", "PagedAdamW", "8bitAdamW"] = "8bitAdamW"

    # Batch / Throughput
    micro_batch_size: int = 16
    grad_accum_steps: int = 32
    max_seq_len: int = 1024

    # Precision / Performance
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    compile_model: bool = False
    compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ] = "reduce-overhead"
    enable_gradient_checkpointing: bool = True

    # Logging
    project_name: str = "Tiny-K3"
    save_interval: int = 500
    num_eval_steps: int = 25
    eval_interval: int = 250
    log_interval: int = 5
    out_dir: str = "/kaggle/working/"
    checkpoint_path: str = ""

    # DeepSpeed
    deepspeed_enabled: bool = True
    wall_clock_breakdown: bool = False
    zero_stage: int = 1
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    allgather_partitions: bool = True
    reduce_scatter: bool = True
    offload_optimizer: bool = False
    offload_param: bool = False
    partition_activations: bool = False
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = False
    num_checkpoints: int = 0
    loss_scale_window: int = 1000
    moe_enabled: bool = False
    ep_size: int = 1
    moe_param_group: bool = False
    use_residual: bool = False

    # DataLoader
    drop_last: bool = True
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True

    # Generation sampling (0 disables)
    gen_interval: int = 250
