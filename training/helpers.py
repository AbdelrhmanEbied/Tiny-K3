import os
import shutil
from typing import Any

from accelerate import Accelerator, DeepSpeedPlugin

from configs.trainer_config import TrainConfig


def build_deepspeed_config(cfg: TrainConfig) -> dict[str, Any]:
    zero_config: dict[str, Any] = {
        "stage": cfg.zero_stage,
        "overlap_comm": cfg.overlap_comm,
        "contiguous_gradients": cfg.contiguous_gradients,
        "reduce_bucket_size": cfg.reduce_bucket_size,
    }

    if cfg.zero_stage == 1:
        zero_config.update(
            {
                "reduce_scatter": True,
            }
        )

    elif cfg.zero_stage == 2:
        zero_config.update(
            {
                "allgather_partitions": True,
                "allgather_bucket_size": cfg.reduce_bucket_size,
            }
        )

    elif cfg.zero_stage == 3:
        zero_config.update(
            {
                "stage3_prefetch_bucket_size": cfg.reduce_bucket_size // 2,
                "stage3_param_persistence_threshold": 1_000_000,
            }
        )

        if cfg.offload_param:
            zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}

    if cfg.zero_stage >= 2 and cfg.offload_optimizer:
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": cfg.micro_batch_size,
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "gradient_clipping": cfg.grad_clip,
        "wall_clock_breakdown": cfg.wall_clock_breakdown,
        "zero_optimization": zero_config,
        "activation_checkpointing": {
            "partition_activations": cfg.partition_activations,
            "cpu_checkpointing": cfg.cpu_checkpointing,
            "contiguous_memory_optimization": cfg.contiguous_memory_optimization,
            "number_checkpoints": cfg.num_checkpoints,
        },
    }

    if cfg.moe_enabled:
        ds_config["moe"] = {
            "enabled": True,
            "ep_size": cfg.ep_size,
            "moe_param_group": cfg.moe_param_group,
            "use_residual": cfg.use_residual,
        }

    if cfg.mixed_precision == "fp16":
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": cfg.loss_scale_window,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    elif cfg.mixed_precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    else:
        ds_config["fp16"] = {"enabled": False}
        ds_config["bf16"] = {"enabled": False}

    return ds_config


def has_checkpoint(path):
    return os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(
        os.path.join(path, "pytorch_model.bin")
    )


def copy_checkpoint_from_dataset(cfg: TrainConfig):
    SRC_BASE = cfg.checkpoint_path
    DST_BASE = cfg.out_dir

    os.makedirs(DST_BASE, exist_ok=True)

    if not os.path.exists(SRC_BASE):
        print(f"Input Path Doesn't Exist: {SRC_BASE}")
        return None

    subdirs = [d for d in os.listdir(SRC_BASE) if d.startswith("step_")]
    if not subdirs:
        print(f"No checkpoints found in: {SRC_BASE}, starting fresh...")
        return None

    latest = max(subdirs, key=lambda x: int(x.split("_")[-1]))

    SRC_PATH = os.path.join(SRC_BASE, latest)
    DST_PATH = os.path.join(DST_BASE, latest)

    if not os.path.exists(DST_PATH):
        print(f"Copying {latest} → working dir...")
        shutil.copytree(SRC_PATH, DST_PATH, dirs_exist_ok=True)
    else:
        print(f"{latest} already exists in working dir")

    print(f"Checkpoint ready at: {DST_PATH}")
    return DST_PATH


def build_accelerator(cfg: TrainConfig) -> Accelerator:

    ds_plugin = None
    if cfg.deepspeed_enabled:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=build_deepspeed_config(cfg))

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        log_with="wandb",
        deepspeed_plugin=ds_plugin,
    )
    return accelerator
