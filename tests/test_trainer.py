from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset

from configs.model_config import ModelConfig
from tests.test_data import collate_fn, make_cfg
from training.trainer import train

VOCAB = 64
SEQ_LEN = 8


class RecordingAccelerator:
    """Delegates everything to a plain CPU Accelerator, records log() calls."""

    def __init__(self) -> None:
        self.inner = Accelerator()
        self.logs: list[tuple[int | None, dict]] = []

    def log(self, data: dict, step: int | None = None) -> None:
        self.logs.append((step, dict(data)))

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class TinyStream(IterableDataset):
    def __init__(
        self, num_samples: int = 4, vocab: int = VOCAB, seq_len: int = SEQ_LEN
    ):
        super().__init__()
        self.num_samples = num_samples
        self.vocab = vocab
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        g = torch.Generator().manual_seed(0)
        for _ in range(self.num_samples):
            yield {
                "input_ids": torch.randint(0, self.vocab, (self.seq_len,), generator=g)
            }


class GenTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str | None = None):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return f"decoded({list(ids)})"


def make_model() -> torch.nn.Module:
    cfg = ModelConfig(
        vocab_size=VOCAB,
        hidden_size=16,
        num_layers=2,
        num_attention_heads=2,
        kv_lora_rank=8,
        qk_nope_dim=4,
        qk_rope_dim=4,
        num_experts=2,
        num_experts_per_token=1,
        moe_intermediate_size=16,
        moe_latent_dim=8,
        attnres_block_layers=1,
    )
    from model.model import TinyK3Model

    model = TinyK3Model(cfg)
    model.config.max_seq_len = max(model.config.max_seq_len, 128)
    return model


@pytest.fixture()
def trainer_env(monkeypatch):
    recorder = RecordingAccelerator()

    def fake_build(cfg):
        recorder.inner = Accelerator()
        recorder.logs.clear()
        return recorder

    monkeypatch.setattr("training.trainer.build_accelerator", fake_build)
    monkeypatch.setattr(
        "training.trainer.copy_checkpoint_from_dataset", lambda *a, **k: None
    )
    return recorder


def run_train(out_dir, steps, tokenizer=None, **overrides):
    defaults: dict = {
        "num_train_steps": steps,
        "micro_batch_size": 2,
        "grad_accum_steps": 1,
        "save_interval": max(1, steps),
        "eval_interval": 10_000,
        "log_interval": 1,
        "out_dir": str(out_dir),
        "enable_gradient_checkpointing": False,
        "compile_model": False,
        "deepspeed_enabled": False,
    }
    defaults.update(overrides)
    cfg = make_cfg(**defaults)

    dl = DataLoader(
        TinyStream(num_samples=8),
        batch_size=cfg.micro_batch_size,
        collate_fn=collate_fn,
    )
    train(make_model(), dl, None, cfg, qb_update_interval=0, tokenizer=tokenizer)
    return cfg


class TestTrainEndToEnd:
    def test_runs_and_saves_final_checkpoint(self, tmp_path, trainer_env):
        run_train(tmp_path / "out", steps=2)

        final_dir = tmp_path / "out" / "step_2"
        assert (final_dir / "model.safetensors").exists()
        assert (final_dir / "config.json").exists()

        meta = torch.load(final_dir / "metadata.pt", map_location="cpu")
        assert meta["step"] == 2
        assert meta["tokens_seen"] > 0
        assert "best_loss" in meta and "best_step" in meta

    def test_logs_expected_metric_names(self, tmp_path, trainer_env):
        run_train(tmp_path / "out", steps=2)

        logged_keys: set[str] = set()
        for _, data in trainer_env.logs:
            logged_keys.update(data)

        for key in (
            "train/loss",
            "train/learning_rate",
            "train/grad_norm",
            "train/weight_norm",
            "train/tokens_per_sec",
            "train/step_time",
            "train/tokens_seen",
        ):
            assert key in logged_keys, f"missing {key}"

        moe_keys = {k for k in logged_keys if k.startswith("moe/")}
        assert "moe/overflow_rate" in moe_keys
        assert "moe/dropped_tokens" in moe_keys

        assert all(step == i + 1 for i, (step, _) in enumerate(trainer_env.logs))

    def test_lr_follows_warmup_then_decay(self, tmp_path, trainer_env):
        run_train(tmp_path / "out", steps=4, warmup_steps=2)

        lrs = [data["train/learning_rate"] for _, data in trainer_env.logs]
        assert len(lrs) == 4
        assert 0.0 < lrs[0] < lrs[1] <= 1e-4
        assert lrs[3] < lrs[1]

    def test_loss_is_finite_and_positive(self, tmp_path, trainer_env):
        run_train(tmp_path / "out", steps=2)
        losses = [data["train/loss"] for _, data in trainer_env.logs]
        assert all(loss > 0 and torch.isfinite(torch.tensor(loss)) for loss in losses)


class TestResume:
    def test_resume_extends_training_from_checkpoint(self, tmp_path, trainer_env):
        out = tmp_path / "out"
        run_train(out, steps=2)
        assert (out / "step_2" / "metadata.pt").exists()

        run_train(out, steps=4)

        meta = torch.load(out / "step_4" / "metadata.pt", map_location="cpu")
        assert meta["step"] == 4
        assert meta["tokens_seen"] > 0

    def test_resume_preserves_tokens_seen_counter(self, tmp_path, trainer_env):
        out = tmp_path / "out"
        run_train(out, steps=2)
        first_meta = torch.load(out / "step_2" / "metadata.pt", map_location="cpu")

        logs_before = sum(d["train/tokens_seen"] for _, d in trainer_env.logs)

        run_train(out, steps=3)
        resumed_first_log = trainer_env.logs[0][1]["train/tokens_seen"]
        assert resumed_first_log >= first_meta["tokens_seen"]
        assert logs_before > 0


class TestGenerationSampling:
    @pytest.fixture(autouse=True)
    def tiny_generation(self, monkeypatch):
        """1 prompt x 1 config x 4 tokens so sampling stays cheap."""
        monkeypatch.setattr("training.trainer._GENERATION_PROMPTS", ("hello",))
        monkeypatch.setattr(
            "training.trainer._GENERATION_CONFIGS", {"greedyish": {"top_k": 2}}
        )
        monkeypatch.setattr("training.trainer._NUM_NEW_TOKENS", 4)

    def test_generation_files_written_per_interval(self, tmp_path, trainer_env):
        tok = GenTokenizer()
        run_train(tmp_path / "out", steps=3, gen_interval=1, tokenizer=tok)

        gen_dir = tmp_path / "out" / "generations"
        assert gen_dir.exists()
        for step in (1, 2, 3):
            path = gen_dir / f"step_{step}.txt"
            assert path.exists(), f"missing {path}"
            content = path.read_text(encoding="utf-8")
            assert f"=== step {step} ===" in content
            assert "--- hello [greedyish] ---" in content
            assert "decoded(" in content

    def test_no_generation_files_when_disabled(self, tmp_path, trainer_env):
        run_train(tmp_path / "out", steps=2, gen_interval=0)
        assert not (tmp_path / "out" / "generations").exists()

    def test_missing_tokenizer_skips_gracefully(self, tmp_path, trainer_env, capsys):
        run_train(tmp_path / "out", steps=1, gen_interval=1, tokenizer=None)
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert not (tmp_path / "out" / "generations").exists()
        assert "no tokenizer passed" in combined or combined != ""
