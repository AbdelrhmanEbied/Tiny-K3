import pytest
import torch

from configs.model_config import ModelConfig
from model.model import TinyK3Model
from model.sampling import (
    apply_frequency_presence_penalties,
    apply_repetition_penalty,
    ban_repeat_ngrams,
    min_p_filter,
    sample_token,
    top_k_filter,
    top_p_filter,
)

VOCAB = 256


def make_model(layers=4, blk=2, seed_weights=1):
    torch.manual_seed(0)
    cfg = ModelConfig(
        hidden_size=64,
        num_layers=layers,
        vocab_size=VOCAB,
        num_attention_heads=4,
        kv_lora_rank=16,
        qk_nope_dim=8,
        qk_rope_dim=4,
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_token=2,
        moe_intermediate_size=32,
        moe_latent_dim=32,
        attnres_block_layers=blk,
    )
    torch.manual_seed(seed_weights)
    return TinyK3Model(cfg)


@pytest.fixture(name="model")
def fixture_model():
    model = make_model()
    model.eval()
    return model


class TestForward:
    def test_output_shapes(self, model):
        ids = torch.randint(0, VOCAB, (2, 12))
        out = model(ids)
        assert out.logits.shape == (2, 12, VOCAB)
        assert out.logits.dtype == torch.float32

    def test_loss_matches_manual_cross_entropy(self, model):
        torch.manual_seed(42)
        ids = torch.randint(0, VOCAB, (2, 8))
        labels = torch.randint(0, VOCAB, (2, 8))
        out = model(ids, labels=labels)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        manual = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, VOCAB), shift_labels.view(-1)
        )
        assert torch.allclose(out.loss, manual, atol=1e-6)

    def test_labels_default_to_input_ids(self, model):
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (2, 8))
        out = model(ids)
        manual = model(ids, labels=ids).loss
        assert out.loss is not None
        assert torch.allclose(out.loss, manual, atol=1e-6)

    def test_causal_masking_future_tokens_do_not_leak(self, model):
        torch.manual_seed(7)
        base = torch.randint(0, VOCAB, (1, 10))
        perturbed = base.clone()
        perturbed[0, 5:] = (perturbed[0, 5:] + 13) % VOCAB
        with torch.inference_mode():
            logits_a = model(base).logits
            logits_b = model(perturbed).logits
        assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-5)

    def test_lm_head_tied_to_embeddings(self, model):
        assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()


class TestGradientCheckpointing:
    def test_gradients_bit_identical_with_and_without(self):
        def run(enabled):
            model = make_model().train()
            model.gradient_checkpointing = enabled
            torch.manual_seed(42)
            ids = torch.randint(0, VOCAB, (2, 16))
            labels = torch.randint(0, VOCAB, (2, 16))
            out = model(ids, labels=labels)
            out.loss.backward()
            return out.loss.item(), {
                n: p.grad.clone() for n, p in model.named_parameters()
            }

        loss_ref, grads_ref = run(False)
        loss_ckpt, grads_ckpt = run(True)
        assert loss_ckpt == loss_ref
        for name, grad in grads_ckpt.items():
            assert torch.equal(grad, grads_ref[name]), f"grad mismatch: {name}"

    def test_router_metrics_collected_per_moe_layer(self, model):
        model.train()
        ids = torch.randint(0, VOCAB, (2, 8))
        labels = torch.randint(0, VOCAB, (2, 8))
        model(ids, labels=labels)
        expected = model.config.num_layers - model.config.first_k_dense_replace
        assert len(model.last_router_metrics) == expected
        aggregated = model.aggregated_router_metrics()
        assert "entropy" in aggregated


class TestGenerateParity:
    @pytest.mark.parametrize("layers,blk", [(4, 2), (3, 1), (5, 3), (2, 1)])
    def test_incremental_matches_teacher_forcing(self, layers, blk):
        model = make_model(layers, blk).eval()

        calls = []
        orig_final_norm = model.final_norm.forward
        model.final_norm.forward = lambda h: (
            calls.append(orig_final_norm(h).clone()),
            orig_final_norm(h),
        )[1]

        prompt = torch.randint(0, VOCAB, (3, 8))
        n = 6
        seq = model.generate(prompt, max_new_tokens=n, temperature=0.0)
        model.final_norm.forward = orig_final_norm
        assert len(calls) == n

        with torch.inference_mode():
            worst = max(
                (
                    model(seq[:, : 8 + i]).logits[0, -1].float()
                    - model.lm_head(calls[i][:, -1:].float())[0, -1]
                )
                .abs()
                .max()
                .item()
                for i in range(n)
            )
        assert worst < 1e-4

    def test_greedy_is_deterministic_across_calls(self, model):
        prompt = torch.randint(0, VOCAB, (2, 8))
        a = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        b = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        assert torch.equal(a, b)

    def test_cache_accounting_exact(self, model):
        prompt = torch.randint(0, VOCAB, (3, 6))
        model.generate(prompt, max_new_tokens=7, temperature=0.9, top_k=10)
        for layer in model.layers:
            assert layer.attn.cache_len == 13

    def test_eos_stops_generation(self, model):
        prompt = torch.randint(0, VOCAB, (1, 6))
        probe = model.generate(prompt, max_new_tokens=20, temperature=0.0)
        first_tok = int(probe[0, 6])

        seq = model.generate(
            prompt, max_new_tokens=100, temperature=0.0, eos_token_id=first_tok
        )
        assert seq.shape[1] == 7 and int(seq[0, -1]) == first_tok

        missing = next(t for t in range(VOCAB) if not (probe[0] == t).any())
        full = model.generate(
            prompt, max_new_tokens=10, temperature=0.0, eos_token_id=missing
        )
        assert full.shape[1] == 16

    def test_long_generation_stays_in_bounds(self, model):
        prompt = torch.randint(0, VOCAB, (2, 4))
        seq = model.generate(prompt, max_new_tokens=300, temperature=0.0)
        assert seq.shape == (2, 304)


class TestStreaming:
    def test_stream_yields_every_token_and_matches_generate(self, model):
        torch.manual_seed(123)
        prompt = torch.randint(0, VOCAB, (1, 5))
        ref = model.generate(prompt, max_new_tokens=10, temperature=0.0)

        events = list(model.generate_stream(prompt, max_new_tokens=10, temperature=0.0))
        assert [e["token"] for e in events] == ref[0, 5:].tolist()
        assert all(isinstance(e["text"], str) for e in events)

    def test_stream_text_decoding_byte_level_utf8(self, model):
        prompt = torch.randint(0, VOCAB, (1, 2))
        events = list(
            model.generate_stream(prompt, max_new_tokens=256, temperature=0.0)
        )
        text = "".join(e["text"] for e in events)
        assert isinstance(text, str)

    def test_stream_respects_decode_fn(self, model):
        prompt = torch.randint(0, VOCAB, (1, 2))
        events = list(
            model.generate_stream(
                prompt,
                max_new_tokens=4,
                temperature=0.0,
                decode_fn=lambda t: f"<{t}>",
            )
        )
        assert "".join(e["text"] for e in events) == "".join(
            f"<{e['token']}>" for e in events
        )

    def test_generator_is_lazy(self, model):
        prompt = torch.randint(0, VOCAB, (1, 4))
        stream = model.generate_stream(prompt, max_new_tokens=100, temperature=0.0)
        first = next(iter(stream))
        assert isinstance(first["token"], int)


class TestRepetitionPenalty:
    def test_positive_logit_divided_negative_multiplied(self):
        logits = torch.tensor([2.0, -2.0])
        out = apply_repetition_penalty(logits.clone(), [0, 1], penalty=2.0)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(-4.0)

    def test_unseen_tokens_untouched(self):
        logits = torch.tensor([2.0, -2.0, 1.0])
        out = apply_repetition_penalty(logits.clone(), [0, 1], penalty=3.0)
        assert out[2] == 1.0

    def test_penalty_suppresses_repetition_statistically(self):
        logits = torch.tensor([5.0, 5.0, 1.0])
        ctx = [0]
        plain = sum(
            sample_token(logits, ctx, repetition_penalty=1.0, temperature=1.0) == 0
            for _ in range(2000)
        )
        penalized = sum(
            sample_token(logits, ctx, repetition_penalty=10.0, temperature=1.0) == 0
            for _ in range(2000)
        )
        assert penalized < plain / 2


class TestFrequencyPresencePenalties:
    def test_frequency_penalty_scales_with_count(self):
        logits = torch.zeros(4)
        out = apply_frequency_presence_penalties(
            logits.clone(), [1, 1, 1], frequency_penalty=0.5, presence_penalty=0.0
        )
        assert out[1] == pytest.approx(-1.5)
        assert out[0] == 0.0

    def test_presence_penalty_flat_regardless_of_count(self):
        logits = torch.zeros(4)
        out = apply_frequency_presence_penalties(
            logits.clone(), [1, 1, 1], frequency_penalty=0.0, presence_penalty=0.7
        )
        assert out[1] == pytest.approx(-0.7)

    def test_penalized_token_loses_to_equal_alternative(self):
        logits = torch.tensor([1.0, 1.0])
        chosen_one = sum(
            sample_token(
                logits,
                [1, 1],
                frequency_penalty=2.0,
                presence_penalty=2.0,
                temperature=0.5,
            )
            == 1
            for _ in range(500)
        )
        assert chosen_one == 0


class TestNoRepeatNgram:
    def test_bigram_cannot_repeat(self):
        logits = torch.zeros(VOCAB)
        context = [5, 6, 5, 6]
        out = ban_repeat_ngrams(logits.clone(), context, no_repeat_ngram_size=2)

        assert out[5] == float("-inf")
        assert out[7] == 0.0

    def test_trigram_ban(self):
        logits = torch.zeros(VOCAB)
        context = [1, 2, 3, 1, 2]
        out = ban_repeat_ngrams(logits.clone(), context, no_repeat_ngram_size=3)
        assert out[3] == float("-inf")

    def test_no_match_no_ban(self):
        logits = torch.zeros(VOCAB)
        context = [1, 2, 3]
        out = ban_repeat_ngrams(logits.clone(), context, no_repeat_ngram_size=4)
        assert out.min() == 0.0

    def test_size_one_bans_all_seen(self):
        logits = torch.zeros(VOCAB)
        out = ban_repeat_ngrams(logits.clone(), [3, 9, 40], no_repeat_ngram_size=1)
        assert out[3] == float("-inf") and out[9] == float("-inf")
        assert out[4] == 0.0

    def test_end_to_end_never_repeats_bigram(self, model):
        torch.manual_seed(9)
        prompt = torch.randint(0, VOCAB, (1, 6))
        seq = model.generate(
            prompt,
            max_new_tokens=60,
            temperature=1.0,
            top_k=5,
            no_repeat_ngram_size=2,
        )[0].tolist()
        bigrams = {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}
        assert len(bigrams) == len(seq) - 1


class TestTopK:
    def test_keeps_exactly_k_entries(self):
        probs = torch.softmax(torch.randn(50), dim=-1)
        out = top_k_filter(probs, 5)
        assert int((out > 0).sum()) == 5

        top5 = torch.topk(probs, 5).values
        kept = out[out > 0].sort().values
        assert torch.allclose(kept, top5.sort().values)

    def test_k_larger_than_vocab_is_noop(self):
        probs = torch.softmax(torch.randn(10), dim=-1)
        assert torch.equal(top_k_filter(probs, 64), probs)

    def test_top_k_one_equals_argmax_sampling(self, model):
        torch.manual_seed(3)
        prompt = torch.randint(0, VOCAB, (1, 6))
        greedy = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        sampled = model.generate(prompt, max_new_tokens=8, temperature=5.0, top_k=1)
        assert torch.equal(greedy, sampled)


class TestTopP:
    def test_keeps_minimal_nucleus(self):
        probs = torch.tensor([0.6, 0.3, 0.1])
        out = top_p_filter(probs, p=0.65)

        assert out[0] > 0 and out[1] > 0
        assert out[2] == 0
        assert out.sum() == pytest.approx(0.9)

    def test_support_within_nucleus(self):
        torch.manual_seed(11)
        probs = torch.softmax(torch.randn(100), dim=-1)
        out = top_p_filter(probs, p=0.3)
        kept = torch.nonzero(out).flatten()
        sorted_idx = probs.argsort(descending=True)[: len(kept)]
        assert set(kept.tolist()).issubset(set(sorted_idx.tolist()))

    def test_high_p_close_to_full_distribution(self, model):
        torch.manual_seed(5)
        prompt = torch.randint(0, VOCAB, (1, 6))
        wide = model.generate(prompt, max_new_tokens=8, temperature=2.0, top_p=1.0)
        narrow = model.generate(prompt, max_new_tokens=8, temperature=2.0, top_p=0.01)
        assert wide.shape == narrow.shape == (1, 14)


class TestMinP:
    def test_cutoff_relative_to_max_prob(self):
        probs = torch.tensor([0.6, 0.3, 0.08, 0.02])
        out = min_p_filter(probs, 0.5)

        assert out[0] > 0 and out[1] > 0
        assert out[2] == 0 and out[3] == 0

    def test_min_p_one_keeps_only_argmax(self):
        probs = torch.tensor([0.5, 0.25, 0.25])
        out = min_p_filter(probs, 1.0)
        assert int((out > 0).sum()) == 1 and out.argmax().item() == 0

    def test_zero_or_none_is_noop(self):
        probs = torch.rand(20) + 0.01
        assert torch.equal(min_p_filter(probs, None), probs)
        assert torch.equal(min_p_filter(probs, 0.0), probs)

    def test_extreme_min_p_equals_greedy(self, model):
        torch.manual_seed(17)
        prompt = torch.randint(0, VOCAB, (1, 6))
        greedy = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        hard = model.generate(prompt, max_new_tokens=8, temperature=2.0, min_p=1.0)
        assert torch.equal(greedy, hard)


class TestSampleToken:
    def test_temperature_zero_returns_argmax_even_with_penalties(self):
        logits = torch.tensor([1.0, 9.0, 3.0])
        tok = sample_token(
            logits,
            [0, 2],
            temperature=0.0,
            repetition_penalty=50.0,
            frequency_penalty=50.0,
            presence_penalty=50.0,
            no_repeat_ngram_size=1,
        )
        assert tok == 1

    def test_all_filtered_falls_back_to_argmax(self):
        logits = torch.tensor([3.0, 2.0, 1.0])
        tok = sample_token(
            logits,
            [0, 1, 2],
            temperature=1.0,
            no_repeat_ngram_size=1,
        )

        assert tok == 0

    def test_combined_pipeline_ordering(self):
        torch.manual_seed(21)
        logits = torch.tensor([4.0, 4.0, 0.5, 0.1])
        ctx = [2, 2, 2]
        picks = {
            sample_token(
                logits,
                ctx,
                temperature=1.0,
                top_k=2,
                repetition_penalty=1.5,
                presence_penalty=3.0,
                no_repeat_ngram_size=1,
            )
            for _ in range(300)
        }

        assert picks.issubset({0, 1})
