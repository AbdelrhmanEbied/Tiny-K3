from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "tiny_k3"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1536,
        num_layers: int = 20,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        max_seq_len: int = 1024,
        # RMSNorm
        rms_norm_eps: float = 1e-6,
        # RoPE & YaRN
        rope_theta: float = 10000.0,
        rope_type: str = "default",
        beta_slow: float = 1.0,
        beta_fast: float = 32.0,
        factor: float = 1.0,
        original_max_seq_len: int = 1024,
        mscale: float = 1.0,
        # MLA
        num_attention_heads: int = 16,
        kv_lora_rank: int = 256,
        qk_nope_dim: int = 64,
        qk_rope_dim: int = 32,
        absorb_weights: bool = False,
        # MoE
        num_experts: int = 32,
        num_shared_experts: int = 1,
        num_experts_per_token: int = 2,
        moe_intermediate_size: int = 1024,
        moe_latent_dim: int = 384,
        situ_beta_gate: float = 4.0,
        situ_beta_up: float = 25.0,
        moe_capacity_factor: float = 1.25,
        # Architecture
        first_k_dense_replace: int = 1,
        attnres_block_layers: int = 4,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.initializer_range = initializer_range
        self.max_seq_len = max_seq_len

        # RMSNorm
        self.rms_norm_eps = rms_norm_eps

        # RoPE & YaRN
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.beta_slow = beta_slow
        self.beta_fast = beta_fast
        self.factor = factor
        self.original_max_seq_len = original_max_seq_len
        self.mscale = mscale

        # MLA
        self.num_attention_heads = num_attention_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.absorb_weights = absorb_weights

        # MoE
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_dim = moe_latent_dim
        self.num_shared_experts = num_shared_experts
        self.situ_beta_gate = situ_beta_gate
        self.situ_beta_up = situ_beta_up
        self.moe_capacity_factor = moe_capacity_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.attnres_block_layers = attnres_block_layers
