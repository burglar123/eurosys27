import torch
import torch.nn.functional as F
from torch import nn
from transformers import OPTConfig


from nano_pearl.layers.attention import Attention
from nano_pearl.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nano_pearl.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
    ReplicatedLinear,
)
from nano_pearl.pearl_config import TPParams


def _get_activation_fn(name: str):
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name in ("gelu_new", "gelu_fast", "quick_gelu"):
        # 先统一回到 gelu，后续如需精确对齐可再细分
        return F.gelu
    raise ValueError(f"Unsupported OPT activation_function: {name}")


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    OPT uses learned positional embeddings with an offset=2.
    HF OPT also applies this offset. In this engine, `positions` are already
    absolute token positions prepared by the scheduler/runner, so we can index
    directly with positions + offset.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return F.embedding(positions + self.offset, self.weight)


class OPTAttention(nn.Module):
    def __init__(self, config: OPTConfig, tp_params: TPParams):
        super().__init__()
        tp_size = tp_params.tp_size

        self.embed_dim = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0, (
            f"num_attention_heads={self.total_num_heads} must be divisible by tp_size={tp_size}"
        )

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.embed_dim // self.total_num_heads
        self.scaling = self.head_dim ** -0.5

        enable_bias = getattr(config, "enable_bias", True)

        # OPT is standard MHA, not GQA/MQA.
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            tp_params=tp_params,
            bias=enable_bias,
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            tp_params=tp_params,
            bias=enable_bias,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_heads,
            tp_params=tp_params,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        local_qkv_dim = self.num_heads * self.head_dim

        q, k, v = qkv.split(
            [local_qkv_dim, local_qkv_dim, local_qkv_dim],
            dim=-1,
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # No RoPE for OPT.
        o = self.attn(q, k, v)
        return self.out_proj(o.flatten(1, -1))


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, tp_params: TPParams):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.do_layer_norm_before = config.do_layer_norm_before
        self.activation_fn = _get_activation_fn(config.activation_function)

        ln_affine = getattr(config, "layer_norm_elementwise_affine", True)
        enable_bias = getattr(config, "enable_bias", True)

        self.self_attn = OPTAttention(config, tp_params)

        # 名字尽量和 HF OPT 对齐，方便 load_model 直接匹配
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=ln_affine,
        )
        self.fc1 = ColumnParallelLinear(
            input_size=self.embed_dim,
            output_size=config.ffn_dim,
            tp_params=tp_params,
            bias=enable_bias,
        )
        self.fc2 = RowParallelLinear(
            input_size=config.ffn_dim,
            output_size=self.embed_dim,
            tp_params=tp_params,
            bias=enable_bias,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=ln_affine,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention block
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # FFN block
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class OPTDecoder(nn.Module):
    def __init__(self, config: OPTConfig, tp_params: TPParams):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.word_embed_proj_dim = config.word_embed_proj_dim
        self.hidden_size = config.hidden_size

        # 名字与 HF 对齐：embed_tokens / embed_positions / project_in / project_out / final_layer_norm / layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            tp_params,
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        # Generic OPT compatibility. For opt-30b these are both None because
        # word_embed_proj_dim == hidden_size.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(
                input_size=config.word_embed_proj_dim,
                output_size=config.hidden_size,
                tp_params=tp_params,
                bias=False,
            )
            self.project_out = ReplicatedLinear(
                input_size=config.hidden_size,
                output_size=config.word_embed_proj_dim,
                tp_params=tp_params,
                bias=False,
            )
        else:
            self.project_in = None
            self.project_out = None

        remove_final_ln = getattr(config, "_remove_final_layer_norm", False)
        ln_affine = getattr(config, "layer_norm_elementwise_affine", True)

        if config.do_layer_norm_before and not remove_final_ln:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=ln_affine,
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config, tp_params) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if self.project_in is not None:
            hidden_states = self.project_in(hidden_states)

        pos_embeds = self.embed_positions(positions)
        hidden_states = hidden_states + pos_embeds

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTModel(nn.Module):
    """
    Keep a nested `decoder` module so parameter names match HF OPT checkpoints:
    `model.decoder.*`
    """
    def __init__(self, config: OPTConfig, tp_params: TPParams):
        super().__init__()
        self.decoder = OPTDecoder(config, tp_params)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions)


class OPTForCausalLM(nn.Module):
    # Only pack q/k/v. Everything else is easier to load by identical HF names.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: OPTConfig, tp_params: TPParams):
        super().__init__()
        self.model = OPTModel(config, tp_params)
        self.decoder = self.model.decoder

        # LM head should operate on word_embed_proj_dim, which equals hidden_size
        # for facebook/opt-30b but not necessarily for all OPT variants.
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.word_embed_proj_dim,
            tp_params,
        )

        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight.data = self.model.decoder.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)