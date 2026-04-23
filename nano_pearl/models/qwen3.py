import torch
from torch import nn
from transformers import Qwen3Config

from nano_pearl.layers.activation import SiluAndMul
from nano_pearl.layers.attention import Attention
from nano_pearl.layers.layernorm import RMSNorm
from nano_pearl.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nano_pearl.layers.rotary_embedding import get_rope
from nano_pearl.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nano_pearl.pearl_config import TPParams


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        tp_params: TPParams,
        max_position: int = 4096 * 32,       # 最大位置编码长度
        head_dim: int | None = None,  
        rms_norm_eps: float = 1e-06,         # RMSNorm 的 epsilon，防止除零
        qkv_bias: bool = False,              # QKV 线性层是否带偏置
        rope_theta: float = 10000,           # RoPE 的底数（旋转角度常数）
        rope_scaling: tuple | None = None,   # RoPE 插值/缩放参数
    ) -> None:
        super().__init__()
        tp_size = tp_params.tp_size

        # --- 张量并行计算：将 Head 均匀分配到不同的 GPU ---

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        # --- 维度计算 ---

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5    # 缩放因子 1/sqrt(d_k)

        # --- 子模块初始化 ---
        # 1. QKV 合并投影层（列并行）：一次性算出本卡所需的 Q、K、V 片段
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            tp_params,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # 2. Output 投影层（行并行）：将多卡算出的注意力结果汇聚（All-Reduce）并还原到 hidden_size
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            tp_params,
            bias=False,
        )

        # 3. 位置编码层：计算旋转矩阵（RoPE）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # 4. 核心计算层：执行真正的 Scaled Dot-Product Attention（通常封装了 FlashAttention）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            tp_params,
        )

        # 5. QK-Norm：Qwen 系列的灵魂特色，在计算相似度前对 Q 和 K 做归一化
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)


    def forward(
        self,
        positions: torch.Tensor,       # 位置索引，用于 RoPE 计算
        hidden_states: torch.Tensor,   # 输入张量 [seq_len, hidden_size]
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        # 2. 切分 Q, K, V：由于 QKV 是拼在一起算的，需要手动拆开
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 3. 执行 QK-Norm：这是为了增强数值稳定性，每个 Head 内部独立进行 RMSNorm
        # 这一步是 Qwen2/3 区别于 Llama 的重要特征
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 4. 应用位置编码 (RoPE)：在 Norm 之后注入位置信息
        q, k = self.rotary_emb(positions, q, k)
        # 5. 计算注意力：调用底层 Kernel（如 FlashAttention）计算上下文加权结果
        # 输出 o 的形状通常是 [seq_len, num_heads, head_dim]
        o = self.attn(q, k, v)
        # 6. 输出投影与行并行通信：打平所有 Head 的输出，通过 o_proj 进行线性变换
        # RowParallelLinear 内部会处理多 GPU 间的 All-Reduce 通信，最终输出 [seq_len, hidden_size]
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        tp_params: TPParams,
    ) -> None:
        super().__init__()
        #gate和up 合并 // Gate Proj（用于激活的分支）和 Up Proj（不经过激活的分支）合并成了一个大的线性层
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            tp_params=tp_params,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_params=tp_params,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

#transformer 层
class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        tp_params: TPParams,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            tp_params=tp_params,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            tp_params=tp_params,
        )
        #两个归一化层 (RMSNorm)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    #注意到这里的 RMSNorm 接受了两个参数。
    #在高性能推理引擎中，Add (残差相加) 和 Norm (归一化) 通常会被融合 (Fused) 成一个 GPU Kernel。
    #这样可以减少一次内存读写（Memory Access），这对受限于带宽（Memory-Bound）的解码过程至关重要。
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: 
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


#加入embedding和lmhead
class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        tp_params: TPParams,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, tp_params)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, tp_params) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    #外部模型名字是分开的，内部执行模块是合并的；加载权重时要靠这张表对号入座。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        tp_params: TPParams,
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config, tp_params)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, tp_params)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

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
