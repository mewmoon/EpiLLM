import math
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class EpiConfig(PretrainedConfig):
    model_type = "epillm"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = False,
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        self.flash_attn = flash_attn
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)  # 精度


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class EpiAttention(torch.nn.Module):
    def __init__(self, config: EpiConfig):
        super().__init__()
        self.num_key_value_heads = (
            config.num_key_value_heads or config.num_attention_heads
        )

        self.n_local_heads = config.num_attention_heads  # q的头数
        self.n_local_kv_heads = self.num_key_value_heads  # k,v的头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = torch.nn.Linear(
            config.hidden_size, self.n_local_heads * self.head_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(
            self.n_local_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.attn_dropout = torch.nn.Dropout(config.dropout)
        self.resid_dropout = torch.nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash_attn = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attn
        )

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        # print("Attention:", attention_mask == None)
        bs, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)

        # 位置编码
        # cos, sin = position_embeddings
        # xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),  # bs,qheads,seq_len,head_dim
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # bs,qheads,seq_len,head_dim
            repeat_kv(xv, self.n_rep).transpose(1, 2),  # bs,qheads,seq_len,head_dim
        )

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
            # 因果掩码
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )  # -seq_len适配kv cache

            # 注意力pad掩码
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores += extended_attention_mask

            # bs,qheads,seq_len,seq_len
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv  # -> bs,qheads,seq_len,dim

        output = output.transpose(1, 2).reshape(bs, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


class EpiMLP(torch.nn.Module):
    def __init__(self, config: EpiConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(8 / 3 * config.hidden_size)
            config.intermediate_size = (intermediate_size // 64) * 64  # 向下取整
        self.gate_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.droupout = torch.nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.droupout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


class EpiMoEMLP(torch.nn.Module):
    pass


class EpiBlock(torch.nn.Module):
    def __init__(self, config: EpiConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = EpiAttention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = EpiMoEMLP(config) if config.use_moe else EpiMLP(config)

    def forward(
        self,
        hidden_state,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        resiual = hidden_state
        hidden_state, present_key_value = self.attn(
            self.input_layernorm(hidden_state),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_state += resiual

        hidden_state = hidden_state + self.mlp(
            self.post_attention_layernorm(hidden_state)
        )
        return hidden_state, present_key_value


class EpiModel(torch.nn.Module):
    def __init__(self, config: EpiConfig):
        super().__init__()
        # 1 嵌入层
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout)
        # 2 Block
        self.blocks = torch.nn.ModuleList(
            [EpiBlock(config) for l in range(config.num_hidden_layers)]
        )
        # 3 归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, input_ids, past_key_values=None, use_cache=False, attention_mask=None
    ):
        hidden_state = self.dropout(self.embeddings(input_ids))
        past_key_values = past_key_values or [None] * len(self.blocks)
        position_embeddings = None  # 位置编码,Attention使用
        presents = []  # 每层的present,用于推理加速???
        for block, past_key_value in zip(self.blocks, past_key_values):
            hidden_state, present = block(
                hidden_state,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_state = self.norm(hidden_state)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.blocks if isinstance(l.mlp, EpiMoEMLP)],
            hidden_state.new_zeros(1).squeeze(),
        )
        return hidden_state, presents, aux_loss


class EpiForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = EpiConfig

    def __init__(self, config: EpiConfig = None):
        self.config = config or EpiConfig()
        super().__init__(self.config)
        self.model = EpiModel(self.config)
        self.lm_head = torch.nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

    def forward(
        self,
        input_ids,
        labels=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
    ):
        hidden_state, presents, aux_loss = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # bs*seq_len, vocab_size
                shift_labels.view(-1),  # bs*seq_len
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_state,
        )
        output.aux_loss = aux_loss
        return output
