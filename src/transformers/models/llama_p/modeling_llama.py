# DMI Hooked Llama variant. Mirrors the gpt2_p / qwen3_p layout.
# Subclasses the canonical Llama classes from `..llama.modeling_llama`
# and inserts HookPoint instances at the observation sites.

from typing import Callable, Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from dmi.hooks.point import HookPoint, HookedRootModule


# Local copy of eager_attention_forward with hook_attn_scores /
# hook_pattern call sites. Mirrors qwen3_p/modeling_qwen3.py:131. The
# canonical eager_attention_forward stays clean of DMI references.
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if hasattr(module, "hook_attn_scores"):
        attn_weights = module.hook_attn_scores(attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if hasattr(module, "hook_pattern"):
        attn_weights = module.hook_pattern(attn_weights)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class HookedLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()
        # hook_result removed: attn_out == o_proj output in all architectures

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.hook_q(query_states)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.hook_k(key_states)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.hook_v(value_states)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = self.hook_z(attn_output)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HookedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = HookedLlamaAttention(config=config, layer_idx=layer_idx)
        from ..llama.modeling_llama import LlamaMLP
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_ln1 = HookPoint()
        self.hook_ln2 = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()
        # hook_resid_post removed: equivalent to next layer's hook_resid_pre

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.hook_resid_pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.hook_ln1(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.hook_attn_out(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.hook_resid_mid(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.hook_ln2(hidden_states)
        hidden_states = self.hook_mlp_in(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hook_mlp_out(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HookedLlamaModel(LlamaModel, HookedRootModule):
    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HookedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.hook_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.hook_final_ln = HookPoint()

        self.post_init()
        self.setup()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.hook_embed(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        self.hook_resid_final(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.hook_final_ln(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class HookedLlamaForCausalLM(LlamaForCausalLM, HookedRootModule):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = HookedLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.token_ids = HookPoint()
        self.final_logits = HookPoint()

        self.post_init()
        self.setup()
        self._normalize_hook_names()

    def _normalize_hook_names(self) -> None:
        normalized_hooks: dict[str, HookPoint] = {}
        for name, hook_point in list(self.hook_dict.items()):
            if name.startswith("model."):
                name = name[len("model.") :]
            normalized_hooks[name] = hook_point
        self.hook_dict = normalized_hooks

        normalized_mods: dict[str, nn.Module] = {}
        for name, module in list(self.mod_dict.items()):
            if name.startswith("model."):
                name = name[len("model.") :]
            normalized_mods[name] = module
        self.mod_dict = normalized_mods

    def get_hook_specs(self) -> list:
        import torch
        from dmi.transport.ring import (
            HookSpec,
            HOOK_TYPE_EMBED, HOOK_TYPE_FINAL_LN, HOOK_TYPE_RESID_FINAL,
            HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_Q, HOOK_TYPE_K,
            HOOK_TYPE_V, HOOK_TYPE_ATTN_SCORES, HOOK_TYPE_PATTERN,
            HOOK_TYPE_Z, HOOK_TYPE_ATTN_OUT,
            HOOK_TYPE_RESID_MID, HOOK_TYPE_LN2, HOOK_TYPE_MLP_IN,
            HOOK_TYPE_MLP_OUT,
            HOOK_TYPE_TOKEN_IDS, HOOK_TYPE_FINAL_LOGITS,
        )

        m = self.model
        is_eager = (self.config._attn_implementation == "eager")
        specs = []
        specs.append(HookSpec(HOOK_TYPE_TOKEN_IDS, self.token_ids, dtype=torch.long))
        specs.append(HookSpec(HOOK_TYPE_EMBED, m.hook_embed))
        for i, layer in enumerate(m.layers):
            specs.append(HookSpec(HOOK_TYPE_RESID_PRE, layer.hook_resid_pre, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_LN1, layer.hook_ln1, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_Q, layer.self_attn.hook_q, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_K, layer.self_attn.hook_k, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_V, layer.self_attn.hook_v, layer_no=i))
            if is_eager:
                specs.append(HookSpec(HOOK_TYPE_ATTN_SCORES, layer.self_attn.hook_attn_scores, layer_no=i))
                specs.append(HookSpec(HOOK_TYPE_PATTERN, layer.self_attn.hook_pattern, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_Z, layer.self_attn.hook_z, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_ATTN_OUT, layer.hook_attn_out, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_RESID_MID, layer.hook_resid_mid, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_LN2, layer.hook_ln2, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_MLP_IN, layer.hook_mlp_in, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_MLP_OUT, layer.hook_mlp_out, layer_no=i))
        specs.append(HookSpec(HOOK_TYPE_RESID_FINAL, m.hook_resid_final))
        specs.append(HookSpec(HOOK_TYPE_FINAL_LN, m.hook_final_ln))
        specs.append(HookSpec(HOOK_TYPE_FINAL_LOGITS, self.final_logits))
        return specs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if input_ids is not None:
            input_ids = self.token_ids(input_ids)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = self.final_logits(logits)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HookedLlamaAttention",
    "HookedLlamaDecoderLayer",
    "HookedLlamaForCausalLM",
    "HookedLlamaModel",
]
