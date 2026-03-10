from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class PTDConfig:
    block_size: int = 6
    segment_size: int = 16
    keep_rate: float = 0.3
    router_rank: int = 16
    router_queries: int = 8
    router_jitter: float = 0.01
    drop_tokens: bool = True
    ste_gating: bool = True


class MultiQueryRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        keep_rate: float,
        num_queries: int,
        rank: int,
        jitter: float,
    ) -> None:
        super().__init__()
        self.keep_rate = keep_rate
        self.jitter = jitter
        self.k_proj = nn.Linear(d_model, rank, bias=False)
        self.queries = nn.Parameter(torch.randn(num_queries, rank))

    def score(
        self,
        segment_embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_seg, _ = segment_embeddings.shape
        k_seg = max(1, int(n_seg * self.keep_rate))
        keys = self.k_proj(segment_embeddings)
        scores = torch.matmul(self.queries.unsqueeze(0), keys.transpose(1, 2)).max(dim=1).values
        if self.training and self.jitter > 0:
            scores = scores + torch.randn_like(scores) * self.jitter
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        _, topk_idx = torch.topk(scores.detach(), k_seg, dim=-1)
        topk_idx, _ = torch.sort(topk_idx, dim=-1)
        return scores, topk_idx


def _build_additive_causal_mask(
    position_ids: torch.Tensor,
    token_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    pos_i = position_ids.unsqueeze(-1)
    pos_j = position_ids.unsqueeze(-2)
    causal = pos_i >= pos_j
    q_valid = token_mask.unsqueeze(-1)
    k_valid = token_mask.unsqueeze(-2)
    allowed = causal & q_valid & k_valid
    additive = torch.zeros(
        position_ids.size(0),
        1,
        position_ids.size(1),
        position_ids.size(1),
        dtype=dtype,
        device=position_ids.device,
    )
    additive = additive.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return additive


def _segment_pool(
    x_pad: torch.Tensor,
    m_pad: torch.Tensor,
    n_seg: int,
    seg_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, _, hidden = x_pad.shape
    x_seg = x_pad.view(bsz, n_seg, seg_size, hidden)
    m_seg = m_pad.view(bsz, n_seg, seg_size)
    denom = m_seg.sum(dim=2, keepdim=True).clamp_min(1).to(x_pad.dtype)
    pooled = (x_seg * m_seg.unsqueeze(-1).to(x_pad.dtype)).sum(dim=2) / denom
    seg_valid = m_seg.any(dim=2)
    return pooled, seg_valid


class PTDQwen2ForCausalLM(nn.Module):
    def __init__(self, base_model: Qwen2ForCausalLM, ptd: PTDConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.ptd = ptd

        n_layers = base_model.config.num_hidden_layers
        if n_layers % ptd.block_size != 0:
            raise ValueError(
                f"num_hidden_layers ({n_layers}) must be divisible by block_size ({ptd.block_size})"
            )
        n_blocks = n_layers // ptd.block_size
        d_model = base_model.config.hidden_size

        self.routers = nn.ModuleList(
            [
                MultiQueryRouter(
                    d_model=d_model,
                    keep_rate=ptd.keep_rate,
                    num_queries=ptd.router_queries,
                    rank=ptd.router_rank,
                    jitter=ptd.router_jitter,
                )
                for _ in range(n_blocks)
            ]
        )
        ref_param = next(self.base_model.parameters())
        self.routers.to(device=ref_param.device, dtype=ref_param.dtype)
        self.layer_groups = [
            [base_model.model.layers[j] for j in range(i * ptd.block_size, (i + 1) * ptd.block_size)]
            for i in range(n_blocks)
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        ptd_config: Optional[PTDConfig] = None,
        **kwargs: Any,
    ) -> "PTDQwen2ForCausalLM":
        if ptd_config is None:
            ptd_config = PTDConfig()
        base = Qwen2ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(base_model=base, ptd=ptd_config)

    @property
    def config(self):
        return self.base_model.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        return self

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("routers."):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def router_parameters(self) -> Iterable[nn.Parameter]:
        return self.routers.parameters()

    def set_keep_rate(self, keep_rate: float) -> None:
        self.ptd.keep_rate = keep_rate
        for router in self.routers:
            router.keep_rate = keep_rate

    def set_drop_tokens(self, drop_tokens: bool) -> None:
        self.ptd.drop_tokens = drop_tokens

    def ptd_config_dict(self) -> Dict[str, Any]:
        return asdict(self.ptd)

    def _forward_hidden_with_aux(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        model = self.base_model.model
        if inputs_embeds is None:
            hidden = model.embed_tokens(input_ids)
        else:
            hidden = inputs_embeds

        bsz, seq_len, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        if attention_mask is None:
            token_mask = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
        else:
            token_mask = attention_mask.to(torch.bool)

        seg_size = self.ptd.segment_size
        selection_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        gate_means = []
        segment_selections = []
        router_entropies = []

        for router, layers in zip(self.routers, self.layer_groups):
            pad_len = (seg_size - (seq_len % seg_size)) % seg_size
            if pad_len > 0:
                x_pad = F.pad(hidden, (0, 0, 0, pad_len))
                p_pad = F.pad(position_ids, (0, pad_len), value=position_ids[:, -1:].max().item() + 1)
                m_pad = F.pad(token_mask, (0, pad_len), value=False)
                i_pad = F.pad(
                    torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1),
                    (0, pad_len),
                    value=-1,
                )
            else:
                x_pad = hidden
                p_pad = position_ids
                m_pad = token_mask
                i_pad = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

            n_pad = x_pad.size(1)
            n_seg = n_pad // seg_size
            pooled, seg_valid = _segment_pool(x_pad, m_pad, n_seg, seg_size)
            seg_scores, seg_ix = router.score(pooled, valid_mask=seg_valid)
            seg_selected = torch.zeros(bsz, n_seg, dtype=torch.bool, device=device)
            seg_selected.scatter_(1, seg_ix, torch.ones_like(seg_ix, dtype=torch.bool))
            segment_selections.append(seg_selected)

            scores_f = seg_scores.float()
            probs = torch.softmax(scores_f, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            router_entropies.append(entropy)

            tok_ix = (
                seg_ix.unsqueeze(-1) * seg_size + torch.arange(seg_size, device=device)
            ).view(bsz, -1)
            k_tok = tok_ix.size(1)
            bat_ix = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, k_tok)

            if self.ptd.drop_tokens:
                x_sp = x_pad[bat_ix, tok_ix]
                p_sp = p_pad[bat_ix, tok_ix]
                m_sp = m_pad[bat_ix, tok_ix]
                idx_sp = i_pad[bat_ix, tok_ix]

                if self.training and self.ptd.ste_gating:
                    sel_soft = torch.gather(seg_scores, 1, seg_ix)
                    gate_soft = torch.sigmoid(sel_soft)
                    tok_gate = gate_soft.unsqueeze(-1).repeat(1, 1, seg_size).view(bsz, k_tok, 1)
                    ste_gate = (tok_gate - tok_gate.detach()) + 1.0
                    x_sp = x_sp * ste_gate

                attn_sp = _build_additive_causal_mask(p_sp, m_sp, dtype=dtype)
                cos_sp, sin_sp = model.rotary_emb(x_sp, p_sp)
                for layer in layers:
                    x_sp = layer(
                        hidden_states=x_sp,
                        attention_mask=attn_sp,
                        position_ids=p_sp,
                        position_embeddings=(cos_sp, sin_sp),
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False,
                    )
                    if isinstance(x_sp, (tuple, list)):
                        x_sp = x_sp[0]
                x_pad = x_pad.clone()
                x_pad[bat_ix, tok_ix] = x_sp
                hidden = x_pad[:, :seq_len, :]

                valid = (idx_sp >= 0) & (idx_sp < seq_len) & m_sp
                selection_mask[bat_ix[valid], idx_sp[valid]] = True
            else:
                if self.training:
                    soft_all = torch.sigmoid(seg_scores).unsqueeze(-1).repeat(1, 1, seg_size).view(bsz, n_pad, 1)
                    gate_mean = (soft_all.squeeze(-1) * m_pad.to(soft_all.dtype)).sum() / m_pad.to(
                        soft_all.dtype
                    ).sum().clamp_min(1.0)
                    gate_means.append(gate_mean)
                    if self.ptd.ste_gating:
                        gate_full = (soft_all - soft_all.detach()) + 1.0
                    else:
                        gate_full = soft_all
                    x_pad = x_pad * gate_full

                attn_full = _build_additive_causal_mask(p_pad, m_pad, dtype=dtype)
                cos_full, sin_full = model.rotary_emb(x_pad, p_pad)
                for layer in layers:
                    x_pad = layer(
                        hidden_states=x_pad,
                        attention_mask=attn_full,
                        position_ids=p_pad,
                        position_embeddings=(cos_full, sin_full),
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False,
                    )
                    if isinstance(x_pad, (tuple, list)):
                        x_pad = x_pad[0]
                hidden = x_pad[:, :seq_len, :]

                idx_sp = i_pad[bat_ix, tok_ix]
                m_sp = m_pad[bat_ix, tok_ix]
                valid = (idx_sp >= 0) & (idx_sp < seq_len) & m_sp
                selection_mask[bat_ix[valid], idx_sp[valid]] = True

        hidden = model.norm(hidden)
        aux = {
            "selection_mask": selection_mask,
            "token_mask": token_mask,
            "gate_means": torch.stack(gate_means) if gate_means else torch.empty(0, device=device, dtype=dtype),
            "segment_selection": (
                torch.stack(segment_selections) if segment_selections else torch.empty(0, device=device, dtype=torch.bool)
            ),
            "router_entropy": (
                torch.stack(router_entropies)
                if router_entropies
                else torch.empty(0, device=device, dtype=torch.float32)
            ),
        }
        return hidden, aux

    def forward_with_aux(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> Tuple[CausalLMOutputWithPast, Dict[str, torch.Tensor]]:
        hidden, aux = self._forward_hidden_with_aux(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.base_model.lm_head(hidden[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.base_model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.base_model.config.vocab_size,
            )
        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        return out, aux

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        # Cache path falls back to dense HF forward to preserve generation semantics.
        if use_cache or past_key_values is not None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        out, _ = self.forward_with_aux(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            logits_to_keep=logits_to_keep,
        )
        return out
