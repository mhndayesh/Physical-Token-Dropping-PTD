"""
qwen_ptd.py  –  Physical Token Dropping (PTD) wrapper for Qwen2.5-0.5B
=======================================================================

Qwen2.5-0.5B architecture (do not change these without re-checking the config):
  hidden_size            : 896
  num_attention_heads    : 14   (Q-heads)
  num_key_value_heads    : 2    (GQA K/V-heads)
  head_dim               : 64   ( = hidden_size / num_attention_heads )
  num_hidden_layers      : 24
  intermediate_size      : 4864
  vocab_size             : 151 936
  RoPE emb output shape  : (batch, seq_len, head_dim=64)
  Qwen2Attention.forward needs: position_embeddings=(cos, sin), NOT position_ids.

Strategy: patch model.model.forward so we own the entire forward loop.
  • Group the 24 Qwen2DecoderLayers into "blocks" (e.g. 4 blocks of 6).
  • Before each block: router picks top-k *segments* → gather sparse tokens.
  • Slice the pre-computed cos/sin for those token positions.
  • Pass sparse (hidden_states, cos_sparse, sin_sparse, mask_sparse) into layers.
  • Scatter the result back into the full-length tensor.
  • After all blocks: norm → return.

This avoids all conflicts with Qwen2Model's internal layer-iteration loop.
"""

import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast


# ──────────────────────────────────────────────────────────────────────────────
# MultiQueryRouter  (from ptd_poc/src/transformer_0_5b.py)
# ──────────────────────────────────────────────────────────────────────────────
class MultiQueryRouter(nn.Module):
    """
    Scores segments.
    • During inference (eval): hard top-k physical gather.
    • During training:         continuous segment weights for soft gating,
                               so gradients can flow back to router params.

    sparsity=1.0 → keep all segments (lossless passthrough).
    """
    def __init__(self, d_model: int, num_queries: int = 8, rank: int = 16,
                 sparsity: float = 0.3, jitter: float = 0.01):
        super().__init__()
        self.sparsity = sparsity
        self.jitter   = jitter
        self.k_proj   = nn.Linear(d_model, rank, bias=False)
        self.queries  = nn.Parameter(torch.randn(num_queries, rank))

    def score(self, x: torch.Tensor):
        """Return (continuous_scores, hard_top_k_indices). scores are differentiable."""
        b, n, _ = x.shape
        k       = max(1, int(n * self.sparsity))
        keys    = self.k_proj(x)                                   # (b, n, rank)
        scores  = torch.matmul(self.queries.unsqueeze(0),
                               keys.transpose(1, 2))               # (b, q, n)
        tok_sc  = scores.max(dim=1).values                         # (b, n)
        if self.training and self.jitter > 0:
            tok_sc = tok_sc + torch.randn_like(tok_sc) * self.jitter
        _, ix   = torch.topk(tok_sc.detach(), k, dim=-1)  # no grad through topk
        ix, _   = torch.sort(ix, dim=-1)
        return tok_sc, ix                                          # (b,n_seg) (b,k_seg)

    def forward(self, x: torch.Tensor):
        """Convenience: returns hard indices only (used at eval)."""
        _, ix = self.score(x)
        return ix


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _gather_sparse(x_pad, indices, batch_indices):
    """Physically shrink tensor to selected positions."""
    return x_pad[batch_indices, indices]          # (b, k, d)


def _scatter_back(x_pad, x_sparse, indices, batch_indices, n):
    """Write sparse result back and strip padding."""
    res = x_pad.clone()
    res[batch_indices, indices] = x_sparse
    return res[:, :n, :]


def _slice_posemb(cos, sin, indices, batch_indices, pad_len, n_pad, device):
    """
    cos/sin: (b, n, head_dim=64) or (1, n, head_dim) from rotary_emb.
    Returns cos_sparse, sin_sparse of shape (b, k_tok, 64).
    """
    b = batch_indices.shape[0]

    # Expand batch dim if the rotary_emb returned shape (1, …)
    if cos.shape[0] == 1 and b > 1:
        cos = cos.expand(b, -1, -1)
        sin = sin.expand(b, -1, -1)

    # Pad cos/sin to n_pad if needed
    cur_n = cos.shape[1]
    if cur_n < n_pad:
        cos = F.pad(cos, (0, 0, 0, n_pad - cur_n))
        sin = F.pad(sin, (0, 0, 0, n_pad - cur_n))

    cos_sp = cos[batch_indices, indices]           # (b, k_tok, 64)
    sin_sp = sin[batch_indices, indices]
    return cos_sp, sin_sp


def _causal_mask_sparse(sparse_pos, dtype, device):
    """
    Build a float causal mask of shape (b, 1, k, k) from original positions.
    A token at position i can attend to position j iff i >= j.
    """
    b, k = sparse_pos.shape
    pos_i = sparse_pos.unsqueeze(-1)               # (b, k, 1)
    pos_j = sparse_pos.unsqueeze(-2)               # (b, 1, k)
    mask  = (pos_i >= pos_j).unsqueeze(1)          # (b, 1, k, k) bool
    float_mask = torch.zeros(b, 1, k, k, dtype=dtype, device=device)
    float_mask = float_mask.masked_fill(~mask, torch.finfo(dtype).min)
    return float_mask


# ──────────────────────────────────────────────────────────────────────────────
# Custom PTD forward (replaces Qwen2Model.forward)
# ──────────────────────────────────────────────────────────────────────────────
def _ptd_model_forward(
    self,                          # this is model.model (Qwen2Model instance)
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=False,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
    cache_position=None,
    **kwargs,
):
    # ── Embed ──────────────────────────────────────────────────────────────────
    if inputs_embeds is None:
        hidden = self.embed_tokens(input_ids)
    else:
        hidden = inputs_embeds

    b, n, d = hidden.shape
    device  = hidden.device
    dtype   = hidden.dtype

    if position_ids is None:
        position_ids = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)

    seg_size     = self._ptd_segment_size
    routers      = self._ptd_routers
    layer_groups = self._ptd_layer_groups
    is_training  = self.training

    # We start with the full sequence
    # Each block routes from the full current sequence, then scatters updates back.
    # This keeps per-block retention aligned with router.sparsity instead of compounding
    # sparsity across blocks (e.g., 0.7^4).
    current_hidden = hidden
    current_pos    = position_ids
    original_indices = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)
    current_indices = original_indices
    last_selected_indices = None

    for router, layers in zip(routers, layer_groups):
        # 1. Padding (must handle currently sparse sequence)
        curr_n  = current_hidden.shape[1]
        pad_len = (seg_size - (curr_n % seg_size)) % seg_size
        if pad_len > 0:
            x_pad    = F.pad(current_hidden, (0, 0, 0, pad_len))
            pos_pad  = F.pad(current_pos, (0, pad_len), value=current_pos[0, -1] + 1) # dummy pos
            idx_pad  = F.pad(current_indices, (0, pad_len), value=-1) # -1 for padding
        else:
            x_pad    = current_hidden
            pos_pad  = current_pos
            idx_pad  = current_indices
        
        n_pad    = x_pad.shape[1]
        n_seg    = n_pad // seg_size

        # 2. Routing
        # We pool the current sequence to segments
        x_seg      = x_pad.view(b, n_seg, seg_size, d)
        x_pool     = x_seg.mean(dim=2)
        seg_sc, seg_ix = router.score(x_pool) # (b, n_seg), (b, k_seg)

        # 3. Physical Drop (Gather)
        # tok_ix: indices within x_pad
        tok_ix  = (seg_ix.unsqueeze(-1) * seg_size +
                   torch.arange(seg_size, device=device)).view(b, -1)
        k_tok   = tok_ix.shape[1]
        bat_ix  = torch.arange(b, device=device).view(-1, 1).expand(-1, k_tok)

        x_sp    = _gather_sparse(x_pad, tok_ix, bat_ix)
        sp_pos  = pos_pad[bat_ix, tok_ix]
        sp_idx  = idx_pad[bat_ix, tok_ix]

        # 4. RoPE for sparse positions
        cos_sp, sin_sp = self.rotary_emb(x_sp, sp_pos)
        if cos_sp.shape[0] == 1 and b > 1:
            cos_sp = cos_sp.expand(b, -1, -1)
            sin_sp = sin_sp.expand(b, -1, -1)
        
        hf_mask = _causal_mask_sparse(sp_pos, dtype, device)

        # 5. STE Gating (Training only)
        # ensures forward is 1.0 but backward has gradients
        if is_training:
            sel_sc_soft   = torch.gather(seg_sc, 1, seg_ix)
            sel_gate_soft = torch.sigmoid(sel_sc_soft)
            tok_gate_soft = sel_gate_soft.unsqueeze(-1).repeat(1, 1, seg_size).view(b, k_tok, 1)
            
            # STE: gate = (soft - soft.detach()) + 1.0
            # Forward: 1.0, Backward: sigmoid'(score)
            ste_gate = (tok_gate_soft - tok_gate_soft.detach()) + 1.0
            x_sp = x_sp * ste_gate

        # 6. Process Layers
        for layer in layers:
            lo = layer(
                hidden_states       = x_sp,
                attention_mask      = hf_mask,
                position_ids        = sp_pos,
                past_key_values     = None,
                output_attentions   = False,
                use_cache           = False,
                position_embeddings = (cos_sp, sin_sp),
            )
            x_sp = lo[0] if isinstance(lo, (tuple, list)) else lo
        
        # 7. Scatter block result back to full-length sequence, then continue.
        x_full = x_pad.clone()
        x_full[bat_ix, tok_ix] = x_sp
        current_hidden = x_full[:, :curr_n, :]
        last_selected_indices = sp_idx

    hidden = self.norm(current_hidden)
    
    # Selection Metadata for masked loss (use last block's routed token set)
    selection_mask = torch.zeros(b, n, device=device, dtype=torch.bool)
    if last_selected_indices is not None:
        valid_mask = (last_selected_indices >= 0) & (last_selected_indices < n)
        batch_indices = torch.arange(b, device=device).unsqueeze(1).expand_as(last_selected_indices)
        selection_mask[batch_indices[valid_mask], last_selected_indices[valid_mask]] = True

    return BaseModelOutputWithPast(
        last_hidden_state = hidden,
        past_key_values    = None,
        # Return (mask, indices) - always available now
        hidden_states      = (selection_mask, last_selected_indices)
    )



# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def apply_ptd_to_qwen2(
    model:        Qwen2ForCausalLM,
    block_size:   int   = 6,     # Qwen layers per PTD block (24 / 6 = 4 blocks)
    sparsity:     float = 0.3,   # segment-keep fraction (1.0 = no dropping)
    segment_size: int   = 16,    # tokens per routing segment
) -> Qwen2ForCausalLM:
    """
    Patches model.model.forward in-place to use Physical Token Dropping.

    The original pretrained weights are never cloned or replaced – we only
    add lightweight router modules and swap the forward method.

    Returns the same model object.
    """
    base       = model.model
    d_model    = model.config.hidden_size          # 896
    num_layers = model.config.num_hidden_layers    # 24

    if num_layers % block_size != 0:
        raise ValueError(
            f"num_hidden_layers ({num_layers}) must be divisible by block_size ({block_size})"
        )
    n_blocks = num_layers // block_size

    # Build routers and store layer groups (lists of existing Qwen2DecoderLayers)
    routers      = nn.ModuleList()
    layer_groups = []
    for i in range(n_blocks):
        start = i * block_size
        routers.append(
            MultiQueryRouter(d_model, sparsity=sparsity)
        )
        layer_groups.append(
            [base.layers[j] for j in range(start, start + block_size)]
        )

    # Attach PTD config and routers to base model so they're tracked by .parameters()
    base._ptd_block_size   = block_size
    base._ptd_segment_size = segment_size
    base._ptd_routers      = routers
    base._ptd_layer_groups = layer_groups
    base.add_module("ptd_routers", routers)

    # Patch forward
    base.forward = types.MethodType(_ptd_model_forward, base)

    return model
