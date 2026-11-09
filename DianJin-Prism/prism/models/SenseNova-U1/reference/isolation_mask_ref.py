"""Standalone extract of SenseNova-U1's official und/gen isolation attention mask.

Copied VERBATIM (pure-torch parts only) from the official training model
``modeling_sensenovavl_chat_mot.py`` — functions ``dense_from_flex_mask`` (:126-138)
and the mask-mod composition inside ``create_flex_mask_padding_image_gen`` (:140-212).
InternEvo imports are stripped; the mask logic is unchanged.

Purpose: an OFFICIAL ORACLE for L1 parity — the adapter's own dense isolation-mask
builder must reproduce ``official_dense_mask(...)`` element-for-element. This module is
imported ONLY by tests, never by the runtime adapter. Do not hand-edit.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import and_masks, or_masks


@torch.no_grad()
def dense_from_flex_mask(mask_fn, slen: int, device="cpu") -> torch.Tensor:
    """Materialize a flex-attention ``mask_mod`` into a dense [slen, slen] bool mask.

    Verbatim from modeling_sensenovavl_chat_mot.py:126-138 (device defaulted to cpu).
    """
    q = torch.arange(slen, device=device)
    k = torch.arange(slen, device=device)
    Q, K = torch.meshgrid(q, k, indexing="ij")
    dummy_b = torch.zeros(1, device=device)
    dummy_h = torch.zeros(1, device=device)
    M = mask_fn(dummy_b, dummy_h, Q, K)
    return M.bool()


def official_mask_mod(
    document_ids: torch.Tensor,
    modality_indicators: torch.Tensor,
    image_gen_indicators: torch.Tensor,
    token_pos: torch.Tensor,
    dup_boundary: torch.Tensor,
):
    """Compose the official mask_mod (create_flex_mask_padding_image_gen:160-195).

    Rules: (causal-by-token_pos AND same-doc) OR same-image; AND gen/und isolation
    (``gen_kv_gate_mask``); AND forbid attending to duplicate-boundary tokens.
    """

    def causal_mask(b, h, q_idx, kv_idx):
        return token_pos[q_idx] >= token_pos[kv_idx]

    def samedoc_mask(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx]

    def sameimg_mask(b, h, q_idx, kv_idx):
        is_image = modality_indicators[q_idx] > 0
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        return is_image & (modality_indicators[q_idx] == modality_indicators[kv_idx]) & same_doc

    samedoc_causal_mask = and_masks(causal_mask, samedoc_mask)
    mask_mod = or_masks(samedoc_causal_mask, sameimg_mask)

    def gen_kv_gate_mask(b, h, q_idx, kv_idx):
        kv_is_gen = image_gen_indicators[kv_idx]
        q_is_gen = image_gen_indicators[q_idx]
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        same_img = (modality_indicators[q_idx] == modality_indicators[kv_idx]) & same_doc
        gate1 = (~kv_is_gen) | (q_is_gen & same_img)
        gate2 = (~q_is_gen) | kv_is_gen | (token_pos[kv_idx] != token_pos[q_idx])
        return gate1 & gate2

    mask_mod = and_masks(mask_mod, gen_kv_gate_mask)

    def dup_boundary_gate_mask(b, h, q_idx, kv_idx):
        return ~dup_boundary[kv_idx]

    mask_mod = and_masks(mask_mod, dup_boundary_gate_mask)
    return mask_mod


def official_dense_mask(
    document_ids: torch.Tensor,
    modality_indicators: torch.Tensor,
    image_gen_indicators: torch.Tensor,
    token_pos: torch.Tensor,
    dup_boundary: torch.Tensor,
) -> torch.Tensor:
    """Official LOGICAL attend-mask [L, L] bool (True = q may attend to kv).

    Uses the unpadded sequence (padding in the original only serves flex block
    alignment; it does not change attendability among real tokens).
    """
    mm = official_mask_mod(document_ids, modality_indicators, image_gen_indicators, token_pos, dup_boundary)
    return dense_from_flex_mask(mm, int(document_ids.numel()), device=document_ids.device)


def build_dup_boundary(modality_indicators: torch.Tensor, gen_tok: torch.Tensor, dup_tok: torch.Tensor) -> torch.Tensor:
    """Verbatim from modeling_sensenovavl_chat_mot.py:215-227."""
    dup_boundary = torch.zeros_like(gen_tok)
    boundary = modality_indicators == -1
    dup_boundary[:-1] |= boundary[:-1] & dup_tok[1:]
    dup_boundary[1:-2] |= boundary[1:-2] & gen_tok[:-3] & boundary[2:-1] & dup_tok[3:]
    return dup_boundary
