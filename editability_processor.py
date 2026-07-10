"""Editability capture processor for FLUX.1-Kontext.

Measures the key quantities of our editability inequality per gen spatial position:

    E(y, x) = γ · [S_text(y,x) / S_ref(y,x)] · [‖μ_text‖ / ‖v_ref(y,x)‖]

Where:
    γ            = CFG guidance scale (constant, ~3.5)
    S_text(y,x)  = sum of attention weights from gen(y,x) to text tokens
    S_ref(y,x)   = sum of attention weights from gen(y,x) to ref tokens
    ‖μ_text‖     = mean text value norm across text tokens (scalar)
    ‖v_ref(y,x)‖ = value norm at same-position ref token

Predicts:
    E > 1  ⟹  editable (text wins over ref anchor)
    E < 1  ⟹  identity-preserving (ref wins)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class EditabilityCaptureProcessor:
    """Captures per-position editability signals at each attention layer."""

    def __init__(self, grid_size: int = 64, layer_name: str = "?") -> None:
        self.grid_size = grid_size
        self.N = grid_size ** 2
        self.layer_name = layer_name
        # Per-forward captures
        self.S_text_gen: list[torch.Tensor] = []   # [N]
        self.S_ref_gen: list[torch.Tensor] = []    # [N]
        self.v_ref_norm: list[torch.Tensor] = []   # [N]
        self.v_text_mean_norm: list[torch.Tensor] = []  # scalar

    def _capture(self, jq_p, jk_p, jv_p, T_full: int, N: int) -> None:
        """Given joint Q/K/V per layer, capture editability quantities."""
        B, H, L, D_h = jq_p.shape
        ci = B - 1  # conditional batch
        scale = D_h ** -0.5

        # ── Attention weights from gen queries (rows) to full sequence (cols) ──
        q_gen = jq_p[ci : ci + 1, :, T_full : T_full + N, :].float()  # [1, H, N, D_h]
        k_all = jk_p[ci : ci + 1].float()  # [1, H, L, D_h]

        scores = (q_gen @ k_all.transpose(-2, -1)) * scale  # [1, H, N, L]
        weights = torch.softmax(scores, dim=-1)  # [1, H, N, L]

        # Sum by column category (text | gen | ref)
        S_text = weights[..., :T_full].sum(dim=-1)  # [1, H, N]
        S_ref = weights[..., T_full + N : T_full + 2 * N].sum(dim=-1)  # [1, H, N]

        # Head-average
        S_text_mean = S_text.mean(dim=1)[0]  # [N]
        S_ref_mean = S_ref.mean(dim=1)[0]  # [N]

        # ── Value norms (per position) ──
        v_all = jv_p[ci : ci + 1].float()  # [1, H, L, D_h]
        v_norm = v_all.norm(dim=-1)  # [1, H, L]
        # Head-average
        v_norm_mean = v_norm.mean(dim=1)[0]  # [L]

        v_text_norm = v_norm_mean[:T_full]  # [T_full]
        # Consider only non-pad text tokens (first ~15 typically)
        v_text_mean_norm = v_text_norm[:20].mean()  # scalar (rough)

        v_ref_norm = v_norm_mean[T_full + N : T_full + 2 * N]  # [N]

        self.S_text_gen.append(S_text_mean.cpu())
        self.S_ref_gen.append(S_ref_mean.cpu())
        self.v_ref_norm.append(v_ref_norm.cpu())
        self.v_text_mean_norm.append(v_text_mean_norm.cpu())

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        from diffusers.models.transformers.transformer_flux import (
            _get_qkv_projections,
            apply_rotary_emb,
        )

        # ── Single-stream block ─────────────────────────────
        if encoder_hidden_states is None:
            query, key, value, _, _, _ = _get_qkv_projections(attn, hidden_states, None)
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            jq_p = query.permute(0, 2, 1, 3)
            jk_p = key.permute(0, 2, 1, 3)
            jv_p = value.permute(0, 2, 1, 3)

            L = hidden_states.shape[1]
            N = self.N
            T_full = L - 2 * N
            if T_full > 0:
                self._capture(jq_p, jk_p, jv_p, T_full, N)

            full_out = F.scaled_dot_product_attention(jq_p, jk_p, jv_p)
            return full_out.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)

        # ── Double-stream block ─────────────────────────────
        query, key, value, eq, ek, ev = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        for t in [query, key, value, eq, ek, ev]:
            t.__class__ = torch.Tensor

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        eq = eq.unflatten(-1, (attn.heads, -1))
        ek = ek.unflatten(-1, (attn.heads, -1))
        ev = ev.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        eq = attn.norm_added_q(eq)
        ek = attn.norm_added_k(ek)

        jq = torch.cat([eq, query], dim=1)
        jk = torch.cat([ek, key], dim=1)
        jv = torch.cat([ev, value], dim=1)
        if image_rotary_emb is not None:
            jq = apply_rotary_emb(jq, image_rotary_emb, sequence_dim=1)
            jk = apply_rotary_emb(jk, image_rotary_emb, sequence_dim=1)

        jq_p = jq.permute(0, 2, 1, 3)
        jk_p = jk.permute(0, 2, 1, 3)
        jv_p = jv.permute(0, 2, 1, 3)

        T_full = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1] // 2
        self._capture(jq_p, jk_p, jv_p, T_full, N)

        full_out = F.scaled_dot_product_attention(jq_p, jk_p, jv_p)
        full_out = full_out.permute(0, 2, 1, 3).flatten(2, 3).to(jq.dtype)
        T_p = encoder_hidden_states.shape[1]
        enc_out = full_out[:, :T_p]
        img_out = full_out[:, T_p:]
        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        enc_out = attn.to_add_out(enc_out)
        return img_out, enc_out


def install_editability_processors(pipe, grid_size: int = 64):
    """Install one EditabilityCaptureProcessor per attention block. Returns
    (processors dict, original processors dict)."""
    processors = {}
    original = {}
    for name, module in pipe.transformer.named_modules():
        if not (name.endswith(".attn") and hasattr(module, "set_processor")):
            continue
        if not ("transformer_blocks" in name or "single_transformer_blocks" in name):
            continue
        proc = EditabilityCaptureProcessor(grid_size=grid_size, layer_name=name)
        original[name] = module.processor
        module.set_processor(proc)
        processors[name] = proc
    print(f"[Editability] installed on {len(processors)} layers")
    return processors, original


def restore_processors(pipe, original_procs):
    for name, module in pipe.transformer.named_modules():
        if name in original_procs:
            module.set_processor(original_procs[name])
    print(f"[Editability] restored {len(original_procs)} layers")
