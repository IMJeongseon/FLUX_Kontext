"""Attention capture processor for FLUX.1-Kontext.

Captures the softmax-normalized attention weights FROM each image token (gen or ref)
TO specific text token indices during a forward pass. Used to derive self-supervised
localization masks without external detectors (GroundingDINO/SAM).

Joint sequence layout (FluxKontext):
  Double-stream block:
    encoder_hidden_states = text (T_full tokens)
    hidden_states         = [gen(N) | ref(N)]
    joint = [text | gen | ref]  after concat inside the block
  Single-stream block:
    hidden_states already = [text | gen | ref]
    encoder_hidden_states = None

Both cases produce the same joint attention layout: [text(T) | gen(N) | ref(N)].
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


class AttentionCaptureProcessor:
    """Wraps FLUX joint attention; captures img→text softmax weights for target tokens.

    Storage layout (self.captured):
        {
          layer_name: {
            'gen': tensor [n_captures, N, K],   # per-step captures stacked
            'ref': tensor [n_captures, N, K],
          }
        }
    where K = len(text_indices), and n_captures grows with each forward call.

    Head-averaging is done on the fly to keep memory manageable.
    Only conditional batch (last batch dim) is captured (CFG assumed).
    """

    def __init__(
        self,
        text_indices: list[int],
        grid_size: int = 64,
        layer_name: str = "unknown",
    ) -> None:
        self.text_indices_list = list(text_indices)
        self.text_indices = torch.tensor(text_indices, dtype=torch.long)
        self.grid_size = grid_size
        self.N = grid_size * grid_size
        self.layer_name = layer_name
        self.captures_gen: list[torch.Tensor] = []
        self.captures_ref: list[torch.Tensor] = []
        self._T_full: int | None = None
        self._N_seq: int | None = None
        # Cache text_indices per device
        self._idx_dev: dict[str, torch.Tensor] = {}

    def _get_idx(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._idx_dev:
            self._idx_dev[key] = self.text_indices.to(device)
        return self._idx_dev[key]

    def _run_sdpa_and_capture(
        self,
        jq_p: torch.Tensor,
        jk_p: torch.Tensor,
        jv_p: torch.Tensor,
        T_full: int,
        N: int,
    ) -> torch.Tensor:
        """Run standard SDPA for output; separately compute softmax weights for capture."""
        # Standard output via SDPA (efficient)
        full_out = F.scaled_dot_product_attention(jq_p, jk_p, jv_p)

        # Capture: compute image→all attention manually for conditional batch only.
        B, H, L, D_h = jq_p.shape
        ci = B - 1  # conditional batch
        scale = D_h ** -0.5

        # Only image queries (gen + ref), all keys
        img_start = T_full
        img_end = T_full + 2 * N
        q_img = jq_p[ci : ci + 1, :, img_start:img_end, :].float()  # [1, H, 2N, D_h]
        k_all = jk_p[ci : ci + 1].float()  # [1, H, L, D_h]

        scores = (q_img @ k_all.transpose(-2, -1)) * scale  # [1, H, 2N, L]
        weights = torch.softmax(scores, dim=-1)  # [1, H, 2N, L]

        # Select text columns of interest
        idx = self._get_idx(jq_p.device)  # [K]
        weights_text = weights[..., idx]  # [1, H, 2N, K]

        # Head-average
        weights_mean = weights_text.mean(dim=1)[0]  # [2N, K]

        gen_map = weights_mean[:N].cpu()  # [N, K]
        ref_map = weights_mean[N:].cpu()  # [N, K]
        self.captures_gen.append(gen_map)
        self.captures_ref.append(ref_map)

        return full_out

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

            # Single-stream layout: hidden_states = [text | gen | ref] of total length T_full + 2N.
            # We know N = self.N (grid_size^2) at init, so derive T_full directly.
            # (Fixes the earlier bug where _T_full only got set from double-stream neighbors.)
            L = hidden_states.shape[1]
            N = self.N
            T_full = L - 2 * N
            if T_full < 0:
                # Something unexpected; fall back to no capture
                full_out = F.scaled_dot_product_attention(jq_p, jk_p, jv_p)
            else:
                full_out = self._run_sdpa_and_capture(jq_p, jk_p, jv_p, T_full, N)

            return full_out.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)

        # ── Double-stream block ─────────────────────────────
        # Record layout info so single-stream blocks (later in same step) can also capture
        self._T_full = encoder_hidden_states.shape[1]
        self._N_seq = hidden_states.shape[1] // 2

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
        full_out = self._run_sdpa_and_capture(jq_p, jk_p, jv_p, T_full, N)

        full_out = full_out.permute(0, 2, 1, 3).flatten(2, 3).to(jq.dtype)
        T_p = encoder_hidden_states.shape[1]
        enc_out = full_out[:, :T_p]
        img_out = full_out[:, T_p:]
        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        enc_out = attn.to_add_out(enc_out)
        return img_out, enc_out


def install_capture_processors(
    pipe,
    text_indices: list[int],
    grid_size: int = 64,
    layer_filter: str = "all",
) -> dict[str, AttentionCaptureProcessor]:
    """Install one AttentionCaptureProcessor per attention block.

    layer_filter:
      - "all"    : install on all attention layers
      - "double" : only double-stream blocks
      - "single" : only single-stream blocks
    """
    processors: dict[str, AttentionCaptureProcessor] = {}
    original: dict[str, object] = {}

    for name, module in pipe.transformer.named_modules():
        if not (name.endswith(".attn") and hasattr(module, "set_processor")):
            continue
        is_double = "transformer_blocks" in name and "single_transformer_blocks" not in name
        is_single = "single_transformer_blocks" in name

        if layer_filter == "double" and not is_double:
            continue
        if layer_filter == "single" and not is_single:
            continue
        if layer_filter == "all" and not (is_double or is_single):
            continue

        proc = AttentionCaptureProcessor(text_indices, grid_size, layer_name=name)
        original[name] = module.processor
        module.set_processor(proc)
        processors[name] = proc

    n_double = sum(1 for n in processors if "single_transformer_blocks" not in n)
    n_single = len(processors) - n_double
    print(f"[Capture] installed on {len(processors)} layers (double={n_double}, single={n_single})")
    return processors, original


def restore_processors(pipe, original_procs: dict) -> None:
    for name, module in pipe.transformer.named_modules():
        if name in original_procs:
            module.set_processor(original_procs[name])
    print(f"[Capture] restored {len(original_procs)} layers")
