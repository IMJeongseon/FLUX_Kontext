"""MRAS Pipeline v2: detector-free end-to-end editing.

Two-phase pipeline in a single process:

  Phase 1 (~15s): 1-step attention capture → self-derived per-object masks
                  (replaces GroundingDINO + SAM entirely)

  Phase 2 (~6min): 28-step regional-binding denoising with those masks
                   (same MRAS logic as mras_pipeline.py)

Usage:
    python mras_pipeline_v2.py \\
        --input data/dog.png \\
        --output outputs/pipeline_v2.png \\
        --prompt "Replace the dog with a cat and the house with a cat tower." \\
        --source-objects "dog,house" \\
        --target-phrases "cat;cat tower" \\
        --save-mask-figure
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MRAS_SRC = HERE.parent / "MRAS" / "src"
sys.path.insert(0, str(MRAS_SRC))
sys.path.insert(0, str(HERE))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_max_memory(spec):
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        k, v = item.split(":", 1)
        k = k.strip()
        out[int(k) if k.isdigit() else k] = v.strip()
    return out


def find_token_span(tokenizer, prompt, target, T_full=512):
    full = tokenizer(prompt, return_tensors="pt",
                     padding="max_length", max_length=T_full, truncation=True
                     )["input_ids"][0].tolist()
    for variant in [" " + target, target]:
        tgt = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not tgt:
            continue
        n = len(tgt)
        for i in range(len(full) - n + 1):
            if full[i:i + n] == tgt:
                return list(range(i, i + n))
    raise ValueError(f"could not locate token span for '{target}' in prompt")


def encode_and_pack(pipe, image):
    import torch
    from torchvision.transforms.functional import to_tensor
    vae_dev = next(pipe.vae.parameters()).device
    vae_dtype = next(pipe.vae.parameters()).dtype
    t = (to_tensor(image).unsqueeze(0) * 2 - 1).to(vae_dev, vae_dtype)
    with torch.no_grad():
        lat = pipe.vae.encode(t).latent_dist.sample()
        lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    b, c, lh, lw = lat.shape
    packed = lat.view(b, c, lh // 2, 2, lw // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(b, (lh // 2) * (lw // 2), c * 4).cpu().float()


def make_repaint_callback(proc, src_packed, bg_tensor, repaint_until_step=28, feather_ramp=0):
    import torch

    def callback(pipeline, step_idx, timestep, kwargs):
        proc.current_step = step_idx
        latents = kwargs["latents"]
        device, dtype = latents.device, latents.dtype

        if step_idx >= repaint_until_step:
            return kwargs
        ramp_start = repaint_until_step - feather_ramp
        if step_idx < ramp_start or feather_ramp == 0:
            alpha = 1.0
        else:
            alpha = max(0.0, 1.0 - (step_idx - ramp_start + 1) / (feather_ramp + 1))

        sched = pipeline.scheduler
        if step_idx + 1 < len(sched.timesteps):
            next_sigma = float(sched.timesteps[step_idx + 1]) / 1000.0
        else:
            next_sigma = 0.0
        slp = src_packed.to(device, dtype)
        noise = torch.randn_like(slp)
        noised = (1.0 - next_sigma) * slp + next_sigma * noise
        bg = bg_tensor.to(device)
        latents = latents.clone()
        expanded = noised.expand(latents.shape[0], -1, -1)
        latents[:, bg, :] = alpha * expanded[:, bg, :] + (1 - alpha) * latents[:, bg, :]
        kwargs["latents"] = latents
        return kwargs

    return callback


def build_mask_from_attention(agg_ref, per_word_offset, source_words,
                              grid_size, top_k_percent,
                              gaussian_sigma=1.5,
                              morph_close_radius=2,
                              morph_open_radius=1):
    """From aggregated ref→text attention, build a binary mask per source word.

    Pipeline per word:
        1. Sum attention over the word's token indices → 2D grid heatmap
        2. Normalize [0, 1]
        3. Gaussian smoothing on the grid (removes checkerboard)
        4. Bicubic upsample to full resolution
        5. Threshold at top-k%
        6. Morphological open (remove isolated pixels) → close (fill holes)

    Args:
        agg_ref: [N, K] tensor of aggregated attention weights
        per_word_offset: dict word → list of indices in K
        grid_size: 64 for 1024x1024
        top_k_percent: e.g. 10.0
        gaussian_sigma: pre-threshold Gaussian smoothing (grid units)
        morph_close_radius: post-threshold morphological close radius (px)
        morph_open_radius: post-threshold morphological open radius (px)

    Returns:
        masks_bool: dict word → [H, W] boolean numpy array
        heatmaps_norm: dict word → [grid_size, grid_size] normalized float array
    """
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter, binary_closing, binary_opening

    def _disk_kernel(radius):
        if radius <= 0:
            return None
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        return (x ** 2 + y ** 2) <= radius ** 2

    masks_bool = {}
    heatmaps_norm = {}
    W_full = grid_size * 16
    for w in source_words:
        offs = per_word_offset[w]
        ref_w = agg_ref[:, offs].sum(dim=1).numpy()  # [N]
        heat = ref_w.reshape(grid_size, grid_size).astype(np.float32)

        # (a) Gaussian smoothing on the low-res grid
        if gaussian_sigma > 0:
            heat = gaussian_filter(heat, sigma=gaussian_sigma)

        # (b) Normalize
        heat_n = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heatmaps_norm[w] = heat_n

        # (c) Bicubic upsample to full res (smoother than nearest)
        heat_pil = Image.fromarray((heat_n * 255).astype(np.uint8), mode="L")
        heat_full = np.array(heat_pil.resize((W_full, W_full), Image.BICUBIC)).astype(np.float32) / 255.0

        # (d) Top-k% threshold on the SMOOTHED full-res heat
        thr = np.percentile(heat_full, 100 - top_k_percent)
        mask_full = heat_full >= thr

        # (e) Morphological open then close to clean up + fill
        if morph_open_radius > 0:
            k = _disk_kernel(morph_open_radius)
            mask_full = binary_opening(mask_full, structure=k)
        if morph_close_radius > 0:
            k = _disk_kernel(morph_close_radius)
            mask_full = binary_closing(mask_full, structure=k)

        masks_bool[w] = mask_full
    return masks_bool, heatmaps_norm


def save_mask_figure(src, out, source_masks, target_phrases, output_path,
                     heatmaps=None, label="Output"):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    palette = [("orange", (255, 80, 0)), ("blue", (0, 140, 255)),
               ("green", (0, 200, 100)), ("purple", (200, 0, 200))]

    def overlay(img, entries, alpha=0.55):
        a = np.array(img.convert("RGB"))
        o = a.copy()
        for m_arr, color in entries:
            m = m_arr > 0
            o[m] = (a[m] * (1 - alpha) + np.array(color) * alpha).clip(0, 255).astype(np.uint8)
        return Image.fromarray(o)

    entries, legend_parts = [], []
    for i, (name, m) in enumerate(source_masks.items()):
        cname, ccode = palette[i % len(palette)]
        entries.append((m, ccode))
        tgt = target_phrases[i]
        legend_parts.append(f"{cname}={name}→{tgt}")

    src_o = overlay(src, entries)
    out_o = overlay(out, entries)
    W, H = src.size
    pad, lh = 20, 40
    canvas = Image.new("RGB", (W * 2 + pad * 3, H + lh + pad * 2), (250, 250, 250))
    canvas.paste(src_o, (pad, lh + pad))
    canvas.paste(out_o, (pad * 2 + W, lh + pad))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    legend = " | ".join(legend_parts)
    d.text((pad + 10, pad), f"Source + self-masks   ({legend})", fill=(20, 20, 20), font=f)
    d.text((pad * 2 + W + 10, pad), f"{label} + masks", fill=(20, 20, 20), font=f)
    fig_path = Path(output_path).with_name(Path(output_path).stem + "_masks.png")
    canvas.save(fig_path)
    print(f"[figure] saved: {fig_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--prompt",
        default="Replace the dog with a cat and the house with a cat tower.",
    )
    p.add_argument("--source-objects", default="dog,house")
    p.add_argument("--target-phrases", default="cat;cat tower")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--suppression-bias", type=float, default=-1e4)
    p.add_argument("--target-bias", type=float, default=0.0,
                   help="+ bias for own target text; 0 = only negative suppression (safer default)")
    p.add_argument("--apply-until-step", type=int, default=28)
    p.add_argument("--repaint-until-step", type=int, default=28)
    p.add_argument("--repaint-feather", type=int, default=0)
    p.add_argument("--top-k-percent", type=float, default=8.0,
                   help="Self-derived mask threshold: top-k%% of attention becomes mask")
    p.add_argument("--capture-steps", type=int, default=3,
                   help="Denoising steps for attention capture (Phase 1)")
    p.add_argument("--gaussian-sigma", type=float, default=1.5,
                   help="Pre-threshold Gaussian smoothing radius (grid units)")
    p.add_argument("--morph-close", type=int, default=4,
                   help="Morphological closing radius (px, fills holes)")
    p.add_argument("--morph-open", type=int, default=2,
                   help="Morphological opening radius (px, removes isolated pixels)")
    p.add_argument("--layer-subset", default="middle-late",
                   choices=["all", "double", "single", "middle-late", "late", "middle"],
                   help="Which layers to aggregate for mask extraction")
    p.add_argument("--max-memory", default="0:22GiB,1:22GiB")
    p.add_argument("--save-mask-figure", action="store_true")
    args = p.parse_args()

    sources = [s.strip() for s in args.source_objects.split(",") if s.strip()]
    targets = [t.strip() for t in args.target_phrases.split(";") if t.strip()]
    if len(sources) != len(targets):
        raise ValueError(f"source-objects count ({len(sources)}) must match target-phrases ({len(targets)})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Prompt : {args.prompt}")
    print(f"Sources: {sources}")
    print(f"Targets: {targets}")

    import numpy as np
    import torch
    from diffusers import FluxKontextPipeline
    from PIL import Image
    from transformers import T5TokenizerFast
    from pom1.mras import mask_to_token_indices
    from regional_processor import RegionalMRASProcessor, install as install_regional
    from attention_capture_processor import (
        install_capture_processors,
        restore_processors,
    )

    src_img = Image.open(args.input).convert("RGB")
    W, H = src_img.size
    GRID = W // 16
    N = GRID * GRID
    T_FULL = 512

    # ── Tokenize prompt ─────────────────────────
    print("\n[tokenize] find text spans")
    tokenizer = T5TokenizerFast.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", subfolder="tokenizer_2"
    )
    source_spans = {s: find_token_span(tokenizer, args.prompt, s, T_FULL) for s in sources}
    target_spans = {t: find_token_span(tokenizer, args.prompt, t, T_FULL) for t in targets}
    for s in sources:
        print(f"  source '{s}': tokens {source_spans[s]}")
    for t in targets:
        print(f"  target '{t}': tokens {target_spans[t]}")

    # Build the flattened list of source-word text indices to capture
    all_capture_indices = []
    per_word_offset = {}
    for s in sources:
        per_word_offset[s] = list(range(len(all_capture_indices),
                                        len(all_capture_indices) + len(source_spans[s])))
        all_capture_indices.extend(source_spans[s])

    # ── Load model ──────────────────────────────
    print("\n[load] FLUX.1-Kontext-dev  device_map=balanced")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=parse_max_memory(args.max_memory),
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # ═══════════════════════════════════════════
    # PHASE 1: Attention capture → self-derived masks
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Self-derived mask extraction (1 forward pass)")
    print("=" * 60)

    processors_capture, original_procs = install_capture_processors(
        pipe, all_capture_indices, grid_size=GRID, layer_filter="all",
    )

    _ = pipe(
        image=src_img,
        prompt=args.prompt,
        height=H,
        width=W,
        num_inference_steps=args.capture_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    )

    # Select layer subset for aggregation
    all_layer_names = list(processors_capture.keys())
    # Layer ordering: double blocks first (0..18), then single blocks (19..56)
    # For "middle-late"/"late"/"middle", we treat them by index in installed order.
    n_total = len(all_layer_names)
    if args.layer_subset == "all":
        keep = list(range(n_total))
    elif args.layer_subset == "double":
        keep = [i for i, n in enumerate(all_layer_names) if "single_transformer_blocks" not in n]
    elif args.layer_subset == "single":
        keep = [i for i, n in enumerate(all_layer_names) if "single_transformer_blocks" in n]
    elif args.layer_subset == "middle-late":
        # Skip first ~20% (early, texture bias) and last ~5% (output projection).
        keep = list(range(int(n_total * 0.20), int(n_total * 0.95)))
    elif args.layer_subset == "late":
        keep = list(range(int(n_total * 0.55), int(n_total * 0.95)))
    elif args.layer_subset == "middle":
        keep = list(range(int(n_total * 0.30), int(n_total * 0.70)))
    keep_names = [all_layer_names[i] for i in keep]

    print(f"[aggregate] layer subset '{args.layer_subset}': {len(keep_names)}/{n_total} layers")

    all_ref_maps = []
    for name in keep_names:
        proc = processors_capture[name]
        if not proc.captures_ref:
            continue
        stacked = torch.stack(proc.captures_ref, dim=0).mean(dim=0)  # [N, K]
        all_ref_maps.append(stacked)
    if not all_ref_maps:
        raise RuntimeError("no layers captured — check single-stream fix and layer-subset")
    agg_ref = torch.stack(all_ref_maps, dim=0).mean(dim=0)  # [N, K]
    print(f"  effective captures: {len(all_ref_maps)} layers")

    source_masks_bool, heatmaps = build_mask_from_attention(
        agg_ref, per_word_offset, sources, GRID, args.top_k_percent,
        gaussian_sigma=args.gaussian_sigma,
        morph_close_radius=args.morph_close,
        morph_open_radius=args.morph_open,
    )
    for s in sources:
        m = source_masks_bool[s]
        print(f"  self-mask '{s}': {m.sum()} px ({m.sum()/(H*W)*100:.1f}%)")

    # Restore original processors before Phase 2 install
    restore_processors(pipe, original_procs)

    # ═══════════════════════════════════════════
    # PHASE 2: Regional binding denoising
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Regional-binding editing (28 steps)")
    print("=" * 60)

    # Convert masks to token index lists
    source_token_lists = {}
    for s in sources:
        m_pil = Image.fromarray((source_masks_bool[s] * 255).astype(np.uint8), mode="L")
        source_token_lists[s] = mask_to_token_indices(m_pil, grid_size=GRID)
        print(f"  '{s}' mask tokens: {len(source_token_lists[s])} / {N}")

    # Union gen list with per-token region label
    seen = {}
    all_gen = []
    for s in sources:
        for g in source_token_lists[s]:
            if g not in seen:
                seen[g] = s
                all_gen.append(g)
    n_gen = len(all_gen)
    bg_idx = [i for i in range(N) if i not in seen]
    print(f"  gen union: {n_gen}, background: {len(bg_idx)}")

    # Build text_bias / ref_bias matrices (same logic as mras_pipeline.py)
    text_bias = torch.zeros(n_gen, T_FULL, dtype=torch.float32)
    ref_bias = torch.zeros(n_gen, N, dtype=torch.float32)
    ref_sets = {s: set(source_token_lists[s]) for s in sources}
    own_target = {s: targets[j] for j, s in enumerate(sources)}

    for i, g in enumerate(all_gen):
        own_src = seen[g]
        # block copy of own source appearance
        for r in ref_sets[own_src]:
            ref_bias[i, r] = args.suppression_bias
        # suppress OTHER sources' source words + destination phrases
        for j, s in enumerate(sources):
            if s == own_src:
                continue
            for t in source_spans[s]:
                text_bias[i, t] = args.suppression_bias
            for t in target_spans[targets[j]]:
                text_bias[i, t] = args.suppression_bias
        # (optional) positively boost own target
        if args.target_bias != 0:
            for t in target_spans[own_target[own_src]]:
                text_bias[i, t] = args.target_bias

    print(f"  text_bias nonzeros: {(text_bias != 0).sum().item()}")
    print(f"  ref_bias  nonzeros: {(ref_bias != 0).sum().item()}")

    src_packed = encode_and_pack(pipe, src_img)

    proc = RegionalMRASProcessor(
        gen_indices=all_gen,
        text_bias=text_bias,
        ref_bias=ref_bias,
        apply_until_step=args.apply_until_step,
        N_total=N,
    )
    install_regional(pipe, proc)

    bg_tensor = torch.tensor(bg_idx, dtype=torch.long)
    callback = make_repaint_callback(
        proc, src_packed, bg_tensor,
        repaint_until_step=args.repaint_until_step,
        feather_ramp=args.repaint_feather,
    )

    print(f"[run] denoising {args.steps} steps  gs={args.guidance_scale}  seed={args.seed}")
    result = pipe(
        image=src_img,
        prompt=args.prompt,
        height=H,
        width=W,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        callback_on_step_end=callback,
    ).images[0]

    result.save(output_path)
    print(f"[output] saved: {output_path}")

    if args.save_mask_figure:
        save_mask_figure(src_img, result, source_masks_bool, targets, output_path,
                         heatmaps=heatmaps, label="Output (self-masks)")


if __name__ == "__main__":
    main()
