"""Extract self-derived object masks from FLUX.1-Kontext's own cross-attention.

No external detector (GroundingDINO, SAM) needed. Uses the model's img→text attention
weights, aggregated across layers and heads, to localize where each prompt word
appears in the source image.

Usage:
    python extract_self_mask.py \\
        --input data/image.png \\
        --prompt "Replace the dog with a cat and the house with a cat tower." \\
        --source-words "dog,house" \\
        --output-dir outputs/self_masks/
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


def aggregate_attention_maps(processors, source: str, layer_filter=None):
    """Aggregate captured maps across all layers.

    Returns tensor [N, K] where K = number of text indices.
    source: 'gen' or 'ref'
    layer_filter: optional list of layer name substrings to keep
    """
    import torch
    all_maps = []
    for name, proc in processors.items():
        if layer_filter is not None:
            if not any(f in name for f in layer_filter):
                continue
        captures = proc.captures_gen if source == "gen" else proc.captures_ref
        if not captures:
            continue
        # Each capture is [N, K]. Stack across time steps and mean.
        stacked = torch.stack(captures, dim=0).mean(dim=0)  # [N, K]
        all_maps.append(stacked)
    if not all_maps:
        raise RuntimeError("no captured maps found")
    # Mean over layers
    agg = torch.stack(all_maps, dim=0).mean(dim=0)  # [N, K]
    return agg


def maps_to_masks(agg_maps, grid_size, top_k_percent=15.0):
    """Convert aggregated [N, K] attention to per-word binary masks."""
    import numpy as np
    N, K = agg_maps.shape
    grid_size2 = grid_size
    masks = []
    heatmaps = []
    for k in range(K):
        heat = agg_maps[:, k].numpy().reshape(grid_size2, grid_size2)
        # Normalize
        heat_n = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heatmaps.append(heat_n)
        # Threshold at top-k%
        thr = np.percentile(heat_n, 100 - top_k_percent)
        mask = heat_n >= thr
        masks.append(mask)
    return masks, heatmaps


def upscale_mask(mask_small, target_hw):
    """Upsample a low-resolution mask to target image size using nearest-neighbor."""
    import numpy as np
    from PIL import Image
    m = Image.fromarray((mask_small * 255).astype(np.uint8), mode="L")
    m = m.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return np.array(m) > 128


def save_visualization(src_pil, mask_bool, heat, word, path):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    src_arr = np.array(src_pil.convert("RGB"))
    H, W = src_arr.shape[:2]

    # Mask overlay
    ov = src_arr.copy()
    m = mask_bool
    ov[m] = (src_arr[m] * 0.35 + np.array([255, 80, 0]) * 0.65).clip(0, 255).astype(np.uint8)
    ov_img = Image.fromarray(ov)

    # Heat visualization (jet-like, resized to full image)
    heat_full = Image.fromarray((heat * 255).astype(np.uint8), mode="L").resize((W, H), Image.BILINEAR)
    heat_arr = np.array(heat_full)
    heat_rgb = np.stack([
        np.clip(heat_arr * 2, 0, 255),                    # R
        np.clip(np.abs(255 - heat_arr * 2), 0, 255),      # G
        np.clip((255 - heat_arr) * 2, 0, 255),            # B (roughly jet)
    ], axis=-1).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb)

    pad = 20
    lh = 40
    canvas = Image.new("RGB", (W * 3 + pad * 4, H + lh + pad * 2), (250, 250, 250))
    canvas.paste(src_pil.resize((W, H)), (pad, lh + pad))
    canvas.paste(heat_img, (pad * 2 + W, lh + pad))
    canvas.paste(ov_img, (pad * 3 + W * 2, lh + pad))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    d.text((pad + 10, pad), "Source", fill=(20, 20, 20), font=f)
    d.text((pad * 2 + W + 10, pad), f"Attn heatmap for '{word}'", fill=(20, 20, 20), font=f)
    d.text((pad * 3 + W * 2 + 10, pad), f"Threshold mask (top-k%)", fill=(20, 20, 20), font=f)
    canvas.save(path)


def compute_iou(mask_a, mask_b):
    import numpy as np
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return inter / max(union, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--source-words", required=True,
                   help="comma-separated words in prompt to localize (e.g. 'dog,house')")
    p.add_argument("--output-dir", default=str(HERE / "outputs" / "self_masks"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture-steps", type=int, default=1,
                   help="how many denoising steps to capture over")
    p.add_argument("--top-k-percent", type=float, default=15.0,
                   help="top-k%% of attention values become the mask")
    p.add_argument("--layer-filter", default="all", choices=["all", "double", "single"])
    p.add_argument("--layer-range", default=None,
                   help="'start:end' e.g. '10:40' for aggregation, uses layer index in installed order")
    p.add_argument("--max-memory", default="0:22GiB,1:22GiB")
    p.add_argument("--compare-masks", default=None,
                   help="comma-separated paths to reference masks (SAM/GroundingDINO) for IoU compare")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_words = [w.strip() for w in args.source_words.split(",") if w.strip()]
    print(f"Source words: {source_words}")
    print(f"Prompt: {args.prompt}")

    import numpy as np
    import torch
    from diffusers import FluxKontextPipeline
    from PIL import Image
    from transformers import T5TokenizerFast
    from attention_capture_processor import install_capture_processors, restore_processors

    src_img = Image.open(args.input).convert("RGB")
    W, H = src_img.size
    GRID = W // 16
    print(f"Image: {W}x{H}, grid_size={GRID}")

    # ── Tokenize + find text indices ────────────────
    T_FULL = 512
    tokenizer = T5TokenizerFast.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", subfolder="tokenizer_2"
    )
    text_indices_per_word = {w: find_token_span(tokenizer, args.prompt, w, T_FULL)
                             for w in source_words}
    all_indices = []
    per_word_offset = {}
    for w in source_words:
        per_word_offset[w] = list(range(len(all_indices),
                                        len(all_indices) + len(text_indices_per_word[w])))
        all_indices.extend(text_indices_per_word[w])
    print(f"Token indices per word: {text_indices_per_word}")
    print(f"Flattened text indices captured: {all_indices}")

    # ── Load Kontext ────────────────────────────────
    print("[load] FLUX.1-Kontext-dev  device_map=balanced")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=parse_max_memory(args.max_memory),
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # ── Install capture processors ─────────────────
    processors, original = install_capture_processors(
        pipe, all_indices, grid_size=GRID, layer_filter=args.layer_filter,
    )

    # ── Run limited denoising (only capture-steps) ──
    print(f"[run] running {args.capture_steps} step(s) for attention capture")
    _ = pipe(
        image=src_img,
        prompt=args.prompt,
        height=H,
        width=W,
        num_inference_steps=args.capture_steps,
        guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    )

    # ── Aggregate ────────────────────────────────────
    layer_filter = None
    if args.layer_range:
        s, e = [int(x) for x in args.layer_range.split(":")]
        # Layer names are like "transformer_blocks.5.attn" or "single_transformer_blocks.12.attn"
        # We'll just take the (s..e)-th layer among the installed ones by insertion order.
        names = list(processors.keys())
        layer_filter = names[s:e]
        print(f"[aggregate] using layers [{s}:{e}] = {len(layer_filter)} layers")

    print("[aggregate] gen→text")
    agg_gen = aggregate_attention_maps(processors, "gen", layer_filter)
    print("[aggregate] ref→text")
    agg_ref = aggregate_attention_maps(processors, "ref", layer_filter)

    # ── Combine per-word (sum over the word's token indices) ──
    print("[combine] per-word maps + save")
    saved_paths = {}
    for w in source_words:
        offs = per_word_offset[w]
        gen_w = agg_gen[:, offs].sum(dim=1)  # [N]
        ref_w = agg_ref[:, offs].sum(dim=1)  # [N]

        # Reshape both, save separately
        for name_src, agg_map in [("gen", gen_w), ("ref", ref_w)]:
            heat = agg_map.numpy().reshape(GRID, GRID)
            heat_n = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            thr = np.percentile(heat_n, 100 - args.top_k_percent)
            mask_small = heat_n >= thr
            mask_full = upscale_mask(mask_small, (H, W))
            # Save mask
            mp = out_dir / f"mask_{w}_{name_src}.png"
            Image.fromarray((mask_full * 255).astype(np.uint8), mode="L").save(mp)
            saved_paths[(w, name_src)] = mp
            # Save visualization
            save_visualization(src_img, mask_full, heat_n, f"{w} ({name_src})",
                               out_dir / f"viz_{w}_{name_src}.png")
            print(f"  '{w}' via {name_src}: mask={mp.name}, {mask_full.sum()} px "
                  f"({mask_full.sum()/(H*W)*100:.1f}%)")

    # ── Compare with reference masks (if provided) ──
    if args.compare_masks:
        print("\n[compare] IoU vs external masks")
        ref_paths = [p.strip() for p in args.compare_masks.split(",")]
        assert len(ref_paths) == len(source_words), \
            f"compare-masks count ({len(ref_paths)}) must match source-words ({len(source_words)})"
        for w, ref_path in zip(source_words, ref_paths):
            ref_mask = np.array(Image.open(ref_path).convert("L")) > 128
            for name_src in ["gen", "ref"]:
                mp = saved_paths[(w, name_src)]
                self_mask = np.array(Image.open(mp).convert("L")) > 128
                iou = compute_iou(self_mask, ref_mask)
                print(f"  '{w}' ({name_src}) vs {Path(ref_path).name}: IoU = {iou:.3f}")

    restore_processors(pipe, original)
    print(f"\nDone. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
