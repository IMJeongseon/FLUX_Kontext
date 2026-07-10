"""Empirical verification of the editability inequality.

For a given source image + prompt, compute per-position editability:

    E(y, x) = γ · [S_text(y,x) / S_ref(y,x)] · [‖μ_text‖ / ‖v_ref(y,x)‖]

Save heatmaps of:
    LHS  = γ · S_text / S_ref                 (text winning ratio)
    RHS  = ‖v_ref‖ / ‖μ_text‖                 (ref anchor strength)
    E    = LHS / RHS                          (editability score)

Overlay E on source image to visualize where editing is theoretically possible.

Usage:
    python verify_editability.py \\
        --input data/dog.png \\
        --prompt "Replace the dog with a cat and the house with a cat tower." \\
        --output-dir outputs/editability_dog/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
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


def viridis_colormap(x):
    """Simple viridis-like colormap for [0,1] scalar → RGB uint8."""
    import numpy as np
    x = np.clip(x, 0, 1)
    r = np.clip(np.abs(x - 0.75) * 4 - 0.5, 0, 1)
    g = np.clip(x * 1.5, 0, 1)
    b = np.clip((1 - x) * 1.3 + x * 0.4, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def save_heatmap(values_2d, source_pil, output_path, title, log_scale=False,
                 cmap="viridis", vmin=None, vmax=None):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    v = values_2d.copy().astype(np.float32)
    if log_scale:
        v = np.log10(np.maximum(v, 1e-8))
    if vmin is None:
        vmin = v.min()
    if vmax is None:
        vmax = v.max()
    v_n = (v - vmin) / (vmax - vmin + 1e-8)

    # Upsample
    W_full = source_pil.size[0]
    v_pil = Image.fromarray((v_n * 255).astype(np.uint8), mode="L").resize((W_full, W_full), Image.BILINEAR)
    v_arr = np.array(v_pil).astype(np.float32) / 255.0
    heat_rgb = viridis_colormap(v_arr)
    heat_pil = Image.fromarray(heat_rgb)

    # Overlay with source (50/50 blend)
    src_arr = np.array(source_pil.convert("RGB").resize((W_full, W_full)))
    blend = (src_arr * 0.4 + heat_rgb * 0.6).clip(0, 255).astype(np.uint8)
    blend_pil = Image.fromarray(blend)

    # Compose figure: source | heat | blend
    W, H = source_pil.size
    pad = 20
    lh = 40
    canvas = Image.new("RGB", (W * 3 + pad * 4, H + lh + pad * 2), (250, 250, 250))
    canvas.paste(source_pil.convert("RGB").resize((W, H)), (pad, lh + pad))
    canvas.paste(heat_pil.resize((W, H)), (pad * 2 + W, lh + pad))
    canvas.paste(blend_pil.resize((W, H)), (pad * 3 + W * 2, lh + pad))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    d.text((pad + 10, pad), f"Source", fill=(20, 20, 20), font=f)
    d.text((pad * 2 + W + 10, pad), f"{title}", fill=(20, 20, 20), font=f)
    d.text((pad * 3 + W * 2 + 10, pad), f"Overlay", fill=(20, 20, 20), font=f)
    canvas.save(output_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture-steps", type=int, default=1)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--layer-subset", default="middle-late",
                   choices=["all", "double", "single", "middle-late", "late", "middle"])
    p.add_argument("--max-memory", default="0:22GiB,1:22GiB")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from diffusers import FluxKontextPipeline
    from PIL import Image
    from editability_processor import (
        install_editability_processors,
        restore_processors,
    )

    src_img = Image.open(args.input).convert("RGB")
    W, H = src_img.size
    GRID = W // 16
    print(f"Image: {W}x{H}, grid_size={GRID}")
    print(f"Prompt: {args.prompt}")

    # ── Load Kontext ────────────────────────────
    print("\n[load] FLUX.1-Kontext-dev")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=parse_max_memory(args.max_memory),
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # ── Install capture ─────────────────────────
    processors, original = install_editability_processors(pipe, grid_size=GRID)

    # ── Run 1 step ──────────────────────────────
    print(f"\n[run] {args.capture_steps} step(s) for capture")
    _ = pipe(
        image=src_img,
        prompt=args.prompt,
        height=H,
        width=W,
        num_inference_steps=args.capture_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    )

    # ── Layer subset selection ──────────────────
    all_names = list(processors.keys())
    n_total = len(all_names)
    if args.layer_subset == "all":
        keep = list(range(n_total))
    elif args.layer_subset == "double":
        keep = [i for i, n in enumerate(all_names) if "single_transformer_blocks" not in n]
    elif args.layer_subset == "single":
        keep = [i for i, n in enumerate(all_names) if "single_transformer_blocks" in n]
    elif args.layer_subset == "middle-late":
        keep = list(range(int(n_total * 0.20), int(n_total * 0.95)))
    elif args.layer_subset == "late":
        keep = list(range(int(n_total * 0.55), int(n_total * 0.95)))
    elif args.layer_subset == "middle":
        keep = list(range(int(n_total * 0.30), int(n_total * 0.70)))
    keep_names = [all_names[i] for i in keep]
    print(f"\n[aggregate] layer subset '{args.layer_subset}': {len(keep_names)}/{n_total} layers")

    # ── Aggregate per-layer captures ────────────
    S_text_all = []
    S_ref_all = []
    v_ref_norm_all = []
    v_text_mean_norm_all = []
    for name in keep_names:
        proc = processors[name]
        if not proc.S_text_gen:
            continue
        # Each list has one entry per forward call. Mean over calls.
        S_text_all.append(torch.stack(proc.S_text_gen, dim=0).mean(dim=0))  # [N]
        S_ref_all.append(torch.stack(proc.S_ref_gen, dim=0).mean(dim=0))
        v_ref_norm_all.append(torch.stack(proc.v_ref_norm, dim=0).mean(dim=0))
        v_text_mean_norm_all.append(torch.stack(proc.v_text_mean_norm, dim=0).mean(dim=0))

    # Layer-mean
    S_text = torch.stack(S_text_all, dim=0).mean(dim=0).numpy()  # [N]
    S_ref = torch.stack(S_ref_all, dim=0).mean(dim=0).numpy()
    v_ref_norm = torch.stack(v_ref_norm_all, dim=0).mean(dim=0).numpy()
    v_text_mean_norm = torch.stack(v_text_mean_norm_all, dim=0).mean(dim=0).item()

    print(f"[stats] S_text: mean={S_text.mean():.4f}, range=[{S_text.min():.4f}, {S_text.max():.4f}]")
    print(f"[stats] S_ref:  mean={S_ref.mean():.4f}, range=[{S_ref.min():.4f}, {S_ref.max():.4f}]")
    print(f"[stats] v_ref_norm: mean={v_ref_norm.mean():.4f}, range=[{v_ref_norm.min():.4f}, {v_ref_norm.max():.4f}]")
    print(f"[stats] v_text_mean_norm: {v_text_mean_norm:.4f}")

    # ── Compute editability components ──────────
    gamma = args.guidance_scale
    LHS = gamma * S_text / np.maximum(S_ref, 1e-8)     # γ · S_text/S_ref
    RHS = v_ref_norm / max(v_text_mean_norm, 1e-8)     # ‖v_ref‖/‖μ_text‖
    E = LHS / np.maximum(RHS, 1e-8)                    # editability score

    print(f"\n[editability]")
    print(f"  LHS (γ · S_text/S_ref): mean={LHS.mean():.4f}, range=[{LHS.min():.4f}, {LHS.max():.4f}]")
    print(f"  RHS (‖v_ref‖/‖μ_text‖): mean={RHS.mean():.4f}, range=[{RHS.min():.4f}, {RHS.max():.4f}]")
    print(f"  E   = LHS/RHS:          mean={E.mean():.4f}, range=[{E.min():.4f}, {E.max():.4f}]")
    print(f"  fraction of E > 1 (editable): {(E > 1).mean() * 100:.1f}%")

    # ── Reshape to 2D grid ──────────────────────
    S_text_2d = S_text.reshape(GRID, GRID)
    S_ref_2d = S_ref.reshape(GRID, GRID)
    v_ref_2d = v_ref_norm.reshape(GRID, GRID)
    LHS_2d = LHS.reshape(GRID, GRID)
    RHS_2d = RHS.reshape(GRID, GRID)
    E_2d = E.reshape(GRID, GRID)

    # ── Save heatmaps ───────────────────────────
    print(f"\n[save] heatmaps → {out_dir}")
    save_heatmap(S_text_2d, src_img, out_dir / "01_S_text.png",
                 title=f"S_text (attn mass to text)")
    save_heatmap(S_ref_2d, src_img, out_dir / "02_S_ref.png",
                 title=f"S_ref (attn mass to ref)")
    save_heatmap(v_ref_2d, src_img, out_dir / "03_v_ref_norm.png",
                 title=f"‖v_ref‖ (ref value norm)")
    save_heatmap(LHS_2d, src_img, out_dir / "04_LHS.png",
                 title=f"LHS = γ · S_text/S_ref")
    save_heatmap(RHS_2d, src_img, out_dir / "05_RHS.png",
                 title=f"RHS = ‖v_ref‖/‖μ_text‖")
    save_heatmap(E_2d, src_img, out_dir / "06_E_editability.png",
                 title=f"Editability E = LHS/RHS")
    save_heatmap(np.log10(np.maximum(E_2d, 1e-6)), src_img, out_dir / "07_E_log10.png",
                 title=f"log₁₀(E)  (>0 editable)", vmin=-2, vmax=2)

    # ── Save summary text ───────────────────────
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Editability verification\n")
        f.write(f"========================\n")
        f.write(f"Input:  {args.input}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"γ (CFG): {gamma}\n")
        f.write(f"Layer subset: {args.layer_subset} ({len(keep_names)} layers)\n")
        f.write(f"\nQuantities (mean over layers, head-averaged):\n")
        f.write(f"  S_text:            mean={S_text.mean():.4f}  range=[{S_text.min():.4f}, {S_text.max():.4f}]\n")
        f.write(f"  S_ref:             mean={S_ref.mean():.4f}  range=[{S_ref.min():.4f}, {S_ref.max():.4f}]\n")
        f.write(f"  ‖v_ref‖:           mean={v_ref_norm.mean():.4f}  range=[{v_ref_norm.min():.4f}, {v_ref_norm.max():.4f}]\n")
        f.write(f"  ‖μ_text‖:          {v_text_mean_norm:.4f}\n")
        f.write(f"\nInequality components:\n")
        f.write(f"  LHS = γ·S_text/S_ref: mean={LHS.mean():.4f}  range=[{LHS.min():.4f}, {LHS.max():.4f}]\n")
        f.write(f"  RHS = ‖v_ref‖/‖μ_text‖: mean={RHS.mean():.4f}  range=[{RHS.min():.4f}, {RHS.max():.4f}]\n")
        f.write(f"  E   = LHS/RHS: mean={E.mean():.4f}  range=[{E.min():.4f}, {E.max():.4f}]\n")
        f.write(f"\nEditable region fraction (E > 1): {(E > 1).mean() * 100:.1f}%\n")
        f.write(f"Highly editable (E > 3):          {(E > 3).mean() * 100:.1f}%\n")
        f.write(f"Very anchored (E < 0.3):          {(E < 0.3).mean() * 100:.1f}%\n")
    print(f"[save] summary → {out_dir/'summary.txt'}")

    restore_processors(pipe, original)
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
