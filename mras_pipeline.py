"""End-to-end MRAS regional-binding pipeline for FLUX.1-Kontext.

Usage:
    python3 mras_pipeline.py --input data/image.png --output outputs/output.png

Full default pipeline (dog→cat, house→cat tower, ball preserved):
    1. GroundingDINO + SAM: 각 source object 마스크 생성
    2. Preserve objects (예: ball) 감산
    3. Prompt tokenize → source/target 텍스트 span 자동 탐지
    4. RegionalMRASProcessor 로 per-object text/ref bias 주입
    5. FLUX-Kontext 로 편집 + RePaint 배경 앵커링
    6. (선택) mask overlay figure 저장

Requires two 24GB GPUs (device_map='balanced'). Change --max-memory for other setups.
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
# Deterministic execution: same seed → same result across runs (bit-exact when possible).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────
def parse_max_memory(spec):
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        k, v = item.split(":", 1)
        k = k.strip()
        out[int(k) if k.isdigit() else k] = v.strip()
    return out


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


def make_repaint_callback(proc, src_packed, bg_tensor, repaint_until_step=None, feather_ramp=4):
    """RePaint bg latent anchor with soft ramp-off near the end.

    - Steps < (repaint_until_step - feather_ramp): full strength (alpha=1.0)
    - Steps in [end - feather_ramp, end): linearly decaying alpha (1 → 0)
    - Steps >= repaint_until_step: no RePaint (model free to refine boundary)
    """
    import torch

    def callback(pipeline, step_idx, timestep, kwargs):
        proc.current_step = step_idx
        latents = kwargs["latents"]
        device, dtype = latents.device, latents.dtype

        if repaint_until_step is not None:
            if step_idx >= repaint_until_step:
                return kwargs
            # Linear ramp
            ramp_start = repaint_until_step - feather_ramp
            if step_idx < ramp_start:
                alpha = 1.0
            else:
                alpha = max(0.0, 1.0 - (step_idx - ramp_start + 1) / (feather_ramp + 1))
        else:
            alpha = 1.0

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


def dilate_bool(mask_bool, radius):
    from scipy.ndimage import binary_dilation
    import numpy as np
    if radius <= 0:
        return mask_bool
    yg, xg = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (xg ** 2 + yg ** 2 <= radius ** 2)
    return binary_dilation(mask_bool, structure=kernel)


# ─────────────────────────────────────────────────
# GroundingDINO + SAM segmentation
# ─────────────────────────────────────────────────
def detect_and_segment(image_pil, text_prompt, box_thr, txt_thr, dilation, device):
    """Detect objects (GroundingDINO) → segment (SAM) → union mask (numpy bool)."""
    import numpy as np
    import torch
    from transformers import (
        AutoModelForZeroShotObjectDetection, AutoProcessor,
        SamModel, SamProcessor,
    )

    W, H = image_pil.size

    dino_id = "IDEA-Research/grounding-dino-base"
    dino_proc = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_id, torch_dtype=torch.float32
    ).to(device)
    inputs = dino_proc(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = dino_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=box_thr, text_threshold=txt_thr,
        target_sizes=[(H, W)],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    del dino_model
    torch.cuda.empty_cache()

    if len(boxes) == 0:
        return np.zeros((H, W), dtype=bool), boxes

    sam_id = "facebook/sam-vit-base"
    sam_proc = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id, torch_dtype=torch.float32).to(device)
    combined = np.zeros((H, W), dtype=bool)
    for box in boxes:
        inp = sam_proc(images=image_pil, input_boxes=[[[*box]]], return_tensors="pt").to(device)
        with torch.no_grad():
            out = sam_model(**inp)
        masks = sam_proc.post_process_masks(
            out.pred_masks.cpu(),
            inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu(),
        )[0]
        best = int(out.iou_scores[0, 0].cpu().numpy().argmax())
        combined |= masks[0, best].numpy().astype(bool)
    del sam_model
    torch.cuda.empty_cache()

    if dilation > 0:
        combined = dilate_bool(combined, dilation)
    return combined, boxes


# ─────────────────────────────────────────────────
# Prompt tokenization: find token spans automatically
# ─────────────────────────────────────────────────
def find_token_span(tokenizer, prompt, target, T_full=512):
    """Return the token indices covering `target` inside `prompt` (padded to T_full)."""
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


# ─────────────────────────────────────────────────
# Mask overlay figure
# ─────────────────────────────────────────────────
def save_mask_figure(src, out, source_masks, target_phrases, output_path):
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
    d.text((pad + 10, pad), f"Source + masks   ({legend})", fill=(20, 20, 20), font=f)
    d.text((pad * 2 + W + 10, pad), "Output + masks", fill=(20, 20, 20), font=f)
    fig_path = Path(output_path).with_name(Path(output_path).stem + "_masks.png")
    canvas.save(fig_path)
    print(f"[figure] saved: {fig_path}")


# ─────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FLUX.1-Kontext + MRAS regional binding editing pipeline"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt",
        default="Replace the dog with a cat and the house with a cat tower.",
    )
    parser.add_argument("--source-objects", default="dog,house",
                        help="comma-separated source terms for GroundingDINO")
    parser.add_argument("--target-phrases", default="cat;cat tower",
                        help="semicolon-separated destination phrases (aligned with sources)")
    parser.add_argument("--preserve-objects", default="ball",
                        help="comma-separated objects to subtract from source masks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--suppression-bias", type=float, default=-1e4)
    parser.add_argument("--target-bias", type=float, default=5.0,
                        help="positive attention bias for each region toward its own target text tokens")
    parser.add_argument("--apply-until-step", type=int, default=28)
    parser.add_argument("--box-thr", type=float, default=0.25)
    parser.add_argument("--txt-thr", type=float, default=0.20)
    parser.add_argument("--dilation", type=int, default=32,
                        help="mask dilation (px) - larger allows cat silhouette to overflow original dog outline")
    parser.add_argument("--preserve-dilation", type=int, default=8)
    parser.add_argument("--repaint-until-step", type=int, default=20,
                        help="release RePaint bg anchoring after this step for natural boundary blend (with soft ramp)")
    parser.add_argument("--repaint-feather", type=int, default=4,
                        help="linear ramp length (steps) before RePaint stops")
    parser.add_argument("--core-erode", type=int, default=0,
                        help="[legacy] uniform erosion px to split source mask. Ignored if --target-shape is used.")
    parser.add_argument("--target-shape", default=None,
                        help="per-source target core shape overrides. Semicolon-separated 'source:WxH' in "
                             "grid tokens (16px each). E.g. 'dog:16x28' → 256x448 px vertical rect for the "
                             "cat, centered at dog's centroid. Sources missing here keep full mask as core.")
    parser.add_argument("--background-target", default=None,
                        help="text phrase in --prompt that annulus tokens positively bind to (e.g. "
                             "'green grass'). Required when --target-shape shrinks a source so the "
                             "annulus (rest of source silhouette) fills as background.")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="enable CUDA deterministic algorithms (torch.use_deterministic_algorithms=True)")
    parser.add_argument("--seg-device", default="cuda:0",
                        help="GPU for GroundingDINO + SAM (before pipe load)")
    parser.add_argument("--max-memory", default="0:22GiB,1:22GiB",
                        help="device_map='balanced' per-GPU cap")
    parser.add_argument("--save-mask-figure", action="store_true",
                        help="save side-by-side mask overlay figure")
    parser.add_argument("--mask-paths", default=None,
                        help="OPTIONAL: bypass GroundingDINO+SAM by loading masks from disk. "
                             "Semicolon-separated PNG paths aligned with --source-objects "
                             "(e.g., 'data/mask_dog_clean.png;data/mask_house.png').")
    args = parser.parse_args()

    sources = [s.strip() for s in args.source_objects.split(",") if s.strip()]
    targets = [t.strip() for t in args.target_phrases.split(";") if t.strip()]
    preserves = [p.strip() for p in args.preserve_objects.split(",") if p.strip()]
    if len(sources) != len(targets):
        raise ValueError(f"source-objects ({len(sources)}) must match target-phrases ({len(targets)})")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Prompt   : {args.prompt}")
    print(f"Sources  : {sources}")
    print(f"Targets  : {targets}")
    print(f"Preserves: {preserves}")

    import numpy as np
    from PIL import Image

    src_img = Image.open(input_path).convert("RGB")
    W, H = src_img.size
    print(f"Image    : {W}x{H}")

    # ── 1) Masks (disk or GroundingDINO+SAM) ────────────
    source_masks = {}
    if args.mask_paths:
        paths = [p.strip() for p in args.mask_paths.split(";") if p.strip()]
        if len(paths) != len(sources):
            raise ValueError(f"--mask-paths count ({len(paths)}) must match --source-objects ({len(sources)})")
        print("\n[1/4] Loading masks from disk (skipping GroundingDINO+SAM)")
        for s, p in zip(sources, paths):
            m = np.array(Image.open(p).convert("L")) > 128
            source_masks[s] = m
            print(f"  {s:>10s}: {m.sum():>7d} px ({m.sum()/(H*W)*100:5.1f}%)  <-  {p}")
        print("\n[2/4] Preserve masks: (skipped - using disk masks as-is)")
    else:
        print("\n[1/4] GroundingDINO + SAM masks")
        for s in sources:
            text = f"{s} . {s} shadow ."
            m, boxes = detect_and_segment(src_img, text, args.box_thr, args.txt_thr, args.dilation, args.seg_device)
            source_masks[s] = m
            print(f"  {s:>10s}: {m.sum():>7d} px ({m.sum()/(H*W)*100:5.1f}%)  boxes={len(boxes)}")

        # ── 2) Preserve subtraction ─────────────────────────
        if preserves:
            print("\n[2/4] Preserve masks (subtract from sources)")
            for p in preserves:
                pm, boxes = detect_and_segment(
                    src_img, f"{p} .",
                    args.box_thr + 0.10, args.txt_thr + 0.05,
                    args.preserve_dilation, args.seg_device,
                )
                print(f"  {p:>10s}: {pm.sum():>7d} px  boxes={len(boxes)}")
                if pm.sum() == 0:
                    continue
                pm = dilate_bool(pm, 4)  # small extra safety margin
                for s in sources:
                    before = source_masks[s].sum()
                    source_masks[s] = source_masks[s] & (~pm)
                    print(f"    - subtracted from {s}: {before} → {source_masks[s].sum()} px")
        else:
            print("\n[2/4] Preserve masks: (none)")

    # ── 3) Prompt spans + bias matrices ─────────────────
    print("\n[3/4] Tokenize prompt & build bias matrices")
    import torch
    from transformers import T5TokenizerFast
    from pom1.mras import mask_to_token_indices

    T_FULL = 512
    tokenizer = T5TokenizerFast.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", subfolder="tokenizer_2"
    )
    source_spans = {s: find_token_span(tokenizer, args.prompt, s, T_FULL) for s in sources}
    target_spans = {t: find_token_span(tokenizer, args.prompt, t, T_FULL) for t in targets}
    for s in sources:
        print(f"  source '{s}': tokens {source_spans[s]}")
    for t in targets:
        print(f"  target '{t}': tokens {target_spans[t]}")

    bg_span = []
    if args.background_target:
        bg_span = find_token_span(tokenizer, args.prompt, args.background_target, T_FULL)
        print(f"  background '{args.background_target}': tokens {bg_span}")

    GRID = W // 16
    N = GRID * GRID

    # ── Core / annulus split for target-aware mask reshaping ────────
    # Core   = target's natural footprint (small vertical rect if target is one animal,
    #          or full source mask if target has similar shape) → positive-bind to target text.
    # Annulus = full source mask minus core → suppress all object text → naturally fills as bg.
    from scipy.ndimage import binary_erosion
    source_core = {}
    source_annulus = {}

    # Parse per-source target shape: "dog:16x28;house:full"
    target_shape_map = {}
    if args.target_shape:
        for item in args.target_shape.split(";"):
            if ":" not in item: continue
            name, spec = item.strip().split(":", 1)
            target_shape_map[name.strip()] = spec.strip()

    for s in sources:
        full = source_masks[s]
        spec = target_shape_map.get(s, None)
        if spec is None or spec.lower() == "full":
            # No reshape: full mask is the core, no annulus
            source_core[s] = full
            source_annulus[s] = np.zeros_like(full)
            print(f"  core-shape '{s}': full mask ({full.sum()}px)")
        elif "x" in spec:
            # Rectangle: WxH in grid tokens (each = 16px)
            w_tok, h_tok = [int(x) for x in spec.lower().split("x")]
            w_px, h_px = w_tok * 16, h_tok * 16
            # Centroid of the source mask
            ys, xs = np.where(full)
            if len(xs) == 0:
                print(f"  warning: source '{s}' empty, skipping")
                source_core[s] = full
                source_annulus[s] = np.zeros_like(full)
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            # For foreground animals (dog→cat), anchor the BOTTOM at the source's bottom
            # so the cat's feet stay grounded where the dog's feet were.
            y_bottom = ys.max()
            y0 = max(0, y_bottom - h_px)
            y1 = min(H, y_bottom)
            x0 = max(0, cx - w_px // 2)
            x1 = min(W, cx + w_px // 2)
            core = np.zeros_like(full)
            core[y0:y1, x0:x1] = True
            # Intersect with source mask so we don't extend outside dog's footprint
            core = core & full
            if core.sum() == 0:
                # Fallback: use rectangle directly even if outside SAM silhouette
                core = np.zeros_like(full)
                core[y0:y1, x0:x1] = True
            source_core[s] = core
            source_annulus[s] = full & (~core)
            print(f"  core-shape '{s}': rect {w_tok}x{h_tok}tok at ({x0},{y0},{x1},{y1}) → "
                  f"core={core.sum()}px annulus={source_annulus[s].sum()}px")
        else:
            print(f"  warning: unknown target-shape spec '{spec}' for '{s}', using full")
            source_core[s] = full
            source_annulus[s] = np.zeros_like(full)

    # Mask → token index list, per source object (core and annulus separately)
    core_token_lists = {}
    annulus_token_lists = {}
    full_ref_sets = {}
    for s in sources:
        core_pil = Image.fromarray((source_core[s] * 255).astype(np.uint8), mode="L")
        core_token_lists[s] = mask_to_token_indices(core_pil, grid_size=GRID)
        ann_pil = Image.fromarray((source_annulus[s] * 255).astype(np.uint8), mode="L")
        annulus_token_lists[s] = mask_to_token_indices(ann_pil, grid_size=GRID)
        # ref set for image-side suppression uses FULL mask (block copy of entire silhouette)
        full_pil = Image.fromarray((source_masks[s] * 255).astype(np.uint8), mode="L")
        full_ref_sets[s] = set(mask_to_token_indices(full_pil, grid_size=GRID))
        print(f"  '{s}' tokens: core={len(core_token_lists[s])} annulus={len(annulus_token_lists[s])} full={len(full_ref_sets[s])}")

    # Union gen list with per-token region label = (source name, 'core'|'annulus')
    # Priority: core > annulus > other-source-core > other-source-annulus (first-seen wins)
    seen = {}   # gen_token_idx → (source_name, 'core'|'annulus')
    all_gen = []
    for s in sources:
        for g in core_token_lists[s]:
            if g not in seen:
                seen[g] = (s, "core")
                all_gen.append(g)
    for s in sources:
        for g in annulus_token_lists[s]:
            if g not in seen:
                seen[g] = (s, "annulus")
                all_gen.append(g)
    n_gen = len(all_gen)
    bg_idx = [i for i in range(N) if i not in seen]
    print(f"  gen union: {n_gen}, background: {len(bg_idx)}")

    text_bias = torch.zeros(n_gen, T_FULL, dtype=torch.float32)
    ref_bias = torch.zeros(n_gen, N, dtype=torch.float32)

    # Map source name → its target phrase (parallel lists)
    own_target = {s: targets[j] for j, s in enumerate(sources)}

    for i, g in enumerate(all_gen):
        own_src, kind = seen[g]

        # (image side) block copy of own source appearance
        for r in full_ref_sets[own_src]:
            ref_bias[i, r] = args.suppression_bias

        # (text side) suppress OTHER sources' source words + destination phrases only.
        # Own source and own target left at 0 (natural attention).
        for j, s in enumerate(sources):
            if s == own_src:
                continue
            for t in source_spans[s]:
                text_bias[i, t] = args.suppression_bias
            for t in target_spans[targets[j]]:
                text_bias[i, t] = args.suppression_bias

        if kind == "core" and args.target_bias != 0:
            # positively boost own target text (only when explicitly enabled)
            for t in target_spans[own_target[own_src]]:
                text_bias[i, t] = args.target_bias
        elif kind == "annulus":
            # annulus: suppress own target too, then positively bind to background phrase
            for t in target_spans[own_target[own_src]]:
                text_bias[i, t] = args.suppression_bias
            if bg_span and args.target_bias != 0:
                for t in bg_span:
                    text_bias[i, t] = args.target_bias

    print(f"  text_bias nonzeros: {(text_bias != 0).sum().item()}")
    print(f"  ref_bias  nonzeros: {(ref_bias != 0).sum().item()}")

    # ── 4) Load pipeline + run ─────────────────────────
    print("\n[4/4] FLUX.1-Kontext-dev  device_map=balanced")
    # Enable deterministic algorithms so same seed → same result even across CUDA runs.
    if args.deterministic:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"  warning: torch.use_deterministic_algorithms failed: {e}")
        print("  deterministic: True (cudnn.deterministic=True, use_deterministic_algorithms(warn_only=True))")
    from diffusers import FluxKontextPipeline
    from regional_processor import RegionalMRASProcessor, install

    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=parse_max_memory(args.max_memory),
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    src_packed = encode_and_pack(pipe, src_img)

    proc = RegionalMRASProcessor(
        gen_indices=all_gen,
        text_bias=text_bias,
        ref_bias=ref_bias,
        apply_until_step=args.apply_until_step,
        N_total=N,
    )
    install(pipe, proc)

    bg_tensor = torch.tensor(bg_idx, dtype=torch.long)
    callback = make_repaint_callback(
        proc, src_packed, bg_tensor,
        repaint_until_step=args.repaint_until_step,
        feather_ramp=args.repaint_feather,
    )

    print(f"  running {args.steps} steps  gs={args.guidance_scale}  seed={args.seed}")
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
        save_mask_figure(src_img, result, source_masks, targets, output_path)


if __name__ == "__main__":
    main()
