"""RTAR: Regional Text-Vision Attention Routing for FLUX.1-Kontext.

Given user-provided per-object masks + a text prompt describing multi-object
edits, this pipeline routes attention per-region so each source area binds to
its own destination text span.

Contribution highlights (versus prior work):
  - Extends single-object editing (KV-Edit, ConsistEdit) to multi-object
    simultaneous editing in FLUX Kontext.
  - Introduces per-region bias on the joint self-attention scores of the
    text | gen | ref sequence: gen tokens in each source region negatively
    suppress the OTHER object's source words + destination phrases, and can
    optionally receive a positive bias toward their own destination.
  - Blocks the gen→ref pathway for same-position identity copying via ref_bias
    so the source object's appearance cannot leak through joint attention.
  - Anchors non-edited background via RePaint on the flow-matching latent.

Masks are assumed to come from an external segmentation pipeline (GroundingDINO
+ SAM, user annotation, or dataset ground truth). This script does NOT perform
detection itself.

Identity-preserve mode: an object marked "preserve" in --modes keeps its SOURCE
appearance while the prompt may move it / change its pose. Mechanism: gen rows
of the destination region (--dest-map) get a finite positive bias toward the
preserved object's ref tokens (cross-region gen→ref appearance pathway), gated
to late steps (--identity-from-step, layout forms early / appearance late) and
to the FreeFlux content-similarity layers (--identity-layers content).

Usage (replace):
    python rtar_pipeline.py \\
        --input data/dog.png \\
        --mask-paths "data/masks/mask_dog_clean.png;data/masks/mask_house.png" \\
        --source-objects "dog,house" \\
        --target-phrases "cat;cat tower" \\
        --prompt "Replace the dog with a cat and the house with a cat tower." \\
        --output outputs/rtar_baseline.png \\
        --save-mask-figure

Usage (preserve: move the dog into the house region, keep its identity):
    python rtar_pipeline.py \\
        --input data/dog.png \\
        --mask-paths "data/masks/mask_dog_clean.png;data/masks/mask_house.png" \\
        --source-objects "dog,house" \\
        --modes "preserve,replace" \\
        --dest-map "dog:house" \\
        --target-phrases "-;toilet" \\
        --prompt "Put the dog into the toilet." \\
        --identity-bias 2.0 --identity-from-step 8 --identity-layers content \\
        --output outputs/rtar/dog_toilet_preserve.png \\
        --save-mask-figure
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <repo>/src
MRAS_SRC = HERE.parent.parent / "MRAS" / "src"  # sibling MRAS checkout
sys.path.insert(0, str(MRAS_SRC))
sys.path.insert(0, str(HERE))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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


def build_repaint_weight(edit_tokens, N, grid, dilate: int = 0, spatial_feather: int = 0):
    """Per-token RePaint anchor weight w in [0,1]  (0 = fully generative,
    1 = fully anchored to source).

    Binary behavior (w=1 outside the edit union) clips anything the model
    draws past a mask boundary — objects get eaten by the restored source.
    `dilate` extends the generative zone D tokens past the union;
    `spatial_feather` linearly ramps w 0→1 over the next F tokens, so
    boundary content blends instead of being cut.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    edit = np.zeros((grid, grid), dtype=bool)
    for g in edit_tokens:
        edit[g // grid, g % grid] = True
    # Euclidean distance (in tokens) from each cell to the nearest edit cell
    dist = distance_transform_edt(~edit)
    if spatial_feather > 0:
        w = np.clip((dist - dilate) / spatial_feather, 0.0, 1.0)
    else:
        w = (dist > dilate).astype(np.float64)
    w[edit] = 0.0
    import torch
    return torch.tensor(w.reshape(-1), dtype=torch.float32)


def make_repaint_callback(proc, src_packed, weight,
                          repaint_until_step: int, feather_ramp: int = 0,
                          latent_gate=None):
    """Anchor background latents to the (renoised) source at every step
    until `repaint_until_step`, blended per token by `weight` [N] (see
    build_repaint_weight), with an optional linear temporal ramp-off.

    latent_gate: optional (tok_idx, row_idx) covering ALL non-source-mask
    tokens in full-canvas soft-background mode. Each is latent-anchored to
    the source by (1 - editedness): true background (low editedness) is
    pixel-preserved instead of re-rendered (which over-sharpens it), while
    the object's new position (high editedness) still regenerates. Source-
    mask tokens are excluded (anchoring them would paste the object back)."""
    import torch

    def callback(pipeline, step_idx, timestep, kwargs):
        step_end = getattr(proc, "step_end", None)
        if step_end is not None:
            step_end()  # fold this step's silhouette harvest into the dyn gain
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

        w_full = weight
        gates = [g for g in (latent_gate,) if g is not None]
        if gates and hasattr(proc, "row_editedness"):
            e = proc.row_editedness()
            w_full = weight.clone()
            for gtok, grow in gates:
                w_full[gtok] = torch.maximum(
                    w_full[gtok], (1.0 - e[grow]).clamp(0.0, 1.0))

        sched = pipeline.scheduler
        if step_idx + 1 < len(sched.timesteps):
            next_sigma = float(sched.timesteps[step_idx + 1]) / 1000.0
        else:
            next_sigma = 0.0
        slp = src_packed.to(device, dtype)
        noise = torch.randn_like(slp)
        noised = (1.0 - next_sigma) * slp + next_sigma * noise
        w = (alpha * w_full.to(device)).to(dtype)[None, :, None]
        expanded = noised.expand(latents.shape[0], -1, -1)
        latents = w * expanded + (1 - w) * latents
        kwargs["latents"] = latents
        return kwargs

    return callback


def find_token_span(tokenizer, prompt, target, T_full=512, required=True):
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
    if required:
        raise ValueError(f"could not locate token span for '{target}' in prompt")
    return []


def save_mask_figure(src, out, source_masks, target_phrases, output_path,
                     label="Output"):
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
    d.text((pad + 10, pad), f"Source + user masks   ({legend})", fill=(20, 20, 20), font=f)
    d.text((pad * 2 + W + 10, pad), f"{label}", fill=(20, 20, 20), font=f)
    fig_path = Path(output_path).with_name(Path(output_path).stem + "_masks.png")
    canvas.save(fig_path)
    print(f"[figure] saved: {fig_path}")


# ─────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RTAR: Regional Text-Vision Attention Routing for FLUX.1-Kontext",
    )
    parser.add_argument("--input", required=True, help="source image")
    parser.add_argument("--output", required=True, help="output image path")
    parser.add_argument("--mask-paths", required=True,
                        help="semicolon-separated PNG paths, aligned with --source-objects")
    parser.add_argument("--source-objects", required=True,
                        help="comma-separated source names (must appear in --prompt)")
    parser.add_argument("--target-phrases", required=True,
                        help="semicolon-separated destination phrases (aligned with sources). "
                             "Use '-' for a preserve-mode object (no fill phrase; the vacated "
                             "source area is filled from prompt context).")
    parser.add_argument("--modes", default=None,
                        help="comma-separated per-object modes aligned with --source-objects: "
                             "'replace' (default) or 'preserve' (keep the object's source "
                             "identity while it moves per prompt)")
    parser.add_argument("--dest-map", default=None,
                        help="semicolon-separated obj:dest entries for preserve objects, e.g. "
                             "'dog:house'. dest = another source name OR a mask PNG path. "
                             "A preserve object without an entry defaults to its own region "
                             "(in-place identity keep).")
    parser.add_argument("--identity-bias", type=float, default=2.0,
                        help="positive attention bias from destination-region gen rows to the "
                             "preserved object's ref tokens. ln(3)~ln(8) = 1.1~2.1 flips the "
                             "measured editability gap; >=5 risks softmax saturation "
                             "(rigid copy-paste, prompt adherence dies)")
    parser.add_argument("--identity-from-step", type=int, default=8,
                        help="first step of the identity boost window (let the prompt set "
                             "layout/pose in earlier steps)")
    parser.add_argument("--identity-until-step", type=int, default=28,
                        help="last step of the identity boost window")
    parser.add_argument("--identity-layers", default="content",
                        choices=["all", "position", "content", "not-content", "not-position"],
                        help="FLUX layer subset for the identity boost (default: FreeFlux "
                             "content-similarity layers, leaving position layers prompt-driven)")
    parser.add_argument("--preserve-vacate", action="store_true",
                        help="keep the same-position gen→ref block on a preserve object's "
                             "own rows even when dest==source: the object cannot copy "
                             "itself in place and must re-form (move/re-pose) via the "
                             "cross-position identity bias. Use for action prompts.")
    parser.add_argument("--identity-dynamic", action="store_true",
                        help="Stage 3a: gain-scale the identity bias per dest row by the "
                             "row's attention mass to the preserved object's text span "
                             "(EMA over steps) — confines appearance to the emerging "
                             "silhouette and stops leak into the rest of the dest region")
    parser.add_argument("--identity-dyn-ema", type=float, default=0.7,
                        help="EMA decay of the per-row silhouette gain across steps")
    parser.add_argument("--identity-dyn-gamma", type=float, default=1.0,
                        help="sharpening exponent on the normalized gain")
    parser.add_argument("--identity-dyn-floor", type=float, default=0.0,
                        help="minimum gain so non-silhouette dest rows keep a trickle")
    parser.add_argument("--identity-topk", type=int, default=0,
                        help="correspondence-sharpened identity transport: boost each "
                             "gen row's own top-K matches among the object's ref columns "
                             "(the model's emergent gen->ref correspondence) instead of "
                             "a uniform region bias (which only transports the mean/low-"
                             "frequency appearance). 0 = uniform (old behavior).")
    parser.add_argument("--identity-ot", action="store_true",
                        help="entropic optimal-transport identity delivery: Sinkhorn plan "
                             "between the object's gen rows and ref columns (row sums=1: "
                             "no averaging; column mass conservation: EVERY source part "
                             "including garment-free regions must transport), injected in "
                             "VALUE space (no softmax competition with text)")
    parser.add_argument("--ot-tau", type=float, default=0.5,
                        help="OT entropic regularization (smaller = sharper plan)")
    parser.add_argument("--ot-lambda", type=float, default=0.6,
                        help="value-injection blend weight at gain=1")
    parser.add_argument("--ot-iters", type=int, default=5,
                        help="Sinkhorn iterations per layer call")
    parser.add_argument("--capture-entropy", action="store_true",
                        help="record within-object gen->ref attention stats (normalized "
                             "entropy, object mass, top-8 share) on identity layers and "
                             "print a summary — motivation measurement for the "
                             "correspondence-sharpening claim")
    parser.add_argument("--target-dyn-bias", type=float, default=0.0,
                        help="dynamic positive text binding for replace targets: each "
                             "region row's own-target bias is gain-scaled by its "
                             "attention silhouette (object forms its own shape instead "
                             "of filling the mask)")
    parser.add_argument("--soft-background", action="store_true",
                        help="mask-FREE background preservation mode: full-image canvas; "
                             "unclaimed rows (low editedness) are pixel-preserved via the "
                             "latent gate and stop hearing the prompt via text "
                             "suppression. Masks only mark what disappears. Implies "
                             "--identity-dynamic.")
    parser.add_argument("--soft-background-anchor", type=float, default=0.0,
                        help="ablation arm: additional attention bias beta*(1-editedness) "
                             "toward each row's own-position ref token. Redundant with "
                             "the latent gate in all validated cases (default off).")
    parser.add_argument("--soft-background-latent", action="store_true",
                        help="in soft-background mode, also latent-anchor (RePaint) the "
                             "non-source-mask tokens by (1-editedness): true background is "
                             "pixel-preserved instead of re-rendered/over-sharpened, only "
                             "the object's new position regenerates. Mask-free (editedness "
                             "decides the region).")
    parser.add_argument("--soft-background-text", type=float, default=4.0,
                        help="with --soft-background: unclaimed rows also get "
                             "-beta*(1-editedness) on all text columns, so the "
                             "background stops listening to the prompt (RePaint's "
                             "second, implicit role). 0 disables.")
    parser.add_argument("--soft-background-halo", type=int, default=0,
                        help="grid-dilate the editedness map this many tokens before "
                             "computing the anchor, leaving a growth band around forming "
                             "objects (0 = sharp per-row anchor; objects stay "
                             "nucleus-sized)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--suppression-bias", type=float, default=-1e4,
                        help="negative attention bias for other objects' text tokens")
    parser.add_argument("--suppress-own-source", action="store_true",
                        help="for replace regions, suppress the region's own source-word "
                             "tokens as well. Useful for instruction prompts that contain "
                             "both source and target words, e.g. dog→cat.")
    parser.add_argument("--apply-until-step", type=int, default=28,
                        help="step at which the regional attention manipulation stops")
    parser.add_argument("--repaint-until-step", type=int, default=28,
                        help="step at which RePaint background anchoring stops")
    parser.add_argument("--repaint-feather", type=int, default=0,
                        help="linear ramp length (in steps) before RePaint stops")
    parser.add_argument("--repaint-dilate", type=int, default=0,
                        help="extend the generative (un-anchored) zone this many tokens "
                             "past the mask union, so boundary-crossing content "
                             "(tails, object bases) is not clipped by the source")
    parser.add_argument("--repaint-spatial-feather", type=int, default=0,
                        help="linearly ramp the RePaint anchor 0→1 over this many tokens "
                             "beyond the dilated zone (soft spatial boundary)")
    parser.add_argument("--max-memory", default="0:22GiB,1:22GiB")
    parser.add_argument("--save-mask-figure", action="store_true")
    parser.add_argument("--no-rtar", action="store_true",
                        help="ablation: skip regional attention manipulation, keep only RePaint. "
                             "Serves as 'Kontext + masking only' baseline.")
    parser.add_argument("--layers", default="all",
                        choices=["all", "position", "content", "not-content", "not-position", "custom"],
                        help="which FLUX layers to apply RTAR on. "
                             "'position' = FreeFlux position-dependent layers "
                             "[1,2,4,26,30,54,55]. "
                             "'content' = FreeFlux content-similarity layers. "
                             "'not-content' = all EXCEPT content-similarity "
                             "(preserves identity, edits everything else). "
                             "'not-position' = all EXCEPT position-dependent. "
                             "'custom' = use --layer-ids.")
    parser.add_argument("--layer-ids", default=None,
                        help="comma-separated global layer indices (0..56) when --layers custom")
    args = parser.parse_args()
    full_canvas = args.soft_background
    if full_canvas:
        args.identity_dynamic = True  # editedness needs the harvest machinery
    if full_canvas and args.no_rtar:
        raise ValueError("--soft-background needs the regional processor "
                         "(incompatible with --no-rtar)")

    sources = [s.strip() for s in args.source_objects.split(",") if s.strip()]
    targets = [t.strip() for t in args.target_phrases.split(";") if t.strip()]
    mask_paths = [p.strip() for p in args.mask_paths.split(";") if p.strip()]
    if not (len(sources) == len(targets) == len(mask_paths)):
        raise ValueError(f"lengths must match: sources={len(sources)}, "
                         f"targets={len(targets)}, masks={len(mask_paths)}")

    # ── Per-object modes + destination map ──────
    if args.modes is None:
        modes = ["replace"] * len(sources)
    else:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        if len(modes) != len(sources):
            raise ValueError(f"--modes length {len(modes)} != sources {len(sources)}")
        bad = [m for m in modes if m not in ("replace", "preserve")]
        if bad:
            raise ValueError(f"invalid modes {bad}: must be 'replace' or 'preserve'")
    preserve_objs = [s for s, m in zip(sources, modes) if m == "preserve"]

    dest_map = {}
    if args.dest_map:
        for entry in args.dest_map.split(";"):
            if not entry.strip():
                continue
            obj, dest = entry.split(":", 1)
            obj, dest = obj.strip(), dest.strip()
            if obj not in preserve_objs:
                raise ValueError(f"--dest-map entry '{obj}' is not a preserve object")
            dest_map[obj] = dest
    for obj in preserve_objs:
        dest_map.setdefault(obj, obj)  # in-place identity keep by default

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Prompt : {args.prompt}")
    for s, t, mp, m in zip(sources, targets, mask_paths, modes):
        extra = f"   [preserve → dest '{dest_map[s]}']" if m == "preserve" else ""
        print(f"  {s!r:>12s}  →  {t!r:>14s}   mask: {mp}{extra}")

    import numpy as np
    import torch
    # the manual attention path does fp32 matmuls — let them use tensor cores
    # (TF32; ~1e-3 relative precision, invisible at attention-score scale)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from diffusers import FluxKontextPipeline
    from PIL import Image
    from transformers import T5TokenizerFast
    from pom1.mras import mask_to_token_indices
    from regional_processor import RegionalMRASProcessor, install

    src_img = Image.open(args.input).convert("RGB")
    W, H = src_img.size
    GRID = W // 16
    N = GRID * GRID
    T_FULL = 512
    print(f"Image  : {W}x{H}, grid_size={GRID}")

    # ── Load masks from disk ────────────────────
    print("\n[1/3] Load user-provided masks")
    source_masks_bool = {}
    for s, mp in zip(sources, mask_paths):
        m = np.array(Image.open(mp).convert("L")) > 128
        source_masks_bool[s] = m
        print(f"  '{s}': {m.sum()} px ({m.sum()/(H*W)*100:.1f}%)   <-  {mp}")

    # ── Tokenize prompt + find text spans ───────
    print("\n[2/3] Tokenize prompt & build regional attention biases")
    tokenizer = T5TokenizerFast.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", subfolder="tokenizer_2"
    )
    # Source spans are for suppression only — a source name absent from the
    # prompt simply has nothing to suppress (empty span + warning).
    source_spans = {s: find_token_span(tokenizer, args.prompt, s, T_FULL, required=False)
                    for s in sources}
    target_spans = {t: find_token_span(tokenizer, args.prompt, t, T_FULL)
                    for t in targets if t != "-"}
    for s in sources:
        if not source_spans[s]:
            print(f"  source '{s}': NOT in prompt (no text suppression for it)")
            continue
        print(f"  source '{s}': tokens {source_spans[s]}")
    for t in targets:
        if t != "-":
            print(f"  target '{t}': tokens {target_spans[t]}")

    # ── Mask → token indices ────────────────────
    source_token_lists = {}
    for s in sources:
        m_pil = Image.fromarray((source_masks_bool[s] * 255).astype(np.uint8), mode="L")
        source_token_lists[s] = mask_to_token_indices(m_pil, grid_size=GRID)
        print(f"  '{s}' mask tokens: {len(source_token_lists[s])} / {N}")

    # ── Resolve preserve destinations → token lists ──
    dest_token_lists = {}
    for obj, dest in dest_map.items():
        if dest in source_token_lists:
            dest_token_lists[obj] = source_token_lists[dest]
        else:
            m_pil = Image.open(dest).convert("L")
            dest_token_lists[obj] = mask_to_token_indices(m_pil, grid_size=GRID)
        print(f"  preserve '{obj}' → dest '{dest}': {len(dest_token_lists[obj])} tokens")

    # Union of gen indices, with per-token region label = source name
    seen = {}
    all_gen = []
    for s in sources:
        for g in source_token_lists[s]:
            if g not in seen:
                seen[g] = s
                all_gen.append(g)
    # Destination-only tokens (dest mask outside all source masks) join the gen
    # union with no region label: no suppression, identity boost only. They must
    # also leave bg_idx, else RePaint would erase the moved object every step.
    for obj in preserve_objs:
        for g in dest_token_lists[obj]:
            if g not in seen:
                seen[g] = None
                all_gen.append(g)

    # Full canvas (--soft-background / --two-pass): every token joins the gen
    # union so objects can form anywhere; unlabeled rows carry no suppression.
    if full_canvas:
        for g in range(N):
            if g not in seen:
                seen[g] = None
                all_gen.append(g)
        print(f"  full canvas: all {N} tokens in gen union")

    n_gen = len(all_gen)
    bg_idx = [i for i in range(N) if i not in seen]
    print(f"  gen union: {n_gen}, background: {len(bg_idx)}")

    # ── Build regional bias matrices ────────────
    # RTAR (Regional Text-vision Attention Routing) construction:
    #   For gen token i belonging to source object own_src:
    #     * ref_bias  : suppress attention to ref tokens at own_src's mask
    #                   (kills gen→ref identity-copy pathway)
    #     * text_bias : suppress OTHER objects' source words and destination phrases
    #     * (optional) positive bias on own destination phrase
    text_bias = torch.zeros(n_gen, T_FULL, dtype=torch.float32)
    ref_bias = torch.zeros(n_gen, N, dtype=torch.float32)
    ref_sets = {s: set(source_token_lists[s]) for s in sources}
    own_target = {s: targets[j] for j, s in enumerate(sources)}
    mode_by_source = {s: modes[j] for j, s in enumerate(sources)}

    for i, g in enumerate(all_gen):
        own_src = seen[g]
        if own_src is None:
            continue  # dest-only token: no suppression (identity boost only)
        # gen→ref suppression for own object (block identity copy)
        for r in ref_sets[own_src]:
            ref_bias[i, r] = args.suppression_bias
        if args.suppress_own_source and mode_by_source[own_src] == "replace":
            for t in source_spans[own_src]:
                text_bias[i, t] = args.suppression_bias
        # gen→text suppression for OTHER objects
        for j, s in enumerate(sources):
            if s == own_src:
                continue
            # full canvas: a preserve object may land anywhere — never
            # suppress its word on other regions' rows
            if not (full_canvas and modes[j] == "preserve"):
                for t in source_spans[s]:
                    text_bias[i, t] = args.suppression_bias
            if targets[j] != "-":
                for t in target_spans[targets[j]]:
                    text_bias[i, t] = args.suppression_bias
    # ── Identity-preserve bias (cross-region gen→ref appearance pathway) ──
    # For each preserve object obj with destination dest:
    #   * dest rows get +identity_bias toward obj's ref tokens
    #   * dest rows release the suppression that fights the preserve goal:
    #     obj's source-word text suppression, and any block on obj's ref
    #     columns (relevant when dest == obj, i.e. in-place preserve)
    row_of = {g: i for i, g in enumerate(all_gen)}

    def prior_gain(prior_tokens, lo=0.25):
        """Full-canvas init gain: 1.0 inside the object's mask prior, lo
        elsewhere — the mask is a soft position hint, not a boundary."""
        pset = set(prior_tokens)
        return [1.0 if g in pset else lo for g in all_gen]

    identity_ref_bias = None
    dyn_setup = []  # (obj, rows, ref_cols, span, init_gain)
    if preserve_objs:
        if not args.identity_dynamic:
            identity_ref_bias = torch.zeros(n_gen, N, dtype=torch.float32)
        for obj in preserve_objs:
            obj_ref = sorted(ref_sets[obj])
            obj_span = source_spans[obj]
            if args.identity_dynamic and not obj_span:
                raise ValueError(f"--identity-dynamic needs '{obj}' in the prompt "
                                 f"(its text span drives the silhouette gain)")
            dest_rows = []
            for g in dest_token_lists[obj]:
                i = row_of[g]
                dest_rows.append(i)
                if identity_ref_bias is not None:
                    identity_ref_bias[i, obj_ref] = args.identity_bias
                text_bias[i, obj_span] = 0.0
                if not args.preserve_vacate:
                    ref_bias[i, obj_ref] = 0.0
            if args.identity_dynamic:
                if full_canvas:
                    if args.preserve_vacate:
                        # the mask rows are ref-BLOCKED under vacate — a prior
                        # peaked there is useless and loses everywhere else to
                        # other specs. Start uniform; the harvest relocates it.
                        g0 = [0.6] * n_gen
                    else:
                        # object may form anywhere; its masks are soft priors
                        prior = set(source_token_lists[obj]) | set(dest_token_lists[obj])
                        g0 = prior_gain(prior)
                    dyn_setup.append((obj, list(range(n_gen)), obj_ref, obj_span, g0))
                else:
                    dyn_setup.append((obj, dest_rows, obj_ref,
                                      obj_span, None))
        if identity_ref_bias is not None:
            print(f"  identity_ref_bias nonzeros: {(identity_ref_bias != 0).sum().item()}"
                  f"  (+{args.identity_bias}, steps [{args.identity_from_step},"
                  f" {args.identity_until_step}], layers '{args.identity_layers}')")
        else:
            for obj, rows, cols, span, _g0 in dyn_setup:
                print(f"  identity (dynamic) '{obj}': {len(rows)} rows × "
                      f"{len(cols)} ref cols, beta={args.identity_bias}, "
                      f"ema={args.identity_dyn_ema}, gamma={args.identity_dyn_gamma}, "
                      f"floor={args.identity_dyn_floor}")

    # Dynamic target specs for replace objects: gain-scaled binding to the own
    # target span (object forms its own silhouette instead of filling the mask);
    # beta=0 still harvests, providing editedness for canvas re-anchoring.
    dyn_target_setup = []  # (obj, rows, target_span, init_gain)
    if args.target_dyn_bias > 0 or full_canvas:
        for j, s in enumerate(sources):
            if modes[j] == "preserve" or targets[j] == "-":
                continue
            span = target_spans[targets[j]]
            if full_canvas:
                rows = list(range(n_gen))
                g0 = prior_gain(source_token_lists[s])
            else:
                rows = [i for i, g in enumerate(all_gen) if seen[g] == s]
                g0 = None
            dyn_target_setup.append((s, rows, span, g0))
            print(f"  target (dynamic) '{s}'→'{targets[j]}': {len(rows)} rows, "
                  f"beta={args.target_dyn_bias}")

    print(f"  text_bias nonzeros: {(text_bias != 0).sum().item()}")
    print(f"  ref_bias  nonzeros: {(ref_bias != 0).sum().item()}")

    # ── Load pipeline + install regional processor ────
    print("\n[3/3] FLUX.1-Kontext-dev  device_map=balanced")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=parse_max_memory(args.max_memory),
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    src_packed = encode_and_pack(pipe, src_img)

    if args.no_rtar:
        # Ablation baseline: no regional attention manipulation. Only RePaint.
        print("  [ablation] --no-rtar: regional attention manipulation DISABLED")
        # Dummy proc that just tracks current_step for the callback
        class _StepTracker:
            def __init__(self):
                self.current_step = 0
        proc = _StepTracker()
    else:
        proc = RegionalMRASProcessor(
            gen_indices=all_gen,
            text_bias=text_bias,
            ref_bias=ref_bias,
            apply_until_step=args.apply_until_step,
            N_total=N,
            identity_ref_bias=identity_ref_bias,
            identity_from_step=args.identity_from_step,
            identity_until_step=args.identity_until_step,
        )
        dyn_kw = dict(ema=args.identity_dyn_ema,
                      gamma=args.identity_dyn_gamma,
                      floor=args.identity_dyn_floor)
        ot_cfg = (dict(tau=args.ot_tau, lam=args.ot_lambda, iters=args.ot_iters)
                  if args.identity_ot else None)
        for obj, rows, cols, span, g0 in dyn_setup:
            proc.add_dynamic(rows, span, cols, args.identity_bias,
                             in_ref=True, init_gain=g0,
                             topk=args.identity_topk, ot=ot_cfg, **dyn_kw)
        if args.identity_topk > 0:
            print(f"  identity transport: correspondence-sharpened top-{args.identity_topk}")
        if ot_cfg:
            print(f"  identity transport: entropic OT (tau={args.ot_tau}, "
                  f"lambda={args.ot_lambda}, iters={args.ot_iters})")
        proc.capture_entropy = args.capture_entropy
        for obj, rows, span, g0 in dyn_target_setup:
            proc.add_dynamic(rows, span, span, args.target_dyn_bias,
                             in_ref=False, init_gain=g0, **dyn_kw)
        if args.soft_background:
            bg_rows = [i for i, g in enumerate(all_gen) if seen[g] is None]
            bg_toks = [all_gen[i] for i in bg_rows]
            proc.set_soft_background(bg_rows, bg_toks, args.soft_background_anchor,
                                     text_suppress=args.soft_background_text,
                                     halo=args.soft_background_halo)
            print(f"  soft background: {len(bg_rows)} rows, "
                  f"anchor={args.soft_background_anchor}, "
                  f"text_suppress={args.soft_background_text}, "
                  f"halo={args.soft_background_halo}")
        # FreeFlux layer role sets (from arXiv 2503.16153, discovered via RoPE probing)
        FREEFLUX_POSITION = [1, 2, 4, 26, 30, 54, 55]
        FREEFLUX_CONTENT = [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]
        ALL_LAYERS = list(range(57))

        def resolve_layers(name):
            if name == "all":
                return None
            if name == "position":
                return FREEFLUX_POSITION
            if name == "content":
                return FREEFLUX_CONTENT
            if name == "not-content":
                return [i for i in ALL_LAYERS if i not in FREEFLUX_CONTENT]
            if name == "not-position":
                return [i for i in ALL_LAYERS if i not in FREEFLUX_POSITION]
            if name == "custom":
                return [int(x) for x in args.layer_ids.split(",")]
            raise ValueError(f"unknown layer set '{name}'")

        install(pipe, proc,
                layer_filter=resolve_layers(args.layers),
                identity_layer_filter=resolve_layers(args.identity_layers))

    repaint_w = build_repaint_weight(
        list(seen.keys()), N, GRID,
        dilate=args.repaint_dilate,
        spatial_feather=args.repaint_spatial_feather,
    )
    n_full = int((repaint_w >= 1.0).sum())
    n_soft = int(((repaint_w > 0) & (repaint_w < 1.0)).sum())
    print(f"  repaint weight: {n_full} anchored, {n_soft} feathered, "
          f"{N - n_full - n_soft} generative "
          f"(dilate={args.repaint_dilate}, spatial_feather={args.repaint_spatial_feather})")
    # editedness-gated latent preservation of the true background (mask-free):
    # every non-source-mask token, anchored to source by (1-editedness)
    latent_gate = None
    if args.soft_background_latent:
        gate_rows = [i for i, g in enumerate(all_gen) if seen[g] is None]
        gate_toks = [all_gen[i] for i in gate_rows]
        latent_gate = (torch.tensor(gate_toks, dtype=torch.long),
                       torch.tensor(gate_rows, dtype=torch.long))
        print(f"  soft-background latent preservation: {len(gate_rows)} tokens "
              f"editedness-gated")

    def run_pass(weight, tag):
        callback = make_repaint_callback(
            proc, src_packed, weight,
            repaint_until_step=args.repaint_until_step,
            feather_ramp=args.repaint_feather,
            latent_gate=latent_gate,
        )
        print(f"  [{tag}] {args.steps} steps  gs={args.guidance_scale}  "
              f"seed={args.seed}  apply_until_step={args.apply_until_step}  "
              f"repaint_until_step={args.repaint_until_step}")
        return pipe(
            image=src_img,
            prompt=args.prompt,
            height=H,
            width=W,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            callback_on_step_end=callback,
        ).images[0]

    result = run_pass(repaint_w, "single pass")

    result.save(output_path)
    print(f"[output] saved: {output_path}")

    if args.capture_entropy and hasattr(proc, "entropy_summary"):
        for s in proc.entropy_summary():
            print(f"[entropy] spec{s['spec']} topk={s['topk']} beta={s['beta']}: "
                  f"within-obj ent_norm={s['ent_norm']:.3f} "
                  f"(early {s['ent_early']:.3f} → late {s['ent_late']:.3f}), "
                  f"obj_mass={s['obj_mass']:.4f}, top8_share={s['top8_share']:.3f}, "
                  f"hi-gain rows={s['n_hi_rows']:.0f}, "
                  f"gain mean={s['gain_mean']:.3f} max={s['gain_max']:.2f}")

    if args.save_mask_figure:
        save_mask_figure(src_img, result, source_masks_bool, targets, output_path,
                         label="RTAR output")


if __name__ == "__main__":
    main()
