"""Identity transport under large motion, on FLUX.1-Kontext.

Move a given real object to the pose/position the prompt demands while
preserving its instance identity (unique patterns, markers, garment look).
Training-free, inversion-free, single pass. The rough mask is a soft prior.

Mechanism — one editedness-gated bias field on the joint [text|gen|ref]
attention, three roles:
  Transport : rows being edited boost their own top-k emergent gen->ref
              matches (+beta*e) — carries high-frequency instance detail to
              the object's NEW location.
  Preserve  : untouched rows are pixel-preserved via an editedness-gated
              latent copy and stop hearing the prompt (text suppression),
              both scaled by (1 - e).
  Vacate    : the object's source rows block same-position copying, so the
              object must re-form where the prompt says (--no-vacate ablates).

Prompt rule: describe preserved attributes with NEUTRAL words ("yellow
outfit", not "raincoat") — wrong structural priors override the reference.

Usage:
    python src/pipeline.py \
        --input data/sources/corgi_raincoat.png \
        --mask data/masks/mask_corgi_rough.png \
        --object corgi \
        --prompt "The corgi wearing a yellow outfit with the number 5 is jumping high in the air to catch a red frisbee." \
        --output outputs/corgi_jump.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # <repo>/src
MRAS_SRC = HERE.parent.parent / "MRAS" / "src"       # sibling MRAS checkout
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


def make_callback(proc, src_packed, latent_gate, until_step: int, feather: int):
    """Per-step: fold the editedness harvest, then pixel-preserve every
    unclaimed non-object token by (1 - editedness) latent copy."""
    import torch

    def callback(pipeline, step_idx, timestep, kwargs):
        step_end = getattr(proc, "step_end", None)
        if step_end is not None:
            step_end()
        proc.current_step = step_idx
        latents = kwargs["latents"]
        device, dtype = latents.device, latents.dtype

        if latent_gate is None or step_idx >= until_step:
            return kwargs
        ramp_start = until_step - feather
        if step_idx < ramp_start or feather == 0:
            alpha = 1.0
        else:
            alpha = max(0.0, 1.0 - (step_idx - ramp_start + 1) / (feather + 1))

        gtok, grow = latent_gate
        e = proc.row_editedness()
        w = torch.zeros(src_packed.shape[1])
        w[gtok] = (1.0 - e[grow]).clamp(0.0, 1.0)

        sched = pipeline.scheduler
        if step_idx + 1 < len(sched.timesteps):
            next_sigma = float(sched.timesteps[step_idx + 1]) / 1000.0
        else:
            next_sigma = 0.0
        slp = src_packed.to(device, dtype)
        noise = torch.randn_like(slp)
        noised = (1.0 - next_sigma) * slp + next_sigma * noise
        wd = (alpha * w.to(device)).to(dtype)[None, :, None]
        latents = wd * noised.expand(latents.shape[0], -1, -1) + (1 - wd) * latents
        kwargs["latents"] = latents
        return kwargs

    return callback


def find_token_span(tokenizer, prompt, target, T_full=512):
    full = tokenizer(prompt, return_tensors="pt", padding="max_length",
                     max_length=T_full, truncation=True)["input_ids"][0].tolist()
    for variant in [" " + target, target]:
        tgt = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not tgt:
            continue
        n = len(tgt)
        for i in range(len(full) - n + 1):
            if full[i:i + n] == tgt:
                return list(range(i, i + n))
    raise ValueError(f"could not locate token span for '{target}' in prompt")


FREEFLUX_POSITION = [1, 2, 4, 26, 30, 54, 55]
FREEFLUX_CONTENT = [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]


def resolve_layers(name, layer_ids=None):
    all_layers = list(range(57))
    return {
        "all": None,
        "position": FREEFLUX_POSITION,
        "content": FREEFLUX_CONTENT,
        "not-content": [i for i in all_layers if i not in FREEFLUX_CONTENT],
        "not-position": [i for i in all_layers if i not in FREEFLUX_POSITION],
        "custom": [int(x) for x in (layer_ids or "").split(",") if x],
    }[name]


def main():
    ap = argparse.ArgumentParser(
        description="Identity transport under large motion (FLUX.1-Kontext)")
    ap.add_argument("--input", required=True, help="source image")
    ap.add_argument("--mask", required=True, help="rough object mask (box is fine)")
    ap.add_argument("--object", required=True, help="object word (must be in --prompt)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    # core knobs (validated defaults)
    ap.add_argument("--identity-bias", type=float, default=3.5,
                    help="transport strength beta")
    ap.add_argument("--identity-topk", type=int, default=8,
                    help="top-k emergent correspondences per row (0 = uniform ablation)")
    ap.add_argument("--no-vacate", action="store_true",
                    help="ablation: allow same-position copy (in-place editing)")
    # ablation / analysis arms
    ap.add_argument("--identity-ot", action="store_true",
                    help="ablation: entropic-OT value injection instead of top-k bias")
    ap.add_argument("--ot-tau", type=float, default=0.5)
    ap.add_argument("--ot-lambda", type=float, default=0.6)
    ap.add_argument("--ot-iters", type=int, default=5)
    ap.add_argument("--bg-anchor", type=float, default=0.0,
                    help="ablation: extra attention anchor toward own-position ref")
    ap.add_argument("--bg-halo", type=int, default=0,
                    help="ablation: dilate editedness before background gating")
    ap.add_argument("--capture-entropy", action="store_true",
                    help="record within-object attention stats (analysis)")
    ap.add_argument("--baseline", action="store_true",
                    help="run plain FLUX Kontext (no intervention) for comparison")
    # schedule / runtime
    ap.add_argument("--bg-text-suppress", type=float, default=4.0,
                    help="prompt-deafening strength for unclaimed rows")
    ap.add_argument("--identity-from-step", type=int, default=0)
    ap.add_argument("--identity-until-step", type=int, default=28)
    ap.add_argument("--layers", default="content",
                    choices=["all", "position", "content", "not-content",
                             "not-position", "custom"])
    ap.add_argument("--layer-ids", default=None)
    ap.add_argument("--preserve-until-step", type=int, default=20,
                    help="latent preservation active until this step")
    ap.add_argument("--preserve-feather", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--max-memory", default="0:22GiB,1:22GiB")
    args = ap.parse_args()

    import numpy as np
    import torch
    # the manual attention path does fp32 matmuls — let them use tensor cores
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
    print(f"Image  : {W}x{H}, grid={GRID}   object: '{args.object}'")
    print(f"Prompt : {args.prompt}")

    # ── object mask → token set (soft prior) ──
    m = np.array(Image.open(args.mask).convert("L")) > 128
    m_pil = Image.fromarray((m * 255).astype(np.uint8), mode="L")
    obj_tokens = mask_to_token_indices(m_pil, grid_size=GRID)
    obj_set = set(obj_tokens)
    print(f"  mask: {m.sum()/(H*W)*100:.0f}% of image, {len(obj_tokens)}/{N} tokens")

    tokenizer = T5TokenizerFast.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", subfolder="tokenizer_2")
    obj_span = find_token_span(tokenizer, args.prompt, args.object, T_FULL)
    print(f"  '{args.object}' text span: {obj_span}")

    # ── full canvas: every token participates ──
    all_gen = list(range(N))
    bg_rows = [g for g in all_gen if g not in obj_set]

    # vacate: object rows cannot copy their own ref position — the object
    # must re-form via cross-position transport wherever the prompt puts it
    text_bias = torch.zeros(N, T_FULL, dtype=torch.float32)
    ref_bias = torch.zeros(N, N, dtype=torch.float32)
    if not args.no_vacate:
        obj_cols = sorted(obj_set)
        for g in obj_tokens:
            ref_bias[g, obj_cols] = -1e4

    # ── model ──
    print("\nFLUX.1-Kontext-dev  device_map=balanced")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=parse_max_memory(args.max_memory),
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    src_packed = encode_and_pack(pipe, src_img)

    if args.baseline:
        class _Tracker:  # step counter only
            current_step = 0
        proc, latent_gate = _Tracker(), None
        print("  [baseline] plain FLUX Kontext, no intervention")
    else:
        proc = RegionalMRASProcessor(
            gen_indices=all_gen, text_bias=text_bias, ref_bias=ref_bias,
            apply_until_step=args.steps, N_total=N,
            identity_from_step=args.identity_from_step,
            identity_until_step=args.identity_until_step,
        )
        ot_cfg = (dict(tau=args.ot_tau, lam=args.ot_lambda, iters=args.ot_iters)
                  if args.identity_ot else None)
        # under vacate the mask rows are ref-blocked, so a mask-peaked prior is
        # useless there and loses elsewhere — start uniform; the harvest
        # relocates the gain to wherever the object re-forms
        if args.no_vacate:
            prior = [1.0 if g in obj_set else 0.25 for g in all_gen]
        else:
            prior = [0.6] * N
        proc.add_dynamic(all_gen, obj_span, sorted(obj_set), args.identity_bias,
                         in_ref=True, init_gain=prior,
                         topk=args.identity_topk, ot=ot_cfg)
        proc.set_soft_background(bg_rows, bg_rows, args.bg_anchor,
                                 text_suppress=args.bg_text_suppress,
                                 halo=args.bg_halo)
        proc.capture_entropy = args.capture_entropy
        lf = resolve_layers(args.layers, args.layer_ids)
        install(pipe, proc, layer_filter=lf, identity_layer_filter=lf)
        latent_gate = (torch.tensor(bg_rows, dtype=torch.long),
                       torch.tensor(bg_rows, dtype=torch.long))
        mode = "OT" if args.identity_ot else (
            f"top-{args.identity_topk}" if args.identity_topk else "uniform")
        print(f"  transport: {mode}  beta={args.identity_bias}  "
              f"vacate={'off' if args.no_vacate else 'on'}")

    callback = make_callback(proc, src_packed, latent_gate,
                             args.preserve_until_step, args.preserve_feather)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {args.steps} steps  gs={args.guidance_scale}  seed={args.seed}")
    result = pipe(
        image=src_img, prompt=args.prompt, height=H, width=W,
        num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        callback_on_step_end=callback,
    ).images[0]
    result.save(out_path)
    print(f"[output] saved: {out_path}")

    if args.capture_entropy and hasattr(proc, "entropy_summary"):
        for s in proc.entropy_summary():
            print(f"[entropy] topk={s['topk']} beta={s['beta']}: "
                  f"ent={s['ent_norm']:.3f} obj_mass={s['obj_mass']:.4f} "
                  f"top8={s['top8_share']:.3f} hi_rows={s['n_hi_rows']:.0f}")


if __name__ == "__main__":
    main()
