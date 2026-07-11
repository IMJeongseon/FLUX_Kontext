"""Solution-A Stage 2: identity by PROJECTION, not preference.

Takes the Stage-1 composite (source background + warped source object) and
harmonizes it under FLUX.1-Kontext with the source as reference. Preservation
is a state-space equality constraint: at every anchored step the latent is
overwritten with the (re-noised) composite latent, so by induction the object
appearance IS the warped source up to VAE quantization — the attention/prior
can only touch seams, lighting and the released late steps.

Schedule:
  step < anchor-all-until   : everything except the vacated region anchored
  step < anchor-core-until  : the object core (eroded Omega) stays anchored
  after                     : free (harmonization)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MRAS_SRC = HERE.parent.parent / "MRAS" / "src"
sys.path.insert(0, str(MRAS_SRC))
sys.path.insert(0, str(HERE))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image, ImageFilter

from pipeline import parse_max_memory, encode_and_pack  # noqa: E402

SIZE, GRID, N = 1024, 64, 4096


def token_weights(mask_img: Image.Image, erode: int = 0) -> torch.Tensor:
    """Soft pixel mask -> per-token weight [N] via 16x16 cell mean."""
    m = mask_img.convert("L").resize((SIZE, SIZE))
    if erode > 0:
        m = m.filter(ImageFilter.MinFilter(2 * erode + 1))
    a = np.array(m, dtype=np.float32) / 255.0
    a = a.reshape(GRID, SIZE // GRID, GRID, SIZE // GRID).mean((1, 3))
    return torch.from_numpy(a.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="original source image (Kontext ref)")
    ap.add_argument("--composite", required=True, help="Stage-1 composite")
    ap.add_argument("--omega-mask", required=True, help="Stage-1 warped object mask")
    ap.add_argument("--vacated-mask", default=None, help="region the object left (stays free)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--anchor-all-until", type=int, default=14)
    ap.add_argument("--anchor-core-until", type=int, default=22)
    ap.add_argument("--core-erode", type=int, default=12, help="px erosion for the core mask")
    ap.add_argument("--feather", type=int, default=4, help="steps to ramp each anchor out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--max-memory", default="0:22GiB,1:22GiB")
    args = ap.parse_args()

    from diffusers import FluxKontextPipeline

    src_img = Image.open(args.source).convert("RGB").resize((SIZE, SIZE))
    comp_img = Image.open(args.composite).convert("RGB").resize((SIZE, SIZE))
    omega = Image.open(args.omega_mask)

    w_omega = token_weights(omega)
    w_core = token_weights(omega, erode=args.core_erode)
    w_all = torch.ones(N)
    if args.vacated_mask:
        w_all = w_all - token_weights(Image.open(args.vacated_mask)).clamp(0, 1)
    print(f"[anchor] all={float(w_all.mean()):.2f} omega={float((w_omega > 0.5).sum())} "
          f"core={float((w_core > 0.5).sum())} tokens")

    print("FLUX.1-Kontext-dev  device_map=balanced")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=parse_max_memory(args.max_memory),
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    comp_packed = encode_and_pack(pipe, comp_img)

    def ramp(step, until):
        if step >= until:
            return 0.0
        start = until - args.feather
        if step < start or args.feather == 0:
            return 1.0
        return max(0.0, 1.0 - (step - start + 1) / (args.feather + 1))

    def callback(pipeline, step_idx, timestep, kwargs):
        latents = kwargs["latents"]
        device, dtype = latents.device, latents.dtype
        a_all = ramp(step_idx, args.anchor_all_until)
        a_core = ramp(step_idx, args.anchor_core_until)
        if a_all <= 0 and a_core <= 0:
            return kwargs
        w = torch.maximum(a_all * w_all, a_core * w_core).clamp(0, 1)

        sched = pipeline.scheduler
        if step_idx + 1 < len(sched.timesteps):
            next_sigma = float(sched.timesteps[step_idx + 1]) / 1000.0
        else:
            next_sigma = 0.0
        clp = comp_packed.to(device, dtype)
        noised = (1.0 - next_sigma) * clp + next_sigma * torch.randn_like(clp)
        wd = w.to(device).to(dtype)[None, :, None]
        kwargs["latents"] = wd * noised.expand(latents.shape[0], -1, -1) \
            + (1 - wd) * latents
        return kwargs

    gen = torch.Generator("cpu").manual_seed(args.seed)
    img = pipe(
        image=src_img,
        prompt=args.prompt,
        height=SIZE, width=SIZE,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=gen,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
