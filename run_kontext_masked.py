"""FLUX.1-Kontext + RePaint: mask 안쪽만 편집, 배경은 source latent로 앵커링.

MRAS의 encode_and_pack + repaint 로직을 재사용 (../MRAS/scripts/run_removal.py).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MRAS_SRC = HERE.parent / "MRAS" / "src"
sys.path.insert(0, str(MRAS_SRC))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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


def make_repaint_callback(src_packed, bg_tensor):
    import torch

    def callback(pipeline, step_idx, timestep, kwargs):
        latents = kwargs["latents"]
        device, dtype = latents.device, latents.dtype

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
        latents[:, bg, :] = noised.expand(latents.shape[0], -1, -1)[:, bg, :]
        kwargs["latents"] = latents
        return kwargs

    return callback


def parse_max_memory(spec):
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        k, v = item.split(":", 1)
        k = k.strip()
        out[int(k) if k.isdigit() else k] = v.strip()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(HERE / "data" / "image.png"))
    parser.add_argument("--mask", default=str(HERE / "data" / "mask.png"))
    parser.add_argument("--output", default=str(HERE / "outputs" / "cat_edit_masked.png"))
    parser.add_argument(
        "--prompt",
        default="Replace the dog with a cat and the house with a cat tower.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-memory", default="0:22GiB,1:22GiB")
    args = parser.parse_args()

    import torch
    from diffusers import FluxKontextPipeline
    from PIL import Image
    from pom1.mras import mask_to_token_indices

    src = Image.open(args.input).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    W, H = src.size
    GRID = W // 16

    mask_idx = mask_to_token_indices(mask, grid_size=GRID)
    mask_set = set(mask_idx)
    bg_idx = [i for i in range(GRID * GRID) if i not in mask_set]
    print(f"mask tokens: {len(mask_idx)}, bg tokens: {len(bg_idx)} / {GRID*GRID}")

    print("[load] FLUX.1-Kontext-dev  device_map=balanced")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=parse_max_memory(args.max_memory),
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print("[encode] source latent")
    src_packed = encode_and_pack(pipe, src)
    bg_tensor = torch.tensor(bg_idx, dtype=torch.long)
    callback = make_repaint_callback(src_packed, bg_tensor)

    print(f"[run] {W}x{H} steps={args.steps} gs={args.guidance_scale} seed={args.seed}")
    result = pipe(
        image=src,
        prompt=args.prompt,
        height=H,
        width=W,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        callback_on_step_end=callback,
    ).images[0]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
