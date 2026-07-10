"""FLUX.1-Kontext + regional binding with distant/blurry cat tower.

- house mask: shifted to 1 o'clock direction and shrunk (represents distant object)
- prompt: 'small distant blurry cat tower' with regional binding
- post-process: Gaussian blur on cat-tower region for DoF effect
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


def make_repaint_callback(proc, src_packed, bg_tensor):
    import torch

    def callback(pipeline, step_idx, timestep, kwargs):
        proc.current_step = step_idx
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


def apply_dof_blur(image, mask_pil, radius=3.5, feather=12):
    """Post-process depth-of-field: Gaussian blur mask region with feathered edge."""
    import numpy as np
    from PIL import Image, ImageFilter

    src = image.convert("RGB")
    blurred = src.filter(ImageFilter.GaussianBlur(radius=radius))
    # Feathered mask for smooth transition
    m = mask_pil.convert("L").filter(ImageFilter.GaussianBlur(radius=feather))
    out = Image.composite(blurred, src, m)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(HERE / "data" / "image.png"))
    parser.add_argument("--dog-mask", default=str(HERE / "data" / "mask_dog_clean.png"))
    parser.add_argument("--house-mask", default=str(HERE / "data" / "mask_house_distant.png"))
    parser.add_argument("--output", default=str(HERE / "outputs" / "cat_edit_distant.png"))
    parser.add_argument(
        "--prompt",
        default="Replace the dog with a cat and the house with a small distant blurry cat tower.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--suppression-bias", type=float, default=-1e4)
    parser.add_argument("--apply-until-step", type=int, default=28)
    parser.add_argument("--blur-radius", type=float, default=3.5)
    parser.add_argument("--blur-feather", type=int, default=12)
    parser.add_argument("--max-memory", default="0:22GiB,1:22GiB")
    args = parser.parse_args()

    import torch
    from diffusers import FluxKontextPipeline
    from PIL import Image, ImageFilter
    from pom1.mras import mask_to_token_indices
    from regional_processor import RegionalMRASProcessor, install

    src = Image.open(args.input).convert("RGB")
    W, H = src.size
    GRID = W // 16
    N = GRID * GRID
    T_FULL = 512

    dog_mask = Image.open(args.dog_mask).convert("L")
    house_mask = Image.open(args.house_mask).convert("L")
    dog_idx = mask_to_token_indices(dog_mask, grid_size=GRID)
    house_idx = mask_to_token_indices(house_mask, grid_size=GRID)
    print(f"dog: {len(dog_idx)} tokens, house(distant): {len(house_idx)} tokens")

    seen = set()
    all_gen = []
    is_dog = []
    for g in dog_idx:
        if g not in seen:
            all_gen.append(g); is_dog.append(True); seen.add(g)
    for g in house_idx:
        if g not in seen:
            all_gen.append(g); is_dog.append(False); seen.add(g)
    n_gen = len(all_gen)
    bg_idx = [i for i in range(N) if i not in seen]
    print(f"gen union: {n_gen}, bg: {len(bg_idx)} / {N}")

    # Prompt tokens (new prompt):
    # 0:Replace 1:the 2:dog 3:with 4:_ 5:a 6:cat 7:and 8:the 9:house
    # 10:with 11:_ 12:a 13:small 14:distant 15:blur 16:ry 17:cat 18:tower
    dog_source_span = [2]
    cat_span = [6]
    house_source_span = [9]
    cat_tower_span = [13, 14, 15, 16, 17, 18]  # "small distant blurry cat tower"

    text_bias = torch.zeros(n_gen, T_FULL, dtype=torch.float32)
    ref_bias = torch.zeros(n_gen, N, dtype=torch.float32)
    dog_ref_set = set(dog_idx)
    house_ref_set = set(house_idx)
    for i, (g, is_d) in enumerate(zip(all_gen, is_dog)):
        if is_d:
            for t in house_source_span + cat_tower_span:
                text_bias[i, t] = args.suppression_bias
            for r in dog_ref_set:
                ref_bias[i, r] = args.suppression_bias
        else:
            for t in dog_source_span + cat_span:
                text_bias[i, t] = args.suppression_bias
            for r in house_ref_set:
                ref_bias[i, r] = args.suppression_bias

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

    proc = RegionalMRASProcessor(
        gen_indices=all_gen,
        text_bias=text_bias,
        ref_bias=ref_bias,
        apply_until_step=args.apply_until_step,
        N_total=N,
    )
    install(pipe, proc)

    bg_tensor = torch.tensor(bg_idx, dtype=torch.long)
    callback = make_repaint_callback(proc, src_packed, bg_tensor)

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
    print(f"saved (no blur): {out_path}")

    # Post-process DoF: blur the cat tower (house-mask) region with feathered edge
    # Use a slightly dilated mask to catch tower silhouette leakage
    dilated = house_mask.filter(ImageFilter.MaxFilter(size=9))
    blurred_result = apply_dof_blur(result, dilated, radius=args.blur_radius, feather=args.blur_feather)
    blur_out = out_path.with_name(out_path.stem + "_dof.png")
    blurred_result.save(blur_out)
    print(f"saved (DoF): {blur_out}")


if __name__ == "__main__":
    main()
