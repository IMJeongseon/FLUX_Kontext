"""FLUX.1-Kontext edit: dog→cat, house→cat tower."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(HERE / "data" / "image.png"))
    parser.add_argument("--output", default=str(HERE / "outputs" / "cat_edit.png"))
    parser.add_argument(
        "--prompt",
        default="Replace the dog with a cat and the house with a cat tower.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--offload",
        choices=["sequential", "model"],
        default="sequential",
        help="sequential = per-module offload (slow, tiny VRAM); model = whole-model (fast, needs ~24GB).",
    )
    args = parser.parse_args()

    import torch
    from diffusers import FluxKontextPipeline
    from PIL import Image

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input image not found: {in_path}")

    src = Image.open(in_path).convert("RGB")
    W, H = src.size

    print(f"[load] FLUX.1-Kontext-dev  offload={args.offload}")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    print(f"[run] {W}x{H} steps={args.steps} gs={args.guidance_scale} seed={args.seed}")
    result = pipe(
        image=src,
        prompt=args.prompt,
        height=H,
        width=W,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
