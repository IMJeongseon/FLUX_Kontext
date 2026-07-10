"""Generate source images for the instance-identity-transport experiment.

Each source carries a DISTINCTIVE non-categorical instance feature (a marked
accessory / text tag) that is localized and easy to crop for an identity
metric — generic regeneration destroys it, instance transport keeps it.
"""
import sys
from pathlib import Path
import torch
from diffusers import FluxPipeline

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "sources"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    # (name, prompt)
    ("corgi_raincoat",
     "A corgi wearing a bright yellow raincoat with a large white number \"5\" "
     "printed on it, standing on green grass in a park, sharp focus, "
     "photorealistic, full body, side view, plain background."),
    ("cat_scarf",
     "A fluffy orange cat wearing a red scarf with white polka dots and a small "
     "golden bell, sitting upright on a wooden floor, photorealistic, full body, "
     "sharp focus, plain background."),
    ("dog_bandana",
     "A white samoyed dog wearing a blue bandana with a big yellow star pattern, "
     "standing on grass, photorealistic, full body, side profile, sharp focus, "
     "plain background."),
]

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

for name, prompt in CANDIDATES:
    img = pipe(
        prompt,
        height=1024, width=1024,
        num_inference_steps=28, guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]
    p = OUT / f"{name}.png"
    img.save(p)
    print(f"[gen] {p}")
