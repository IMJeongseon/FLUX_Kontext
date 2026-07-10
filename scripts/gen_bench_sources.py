"""Generate benchmark source images (scale-up set) with FLUX.1-dev.

Same design rule as gen_source.py: every subject carries a DISTINCTIVE,
non-categorical, localized instance marker (number / letter / pattern /
accessory) on the torso so it stays croppable under large motion, and the
composition is full-body side view on a plain ground so motion edits
(jump / run / rear up) are plausible.
"""
import sys
from pathlib import Path
import torch
from diffusers import FluxPipeline

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "sources"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    # (name, prompt) — markers vary in type: digits, letters, geometric
    # patterns, accessories; subjects vary in species and background.
    ("pug_shirt",
     "A fawn pug wearing a green sleeveless shirt with a large white letter "
     "\"K\" printed on the side, standing on a paved park path, "
     "photorealistic, full body, side view, sharp focus, plain background."),
    ("rabbit_vest",
     "A white rabbit wearing a small orange vest with a black diamond "
     "pattern, sitting on short green grass in a garden, photorealistic, "
     "full body, side view, sharp focus, plain background."),
    ("horse_blanket",
     "A brown horse wearing a blue and white checkered blanket with a large "
     "red number \"7\" on it, standing in a grassy field, photorealistic, "
     "full body, side view, sharp focus, plain background."),
    ("penguin_scarf",
     "A penguin wearing a green and yellow striped scarf, standing on snow, "
     "photorealistic, full body, side view, sharp focus, plain background."),
    ("goat_blanket",
     "A white goat wearing a purple blanket with three yellow stars printed "
     "on it, standing on a dirt farm yard, photorealistic, full body, "
     "side view, sharp focus, plain background."),
    ("duck_ribbon",
     "A white duck wearing a red and white striped ribbon around its neck, "
     "standing on grass near a pond, photorealistic, full body, side view, "
     "sharp focus, plain background."),
    ("sheep_tag",
     "A woolly sheep with a large blue number \"3\" spray-marked on its "
     "side, standing in a green meadow, photorealistic, full body, "
     "side view, sharp focus, plain background."),
    ("robot_decal",
     "A small white toy robot with a red lightning bolt decal on its chest, "
     "standing on a light wooden table, photorealistic, full body, "
     "front three-quarter view, sharp focus, plain background."),
    ("kitten_sweater",
     "A gray kitten wearing a knitted red sweater vest with a white "
     "snowflake pattern, standing on a light carpet indoors, photorealistic, "
     "full body, side view, sharp focus, plain background."),
    ("dalmatian_collar",
     "A dalmatian dog with distinctive black spots wearing a wide yellow "
     "collar with a round silver name tag, standing on a sidewalk, "
     "photorealistic, full body, side view, sharp focus, plain background."),
]

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

for name, prompt in CANDIDATES:
    p = OUT / f"{name}.png"
    if p.exists():
        print(f"[skip] {p}")
        continue
    img = pipe(
        prompt,
        height=1024, width=1024,
        num_inference_steps=28, guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]
    img.save(p)
    print(f"[gen] {p}", flush=True)
