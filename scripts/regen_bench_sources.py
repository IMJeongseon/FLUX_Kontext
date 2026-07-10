"""Regenerate benchmark sources that came out as cartoon/illustration.

Lesson from the first pass: "photorealistic ... plain background" drifts to
character illustration for animal-in-clothes subjects. Anchor the PHOTO domain
hard instead: camera/lens words up front, natural setting, no "plain
background", and keep accessories that exist in real photos (harness, rug,
livestock spray mark) rather than costume-like ones.
"""
import sys
from pathlib import Path
import torch
from diffusers import FluxPipeline

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "sources"

PHOTO = ("Amateur DSLR photograph, 50mm lens, natural daylight, "
         "realistic fur and texture, high detail photo. ")

CANDIDATES = [
    ("rabbit_vest",
     PHOTO + "A real white rabbit wearing a small orange pet harness vest "
     "with a black diamond pattern printed on it, sitting on short grass in "
     "a backyard garden, full body, side view."),
    ("penguin_scarf",
     PHOTO + "A real penguin standing on snow wearing a green and yellow "
     "striped knitted scarf, wildlife photography, full body, side view."),
    ("goat_blanket",
     PHOTO + "A real white goat at a farm wearing a purple goat coat with "
     "three yellow stars printed on the side, standing on a dirt farmyard, "
     "full body, side view."),
    ("duck_ribbon",
     PHOTO + "A real white domestic duck with a red and white striped "
     "ribbon tied around its neck, standing on grass near a pond, "
     "full body, side view."),
    ("sheep_tag",
     PHOTO + "A real woolly sheep in a green meadow with a large blue "
     "number \"3\" spray-marked on the wool of its side, livestock marking "
     "paint, full body, side view."),
    ("horse_blanket",
     PHOTO + "A real brown horse in a grassy paddock wearing a blue and "
     "white checkered stable rug with a large red number \"7\" printed on "
     "it, full body, side view."),
    ("dalmatian_collar",
     PHOTO + "A real dalmatian dog with black spots wearing a wide yellow "
     "collar with a round silver name tag, standing on a sidewalk in a "
     "park, full body, side view."),
]

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
for name, prompt in CANDIDATES:
    img = pipe(
        prompt,
        height=1024, width=1024,
        num_inference_steps=28, guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    p = OUT / f"{name}.png"
    img.save(p)
    print(f"[regen seed={seed}] {p}", flush=True)
