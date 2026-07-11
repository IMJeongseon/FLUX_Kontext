"""Prior ceiling test: can FLUX.1-dev even COMPOSE the failing scenes as
pure T2I, with no editing constraint? If not, the editing failures are an
inherited prior ceiling, not a method deficiency."""
import torch
from pathlib import Path
from diffusers import FluxPipeline

OUT = Path(__file__).resolve().parent.parent / "outputs" / "bench"

CASES = [
    ("t2i_goat_rearing",
     "Amateur DSLR photograph, 50mm lens, natural daylight. A real white "
     "goat wearing a purple coat with yellow stars, rearing up on its hind "
     "legs, front legs high in the air, on a dirt farmyard, full body, "
     "side view."),
    ("t2i_penguin_slide",
     "Amateur DSLR photograph, 50mm lens, natural daylight. A real penguin "
     "wearing a green and yellow striped scarf, tobogganing flat on its "
     "belly across the snow, body horizontal, wildlife photography, "
     "full body, side view."),
]

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

for name, prompt in CASES:
    for seed in (42, 123):
        img = pipe(prompt, height=1024, width=1024,
                   num_inference_steps=28, guidance_scale=3.5,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        p = OUT / f"{name}_s{seed}.png"
        img.save(p)
        print(f"[t2i] {p}", flush=True)
