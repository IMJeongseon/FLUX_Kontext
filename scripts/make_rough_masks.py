"""Draw rough box masks for benchmark sources (loose boxes suffice —
the mask is only a soft location prior for the pipeline)."""
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "masks"
OUT.mkdir(parents=True, exist_ok=True)

# name -> (left, top, right, bottom) in fractions of the 1024 canvas,
# eyeballed loosely around the full subject
BOXES = {
    "pug_shirt": (0.10, 0.13, 0.99, 0.99),
    "robot_decal": (0.24, 0.14, 0.76, 0.90),
    "kitten_sweater": (0.08, 0.08, 0.97, 0.99),
    "rabbit_vest": (0.10, 0.06, 0.92, 0.97),
    "penguin_scarf": (0.14, 0.06, 0.93, 0.99),
    "goat_blanket": (0.08, 0.05, 0.95, 0.99),
    "duck_ribbon": (0.03, 0.06, 0.95, 1.00),
    "sheep_tag": (0.05, 0.10, 0.92, 0.97),
    "horse_blanket": (0.06, 0.08, 0.99, 0.99),
    "dalmatian_collar": (0.27, 0.08, 0.88, 1.00),
}

for name, (l, t, r, b) in BOXES.items():
    m = Image.new("L", (1024, 1024), 0)
    d = ImageDraw.Draw(m)
    d.rectangle([int(l * 1024), int(t * 1024), int(r * 1024), int(b * 1024)],
                fill=255)
    p = OUT / f"mask_{name}_rough.png"
    m.save(p)
    print(f"[mask] {p}")
