"""Best verified RTAR recipe — v3 "soft attention background" (2026-07-09).

Mask-FREE editing: reproduces outputs/rtar/dog_sitting_softbg_v3.png — the
source dog sits on a real toilet (no mask-shape forcing) with its identity
preserved, and the grass-yard background is kept, all through attention alone
(no RePaint latent surgery).

Recipe:
  * full-image canvas; masks act only as soft position priors and mark the
    source objects that disappear
  * preserve mode w/ dynamic identity bias beta=2.5 (silhouette gain-scaled),
    from step 0, on FreeFlux content-similarity layers
  * dynamic target binding beta=2.0 ("toilet" forms its own shape)
  * soft background anchor: +2.5*(1-editedness) to own-position ref token AND
    -4.0*(1-editedness) on text columns (background does not hear the prompt)
  * pose-explicit prompt ("The dog is sitting on the toilet.")

Usage:
    python best_ver.py                        # reproduce the reference result
    python best_ver.py --seed 7               # any rtar_pipeline.py flag
    python best_ver.py --two-pass             # + pass-2 pixel background lock
    python best_ver.py --input other.png --mask-paths "..." ...

Later flags override the frozen defaults (argparse keeps the last occurrence).
The previous mask-based recipe (b25_soft_r12) remains reachable by overriding:
    python best_ver.py --soft-background 0 --repaint-dilate 2 \
        --repaint-spatial-feather 3 --identity-dynamic
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULTS = [
    "--input", "data/dog.png",
    "--mask-paths", "data/masks/mask_dog_clean.png;data/masks/mask_house.png",
    "--source-objects", "dog,house",
    "--modes", "preserve,replace",
    "--dest-map", "dog:house",
    "--target-phrases=-;toilet",
    "--prompt", "The dog is sitting on the toilet.",
    "--identity-bias", "2.5",
    "--identity-from-step", "0",
    "--identity-layers", "content",
    # manual attention only on the 13 content layers: 2.1x faster (10.3 vs
    # 21.6 s/it) with no visible quality loss; pass --layers all to revert
    "--layers", "content",
    "--soft-background", "2.5",
    "--soft-background-text", "4.0",
    "--soft-background-latent",   # editedness-gated latent bg preservation:
                                  # background stays source pixels (no over-sharpen),
                                  # only the edited object regenerates. mask-free.
    "--target-dyn-bias", "2.0",
    "--repaint-until-step", "16",
    "--repaint-feather", "4",
    "--output", "outputs/rtar/best_ver.png",
    "--save-mask-figure",
]

cmd = [sys.executable, str(HERE / "rtar_pipeline.py"), *DEFAULTS, *sys.argv[1:]]
print("[best_ver]", " ".join(cmd[1:]))
sys.exit(subprocess.run(cmd, cwd=HERE).returncode)
