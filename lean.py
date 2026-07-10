"""RTAR-lean: the core method, stripped to essentials.

One idea — editedness-gated routing: harvest an online "editedness" map from
the model's own text->image attention, then use it to gate, per token,
(a) the identity/suppression score bias on Kontext's [text | gen | ref]
attention, AND (b) a latent RePaint anchor that pixel-preserves everything
the object does NOT claim. Same map drives both. The (rough) mask is a soft
prior; a loose box works because editedness — not the mask shape — decides
what is edited vs preserved. Latent preservation of the un-edited region is
what keeps the background pixel-faithful (no over-sharpening).

Three modes, one mechanism:
  replace : object -> target phrase   (suppress source text + block gen->ref)
  keep    : preserve identity in place (identity bias, static pose)
  move    : preserve identity + let the prompt move/re-pose it (adds vacate:
            block same-position copy so the object must re-form elsewhere)

Everything else in rtar_pipeline.py (soft-background variants, canvas ring,
ramp, two-pass, extra-objects) is exploration/ablation and NOT part of the
lean method. This wrapper exposes only the load-bearing knobs.

Usage:
  # move: keep THIS corgi's identity, let it jump (large motion)
  python lean.py --input data/sources/corgi_raincoat.png \
      --mask data/masks/mask_corgi_rough.png --object corgi --mode move \
      --prompt "The corgi is jumping high to catch a red frisbee." \
      --output outputs/lean/corgi_jump.png

  # replace: dog -> cat
  python lean.py --input data/dog.png --mask data/masks/mask_dog_clean.png \
      --object dog --mode replace --target cat \
      --prompt "Replace the dog with a cat." --output outputs/lean/cat.png

  # keep: identity, in place
  python lean.py --input img.png --mask m.png --object cat --mode keep \
      --prompt "The cat wearing sunglasses." --output out.png
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="RTAR-lean: core Kontext identity editing")
    ap.add_argument("--input", required=True)
    ap.add_argument("--mask", required=True, help="rough object mask (box is fine)")
    ap.add_argument("--object", required=True, help="source object word (must be in --prompt)")
    ap.add_argument("--mode", required=True, choices=["replace", "keep", "move"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--target", default=None, help="destination phrase (replace mode)")
    ap.add_argument("--output", required=True)
    # the only two strength knobs that matter
    ap.add_argument("--identity", type=float, default=2.5,
                    help="identity transport strength (keep/move)")
    ap.add_argument("--topk", type=int, default=0,
                    help="correspondence-sharpened transport: boost each row's top-K "
                         "emergent gen->ref matches instead of uniform region bias")
    ap.add_argument("--suppress", type=float, default=-1e4,
                    help="source-removal strength (replace)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-figure", action="store_true")
    args = ap.parse_args()

    if args.mode == "replace" and not args.target:
        ap.error("--mode replace needs --target")

    # ---- fixed, validated defaults for the lean core ----
    # editedness-gated routing: soft-background attention anchor + text-suppress
    # keep the un-edited region on-content and prompt-deaf; soft-background-latent
    # pixel-preserves it (no over-sharpening); a loose box works because the
    # editedness map, not the mask, decides the edited region.
    common = [
        "--input", args.input,
        "--mask-paths", args.mask,
        "--source-objects", args.object,
        "--prompt", args.prompt,
        "--seed", str(args.seed),
        "--identity-layers", "content",   # appearance channel only
        "--layers", "content",            # 13 manual layers: 2x faster, no quality loss
        "--suppression-bias", str(args.suppress),
        "--soft-background", "2.5",
        "--soft-background-text", "4.0",
        "--soft-background-latent",        # pixel-faithful background
        "--repaint-until-step", "20",      # hold the latent anchor late so the
        "--repaint-feather", "4",          # far background stays source-soft

        "--output", args.output,
    ]
    if args.save_figure:
        common.append("--save-mask-figure")

    if args.mode == "replace":
        mode = ["--modes", "replace", f"--target-phrases={args.target}",
                "--target-dyn-bias", "2.0"]
    elif args.mode == "keep":
        mode = ["--modes", "preserve", "--dest-map", f"{args.object}:{args.object}",
                "--target-phrases=-", "--identity-dynamic",
                "--identity-bias", str(args.identity),
                "--identity-topk", str(args.topk)]
    else:  # move
        mode = ["--modes", "preserve", "--dest-map", f"{args.object}:{args.object}",
                "--target-phrases=-", "--identity-dynamic", "--preserve-vacate",
                "--identity-bias", str(args.identity),
                "--identity-topk", str(args.topk)]

    cmd = [sys.executable, str(HERE / "rtar_pipeline.py"), *common, *mode]
    print("[lean]", args.mode, "->", args.output)
    sys.exit(subprocess.run(cmd, cwd=HERE).returncode)


if __name__ == "__main__":
    main()
