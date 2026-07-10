# FLUX_Kontext — Instance Identity Transport under Large Motion

Training-free, inversion-free, single-pass image editing on FLUX.1-Kontext that
**moves / re-poses an object with large motion while preserving its instance
identity** (unique patterns, markers, garment appearance) — the corner that
existing methods (FLUX Kontext, KV-Edit, FlowEdit, SynPS, Cora) each miss.

## Method in one line

An online **editedness** map `e` (harvested from the model's own text→image
attention) gates a single bias field on Kontext's joint `[text|gen|ref]`
attention, with three roles:

| role | rows | action |
|---|---|---|
| **Transport** | being edited (`e`↑) | boost each row's top-k emergent gen→ref matches (`+β·e`) — carries high-frequency instance detail to the object's **new** location |
| **Preserve** | untouched (`e`↓) | latent pixel copy + prompt-deafening (`×(1−e)`) — background stays source-faithful |
| **Vacate** (move mode) | object's source rows | block same-position copy → the object must re-form where the prompt says |

Constants: FreeFlux content layers (13), `β=3.5`, `k=8`. Rough box masks
suffice (soft prior only). Prompt rule: describe preserved attributes with
**neutral words** (e.g. "yellow outfit", not "raincoat" — wrong structural
priors override the reference).

## Quickstart

```bash
# move: keep THIS corgi's identity, let it jump (large motion)
python lean.py --input data/sources/corgi_raincoat.png \
    --mask data/masks/mask_corgi_rough.png --object corgi --mode move \
    --identity 3.5 --topk 8 \
    --prompt "The corgi wearing a yellow outfit with the number 5 is jumping high in the air to catch a red frisbee." \
    --output outputs/corgi_jump.png
```

Modes: `replace` (object → target phrase) · `keep` (identity, in place) ·
`move` (identity + prompt-driven motion).

Requires: 2×24GB GPUs (`device_map=balanced`), diffusers ≥0.35,
FLUX.1-Kontext-dev weights (HF gated).

## Layout

```
lean.py            # main entry point (3 modes, core knobs only)
best_ver.py        # frozen best-recipe launcher
src/               # core: pipeline + attention processor
eval/              # metrics (marker-survival, CLIP-direction motion) + manifests
scripts/           # data generation utilities
legacy/            # exploration-era scripts (kept for ablation reproduction)
data/              # source images + rough masks
outputs/           # results (gitignored)
```

## Evaluation

`eval/metrics.py` scores each edit with: **marker_survival** (retrieval-style
DINOv2 max-cosine of the source marker crop over sliding output crops + HSV
histogram of the best match), **motion** (CLIP direction: action prompt vs
static counterpart), and CLIP-T. See `eval/pilot_manifest.json` for the
case format.

## Notes

- Full exploration history (including ablated mechanisms: uniform/OT
  transport arms, ramp, semantic gate, two-pass) is preserved in git history —
  see the `snapshot` commit.
- Known limitation: garment *topology* is decided by the text prior; use
  neutral garment words (see prompt rule above).
