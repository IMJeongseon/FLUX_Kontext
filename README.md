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
| **Vacate** | object's source rows | block same-position copy → the object must re-form where the prompt says |

Constants: FreeFlux content layers (13), `β=3.5`, `k=8`. Rough box masks
suffice (soft prior only). Prompt rule: describe preserved attributes with
**neutral words** (e.g. "yellow outfit", not "raincoat" — wrong structural
priors override the reference).

## Quickstart

```bash
# keep THIS corgi's identity, let it jump (large motion)
python src/pipeline.py \
    --input data/sources/corgi_raincoat.png \
    --mask data/masks/mask_corgi_rough.png \
    --object corgi \
    --prompt "The corgi wearing a yellow outfit with the number 5 is jumping high in the air to catch a red frisbee." \
    --output outputs/corgi_jump.png
```

Defaults are the validated recipe (top-8 transport, β=3.5, vacate on,
content layers). Ablation arms: `--no-vacate`, `--identity-topk 0` (uniform),
`--identity-ot`, `--baseline` (plain Kontext).

Requires: 2×24GB GPUs (`device_map=balanced`), diffusers ≥0.35,
FLUX.1-Kontext-dev weights (HF gated).

## Layout

```
src/pipeline.py    # THE method (single entry point, validated defaults)
src/regional_processor.py   # gated-bias attention processor
eval/              # metrics (marker-survival, CLIP-direction motion) + manifests
scripts/           # data generation utilities
data/              # source images + rough masks
outputs/           # results (gitignored)
```

## Evaluation

`eval/metrics.py` scores each edit with: **marker_survival** (retrieval-style
DINOv2 max-cosine of the source marker crop over sliding output crops + HSV
histogram of the best match), **motion** (CLIP direction: action prompt vs
static counterpart), and CLIP-T. See `eval/pilot_manifest.json` for the
case format.

## Status — the open problem (2026-07-11)

**Goal**: identity preserved under LARGE motion. Current standing on the
10-case benchmark (`eval/bench_manifest.json`): motion+identity on **8/10**
after seed/prompt-rule fixes; the residual problem is that **fine identity
degrades exactly when motion succeeds** — e.g. the pug's letter "K":
marker-DINO 0.80 when static, 0.59–0.64 mid-jump.

Mathematical diagnosis (error decomposition of attention transport
`o_i = Σ_j P_ij v_j` against the true correspondence π):

| term | meaning | attack |
|---|---|---|
| E_mix | row entropy — convex mixing blurs high-freq | top-k (in), sharper β saturates |
| E_assign | no column constraint — many-to-one collapse | `--identity-ot` (Sinkhorn), **experiment queued** |
| E_arrange | matches ignore spatial coherence — detail scrambles | motion-model feedback (planned) |
| E_quant | VAE 16×16px patch floor | irreducible, stated as limit |

Bias-only methods (everything operating on attention scores) provably leave
E > 0: softmax output stays inside the convex hull, and 44 uncontrolled
layers keep injecting the category prior. Hence **solution A** (`a62cf92`):
make the correspondence OBSERVED (DINO matching + locally-rigid fit around
the marker, `src/observe_warp.py`) and enforce identity as a state-space
**projection** (re-noised composite latent overwrites the trajectory,
`src/twostage.py`) — the same mechanism class that already makes our
background preservation exact. Verification run is queued (see HANDOFF.md).

## TODO (priority order)

1. **Stage-2 projection test** (pug s123): does the observed-warp anchor
   restore the "K"? — commands + pass criteria in [HANDOFF.md](HANDOFF.md)
2. **OT vs top-k** on motion-success cases → measures the E_assign share
3. T2I prior-ceiling test for the 2 residual motion failures
   (goat rearing / penguin belly-slide): our defect vs inherited ceiling
4. Full 10 cases × 3 seeds success table; then baselines
   (plain Kontext / KV-Edit / FlowEdit) on identical neutral prompts
5. Metrics v3: 3×3 grid-cell color histogram (spatially aware — E_arrange)

## Notes

- Full exploration history (including ablated mechanisms: uniform/OT
  transport arms, ramp, semantic gate, two-pass) is preserved in git history —
  see the `snapshot` commit.
- Known limitation: garment *topology* is decided by the text prior; use
  neutral garment words (see prompt rule above).
- Motion reliability: displacement is delegated to the text prior — verbs
  whose layout the prior barely contains (body-axis rotation, biped pose for
  a clothed quadruped) fail across seeds; `--motion-boost` amplifies an
  existing layout basin but cannot create one (saturates by β=4).
