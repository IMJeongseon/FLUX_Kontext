"""Quantitative metrics for instance-identity-under-motion editing.

Metrics
-------
marker_survival : does the source's distinctive marker crop appear anywhere in
    the output? max-over-locations DINOv2 cosine between the source marker crop
    and sliding crops of the output (scales x stride grid). Retrieval-style, so
    no manual box on the output is needed (the object may have moved).
clip_t          : CLIP text-image similarity to the TARGET prompt (edit
    adherence incl. motion).
src_similarity  : whole-image DINO similarity to source (high + high clip_t is
    fine; ~1.0 with low clip_t = "did nothing").

Usage:
    python metrics.py --manifest pilot_manifest.json [--out results.json]

Manifest entry:
    {"name": "corgi/plain", "source": "...", "output": "...",
     "marker_box": [x1,y1,x2,y2], "target_prompt": "..."}
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_dino():
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return m.to(DEV).eval()


def dino_embed(model, pil_imgs):
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = torch.stack([tf(im.convert("RGB")) for im in pil_imgs]).to(DEV)
    with torch.no_grad():
        f = model(x)
    return F.normalize(f, dim=-1)


def color_hist_sim(a, b, bins=16):
    """HSV histogram correlation — patterns are color layouts, DINO alone is
    too semantic to feel them."""
    import numpy as np
    ha = np.asarray(a.convert("HSV").resize((128, 128))).reshape(-1, 3)
    hb = np.asarray(b.convert("HSV").resize((128, 128))).reshape(-1, 3)
    sims = []
    for ch in range(3):
        pa, _ = np.histogram(ha[:, ch], bins=bins, range=(0, 255), density=True)
        pb, _ = np.histogram(hb[:, ch], bins=bins, range=(0, 255), density=True)
        pa, pb = pa + 1e-8, pb + 1e-8
        sims.append(float(np.minimum(pa, pb).sum() / np.maximum(pa, pb).sum()))
    return sum(sims) / 3


def marker_survival(model, src, out, box, stride=40, scales=(0.7, 1.0, 1.35)):
    """Retrieval-style: max-over-locations DINO cosine of the source marker
    crop vs sliding output crops; also returns the HSV-histogram similarity of
    the best-matching crop (fine color-pattern check on top of semantics)."""
    src_crop = src.crop(box)
    q = dino_embed(model, [src_crop])          # [1, D]
    bw, bh = box[2] - box[0], box[3] - box[1]
    W, H = out.size
    crops, boxes = [], []
    for s in scales:
        cw, ch = int(bw * s), int(bh * s)
        if cw > W or ch > H:
            continue
        for y in range(0, H - ch + 1, stride):
            for x in range(0, W - cw + 1, stride):
                crops.append(out.crop((x, y, x + cw, y + ch)))
                boxes.append((x, y, x + cw, y + ch))
    best, best_i = 0.0, 0
    for i in range(0, len(crops), 64):
        e = dino_embed(model, crops[i:i + 64])  # [B, D]
        sims = (e @ q.T).squeeze(-1)
        j = int(sims.argmax())
        if float(sims[j]) > best:
            best, best_i = float(sims[j]), i + j
    csim = color_hist_sim(src_crop, crops[best_i]) if crops else 0.0
    return best, csim


def clip_t(out, prompt):
    from transformers import CLIPModel, CLIPProcessor
    global _CLIP
    if "_CLIP" not in globals():
        _CLIP = (CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval(),
                 CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"))
    model, proc = _CLIP
    inp = proc(text=[prompt], images=out, return_tensors="pt",
               padding=True, truncation=True).to(DEV)
    with torch.no_grad():
        o = model(**inp)
    img = F.normalize(o.image_embeds, dim=-1)
    txt = F.normalize(o.text_embeds, dim=-1)
    return float((img @ txt.T).squeeze())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = json.loads(Path(args.manifest).read_text())
    dino = load_dino()
    rows = []
    for c in cases:
        src = Image.open(c["source"]).convert("RGB").resize((1024, 1024))
        out = Image.open(c["output"]).convert("RGB").resize((1024, 1024))
        ms, mc = marker_survival(dino, src, out, tuple(c["marker_box"]))
        ct = clip_t(out, c["target_prompt"])
        # motion via CLIP direction: action vs its static counterpart —
        # shared tokens (object/outfit/scene) cancel, isolating the motion
        motion = ct - clip_t(out, c["static_prompt"]) if c.get("static_prompt") else None
        glob = float((dino_embed(dino, [src]) @ dino_embed(dino, [out]).T).squeeze())
        rows.append(dict(name=c["name"], marker_dino=round(ms, 4),
                         marker_color=round(mc, 4), clip_t=round(ct, 4),
                         motion=None if motion is None else round(motion, 4),
                         global_dino=round(glob, 4)))
        mstr = "  n/a " if motion is None else f"{motion:+.3f}"
        print(f"{c['name']:28s} mkr_dino={ms:.3f} mkr_color={mc:.3f} "
              f"motion={mstr} clip_t={ct:.3f} global={glob:.3f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print("saved", args.out)


if __name__ == "__main__":
    main()
