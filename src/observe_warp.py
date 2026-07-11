"""Solution-A Stage 1: OBSERVE the correspondence, don't let it stay emergent.

Given the source image and a layout draft (any output whose pose is right but
whose fine identity drifted — e.g. the single-pass result), recover the object
motion as an explicit affine map phi via DINO mutual-nearest-neighbor matches
+ RANSAC, then build in PIXEL space:

  composite = source background + warp_phi(source object)

plus the token masks Stage 2 needs (anchor region Omega, vacated region).
Warping in pixel space and re-encoding sidesteps latent non-equivariance:
the residual error class is only VAE quantization + bilinear interpolation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

SIZE = 1024


def dino_patches(model, img: Image.Image, device):
    """DINOv2 patch features on a 37x37 grid (518px, patch 14)."""
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.forward_features(x)["x_norm_patchtokens"][0]  # [1369, d]
    return F.normalize(feats, dim=-1), 37


def mnn_matches(fs, fd, src_keep, cos_min=0.55):
    """Mutual nearest neighbors between kept source patches and all draft
    patches. Returns (src_idx, dst_idx, cos)."""
    sim = fs[src_keep] @ fd.T                       # [S, P]
    fwd = sim.argmax(1)                             # best draft per source
    bwd = (fd @ fs[src_keep].T).argmax(1)           # best source per draft
    s_ids = torch.arange(len(src_keep), device=sim.device)
    mutual = bwd[fwd] == s_ids
    cos = sim[s_ids, fwd]
    ok = mutual & (cos >= cos_min)
    src_idx = torch.tensor(src_keep, device=sim.device)[ok]
    return src_idx.cpu(), fwd[ok].cpu(), cos[ok].cpu()


def ransac_affine(P, Q, iters=2000, tol=1.8, seed=0):
    """Robust 2D affine Q ~ A @ P + t on point sets [N,2] (patch coords)."""
    rng = np.random.default_rng(seed)
    N = len(P)
    if N < 4:
        raise SystemExit(f"too few matches ({N}) for affine fit")
    ones = np.ones((N, 1))
    Ph = np.hstack([P, ones])                       # [N,3]
    best_inl, best_M = None, None
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        try:
            M, *_ = np.linalg.lstsq(Ph[idx], Q[idx], rcond=None)  # [3,2]
        except np.linalg.LinAlgError:
            continue
        r = np.linalg.norm(Ph @ M - Q, axis=1)
        inl = r < tol
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_M = inl, M
    # final least-squares refit on inliers
    M, *_ = np.linalg.lstsq(Ph[best_inl], Q[best_inl], rcond=None)
    res = np.linalg.norm(Ph[best_inl] @ M - Q[best_inl], axis=1).mean()
    return M, best_inl, res


def warp_image(img: Image.Image, M, out_size=SIZE):
    """Inverse-warp img by affine M (source->draft, in [0,1] normalized
    coords) onto the draft frame with bilinear sampling."""
    A = M[:2].T                                     # [2,2]
    t = M[2]                                        # [2]
    Ainv = np.linalg.inv(A)
    x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    ys, xs = torch.meshgrid(torch.linspace(0, 1, out_size),
                            torch.linspace(0, 1, out_size), indexing="ij")
    tgt = torch.stack([xs, ys], -1).reshape(-1, 2).numpy()      # (x, y)
    src = (tgt - t) @ Ainv.T                                    # inverse map
    grid = torch.from_numpy(src.astype(np.float32)).reshape(1, out_size, out_size, 2)
    grid = grid * 2 - 1                                         # grid_sample coords
    out = F.grid_sample(x, grid, mode="bilinear", align_corners=False,
                        padding_mode="zeros")
    return out[0].permute(1, 2, 0).clamp(0, 1).numpy()


def patch_mask_to_pixel(idxs, grid, blur=12, dilate=2, close=3):
    """37x37 patch index set -> soft pixel mask [SIZE,SIZE] in [0,1].

    Articulation makes inlier support ragged (holes exactly where the local
    motion deviates from the fitted model) — close the support and keep the
    largest connected component before feathering."""
    m = np.zeros(grid * grid, dtype=np.float32)
    m[idxs] = 1.0
    m = m.reshape(grid, grid)

    def pool(a, k):
        return torch.nn.functional.max_pool2d(
            torch.from_numpy(a)[None, None], 2 * k + 1, stride=1, padding=k)[0, 0].numpy()

    if close > 0:
        m = 1.0 - pool(1.0 - pool(m, close), close)    # dilate then erode
    from scipy import ndimage
    lab, n = ndimage.label(m > 0.5)
    if n > 1:
        sizes = ndimage.sum(m > 0.5, lab, range(1, n + 1))
        m = (lab == (1 + int(np.argmax(sizes)))).astype(np.float32)
    if dilate > 0:
        m = pool(m, dilate)
    img = Image.fromarray((m * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.array(img, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--draft", required=True, help="layout draft (pose right, detail drifted)")
    ap.add_argument("--source-mask", required=True, help="rough box mask of the object in the source")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cos-min", type=float, default=0.55)
    ap.add_argument("--static-cos", type=float, default=0.97,
                    help="matches above this cos AND below --static-disp are "
                         "treated as copied background and dropped")
    ap.add_argument("--static-disp", type=float, default=1.0, help="patch units")
    ap.add_argument("--tol", type=float, default=1.8, help="RANSAC inlier tol (patch units)")
    ap.add_argument("--focus-box", required=True,
                    help="marker box 'l,t,r,b' in source pixels — the surface "
                         "whose fidelity must be exact")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    src = Image.open(args.source).convert("RGB").resize((SIZE, SIZE))
    drf = Image.open(args.draft).convert("RGB").resize((SIZE, SIZE))
    box = np.array(Image.open(args.source_mask).convert("L").resize((37, 37), Image.BILINEAR))

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
    fs, grid = dino_patches(model, src, device)
    fd, _ = dino_patches(model, drf, device)

    src_keep = np.where(box.reshape(-1) > 32)[0].tolist()
    si, di, cos = mnn_matches(fs, fd, src_keep, args.cos_min)
    print(f"[match] {len(si)} mutual matches (cos>={args.cos_min})")

    # patch coords in [0,1]
    def coords(idx):
        idx = idx.numpy()
        return np.stack([(idx % grid + 0.5) / grid, (idx // grid + 0.5) / grid], 1)

    P, Q = coords(si), coords(di)
    si_k = si.numpy()

    # sequential RANSAC: the dominant model over box matches is usually the
    # (near-identity) background alignment — peel it off, then fit the OBJECT
    # motion on the remainder. Without this the copied background hijacks
    # the fit and the recovered "motion" is a small global shift.
    tol = args.tol / grid          # tol is given in patch units; coords are in [0,1]
    M1, inl1, res1 = ransac_affine(P, Q, tol=tol)
    shift1 = np.linalg.norm(M1[2]) * grid
    A1 = M1[:2].T
    scale_dev = max(abs(np.linalg.norm(A1[:, 0]) - 1), abs(np.linalg.norm(A1[:, 1]) - 1))
    is_bg = shift1 < 1.5 and scale_dev < 0.08
    print(f"[fit1] inliers {inl1.sum()}/{len(P)} res {res1 * grid:.2f}p "
          f"shift {shift1:.2f} patch scale-dev {scale_dev:.2f} -> "
          f"{'background' if is_bg else 'object'}")
    if is_bg:
        rest = ~inl1
        if rest.sum() < 30:
            raise SystemExit(f"only {rest.sum()} non-background matches — "
                             "object did not move or matching failed")
        M, inl_r, res = ransac_affine(P[rest], Q[rest], tol=tol)
        inl = np.zeros(len(P), bool)
        inl[np.where(rest)[0][inl_r]] = True
        print(f"[fit2] object: inliers {inl.sum()}/{int(rest.sum())} res {res * grid:.2f}p "
              f"shift {M[2] * grid} patch")
    else:
        M, inl, res = M1, inl1, res1

    # object matches = everything the background model does not explain,
    # filtered for local displacement coherence (kills stray false matches);
    # articulation/rotation is then absorbed by an ELASTIC (thin-plate) warp
    # instead of one global affine.
    obj_sel = ~inl1 if is_bg else np.ones(len(P), bool)
    Po, Qo, so = P[obj_sel], Q[obj_sel], si_k[obj_sel]
    D = Qo - Po
    keep = np.ones(len(Po), bool)
    for a in range(len(Po)):
        d2 = np.linalg.norm(Po - Po[a], axis=1)
        nb = np.argsort(d2)[1:9]
        med = np.median(D[nb], axis=0)
        if np.linalg.norm(D[a] - med) * grid > 3.0:
            keep[a] = False
    Po, Qo, so = Po[keep], Qo[keep], so[keep]
    print(f"[flow] {len(Po)} coherent object matches "
          f"(dropped {int((~keep).sum())} incoherent)")
    if len(Po) < 20:
        raise SystemExit("too few coherent object matches for TPS")

    # LOCALLY-RIGID marker anchor: the goal is exact fidelity of the marked
    # surface, which is the least articulated part of the object. Fit ONE
    # affine on the matches around the marker box only (tight residual, no
    # melting, no part-blend ghosting) and paste just that support region.
    fb = np.array([float(v) for v in args.focus_box.split(",")]) / SIZE  # l,t,r,b
    pad = 2.0 / grid
    in_focus = ((Po[:, 0] >= fb[0] - pad) & (Po[:, 0] <= fb[2] + pad)
                & (Po[:, 1] >= fb[1] - pad) & (Po[:, 1] <= fb[3] + pad))
    Pf, Qf = Po[in_focus], Qo[in_focus]
    print(f"[focus] {len(Pf)} matches inside marker box")
    if len(Pf) < 12:
        raise SystemExit("too few matches around the marker box")
    Ph = np.hstack([Pf, np.ones((len(Pf), 1))])
    Mf, *_ = np.linalg.lstsq(Ph, Qf, rcond=None)
    res_f = np.linalg.norm(Ph @ Mf - Qf, axis=1).mean() * grid
    print(f"[focus] affine res {res_f:.2f}p shift {Mf[2] * grid} patch")

    warped = warp_image(src, Mf)
    # validity: only pixels whose inverse-mapped source point falls inside
    # the (padded) marker box may be pasted — never unrelated content
    A = Mf[:2].T
    t = Mf[2]
    Ainv = np.linalg.inv(A)
    ys, xs = np.meshgrid(np.linspace(0, 1, SIZE), np.linspace(0, 1, SIZE),
                         indexing="ij")
    tgt = np.stack([xs, ys], -1).reshape(-1, 2)
    back = (tgt - t) @ Ainv.T
    bpad = 0.03
    valid = ((back[:, 0] >= fb[0] - bpad) & (back[:, 0] <= fb[2] + bpad)
             & (back[:, 1] >= fb[1] - bpad) & (back[:, 1] <= fb[3] + bpad))
    valid = valid.reshape(SIZE, SIZE).astype(np.float32)
    valid = np.array(Image.fromarray((valid * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(8)), np.float32) / 255.0

    di_f = (Qf * grid).astype(int)
    idx_f = (np.clip(di_f[:, 1], 0, grid - 1) * grid
             + np.clip(di_f[:, 0], 0, grid - 1))
    support = patch_mask_to_pixel(idx_f, grid, blur=8, dilate=1, close=2)
    omega = support * valid
    drf_np = np.array(drf, dtype=np.float32) / 255.0
    composite = omega[..., None] * warped + (1 - omega[..., None]) * drf_np

    Image.fromarray((composite * 255).astype(np.uint8)).save(out / "composite.png")
    Image.fromarray((omega * 255).astype(np.uint8)).save(out / "omega_mask.png")
    (out / "warp.json").write_text(json.dumps(
        {"tps_matches": int(len(Po)), "bg_inliers": int(inl1.sum()),
         "affine_res_patch": float(res * grid)}, indent=1))
    print(f"[out] {out}/composite.png, omega_mask.png, warp.json")


if __name__ == "__main__":
    main()
