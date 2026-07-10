"""Regional attention processor for FLUX-Kontext.

Applies per-region bias on both the text tokens AND the ref-image tokens so that
each gen-region binds to its own text destination and doesn't copy its own ref
appearance.

Identity-preserve extension: an optional identity_ref_bias (positive,
cross-region gen→ref) opens an appearance pathway from a destination region's
gen tokens to a preserved object's ref tokens, gated to its own step window
and (via install's identity_layer_filter) its own layer subset.

Joint sequence layout (FluxKontext): [text(T_full) | gen(N) | ref(N)].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class RegionalMRASProcessor:
    def __init__(
        self,
        gen_indices: list[int],
        text_bias: torch.Tensor,   # [n_gen, T_full]
        ref_bias: torch.Tensor,    # [n_gen, N]
        apply_until_step: int = 28,
        N_total: int = 4096,
        identity_ref_bias: torch.Tensor | None = None,  # [n_gen, N], positive
        identity_from_step: int = 0,
        identity_until_step: int = 28,
    ) -> None:
        assert text_bias.shape[0] == len(gen_indices)
        assert ref_bias.shape == (len(gen_indices), N_total)
        if identity_ref_bias is not None:
            assert identity_ref_bias.shape == (len(gen_indices), N_total)
        self.gen_t = torch.tensor(gen_indices, dtype=torch.long)
        self.text_bias = text_bias
        self.ref_bias = ref_bias
        self.identity_ref_bias = identity_ref_bias
        self.apply_until = apply_until_step
        self.id_from = identity_from_step
        self.id_until = identity_until_step
        self.current_step = 0
        self._cache: dict = {}
        self._T_full: int | None = None
        self._N = N_total
        # id(attn_module) -> (do_suppress, do_identity), registered by install().
        # Default: suppress everywhere, identity everywhere it exists.
        self._roles: dict[int, tuple[bool, bool]] = {}
        self._ng_cache: dict = {}  # (T, N, device) -> non-gen row indices
        # Dynamic identity localization (Stage 3a): per preserve object,
        # the identity bias is gain-scaled per dest row by how much that row
        # attends the object's text span (the emerging silhouette), so
        # appearance does not leak into non-object parts of the dest region.
        self.dyn_specs: list[dict] = []   # {rows, ref_cols, span, beta}
        self.dyn_ema = 0.7
        self.dyn_gamma = 1.0
        self.dyn_floor = 0.0
        # Late-step identity ramp: multiply the in_ref (identity) bias by a
        # factor that grows from ramp_lo to ramp_hi over [ramp_from, ramp_to].
        # Early steps stay weak (pose forms from text, old pose not revived);
        # late steps go strong (pose is locked, so strong ref transports fine
        # detail from the reference WITHOUT reviving the old pose).
        self.id_ramp = False
        self.ramp_lo = 1.0
        self.ramp_hi = 1.0
        self.ramp_from = 0
        self.ramp_to = 28
        self.capture_entropy = False  # record within-object attention stats
        # correspondence-gated semantic routing: each gen row's binding to an
        # attribute phrase (e.g. "raincoat") is gated by the semantics of its
        # MATCHED source part — rows matched to garment-free source parts stop
        # hearing the garment words, so the text prior cannot paint sleeves
        # where the source instance has none. Source-part semantics come from
        # the ref rows' own affinity to the phrase (free, inside attention).
        self.sem_gate = None  # {"span": LongTensor, "kappa": float}
        self._garm = None     # per-call garmentness of object ref cols [C]

    def _ramp(self) -> float:
        if not self.id_ramp:
            return 1.0
        s = self.current_step
        if s <= self.ramp_from:
            return self.ramp_lo
        if s >= self.ramp_to:
            return self.ramp_hi
        f = (s - self.ramp_from) / max(1, self.ramp_to - self.ramp_from)
        return self.ramp_lo + (self.ramp_hi - self.ramp_lo) * f

    def _nongen_rows(self, T: int, N: int, device):
        """Joint-sequence rows NOT manually recomputed (text + non-gen image +
        ref). When the manual path runs, the fused SDPA only needs these rows —
        gen rows' fused output would be discarded."""
        key = (T, N, str(device))
        if key not in self._ng_cache:
            S = T + 2 * N
            mask = torch.ones(S, dtype=torch.bool)
            mask[T + self.gen_t] = False
            self._ng_cache[key] = mask.nonzero(as_tuple=True)[0].to(device)
        return self._ng_cache[key]

    @property
    def has_identity(self) -> bool:
        return self.identity_ref_bias is not None or len(self.dyn_specs) > 0

    def add_dynamic(self, rows, span, bias_cols, beta, in_ref: bool,
                    ema: float = 0.7, gamma: float = 1.0, floor: float = 0.0,
                    init_gain=None, topk: int = 0, ot=None):
        """Register a dynamic gain spec.

        rows      : gen-row indices the gain covers (dest region + canvas ring,
                    or all rows in full-canvas mode)
        span      : text token indices harvested to form the silhouette gain
        bias_cols : columns receiving gain*beta — ref token indices when
                    in_ref=True (identity), text span when in_ref=False (target)
        beta      : bias magnitude (0 = harvest only, e.g. for editedness)
        init_gain : optional per-row prior gain in [0,1] (e.g. 1 inside the
                    object's mask, lower elsewhere); default uniform 1
        """
        import torch as _t
        g0 = (_t.as_tensor(init_gain, dtype=_t.float32)
              if init_gain is not None else _t.ones(len(rows)))
        assert g0.shape == (len(rows),)
        self.dyn_specs.append({
            "rows": _t.tensor(rows, dtype=_t.long),
            "bias_cols": _t.tensor(sorted(bias_cols), dtype=_t.long),
            "span": _t.tensor(span, dtype=_t.long),
            "beta": float(beta),
            "in_ref": bool(in_ref),
            "gain": g0,
            "n_acc": 0,   # harvested layers this step (acc_dev holds the mass)
            # correspondence-sharpened transport: instead of a uniform bias over
            # ALL bias_cols (which only raises HOW MUCH is copied — the within-
            # object attention shape stays diffuse for displaced parts, so only
            # the mean/low-frequency appearance transports), amplify each row's
            # OWN top-k matches among bias_cols (the model's emergent gen->ref
            # correspondence), imposing a sharp structure that can carry
            # high-frequency instance detail.
            "topk": int(topk),
            # entropic optimal-transport transport plan (Sinkhorn) + value-space
            # barycentric injection. Row constraint kills averaging (low-pass),
            # COLUMN constraint (mass conservation) forces EVERY source part to
            # be transported — including "absence" regions (bare legs), which
            # row-only sharpening (topk) cannot enforce. dict(tau, lam, iters).
            "ot": dict(ot) if ot else None,
            "_ent": [],   # (step, mean_within_entropy, obj_mass, top8_share)
        })
        self.dyn_ema, self.dyn_gamma, self.dyn_floor = ema, gamma, floor

    def set_soft_background(self, rows, tokens, beta, text_suppress: float = 0.0,
                            halo: int = 0):
        """Soft attention background anchor (RePaint replacement): row i gets
        beta * (1 - editedness_i) bias toward its own-position ref token, AND
        -text_suppress * (1 - editedness_i) on ALL text columns — unclaimed
        rows keep their source content and stop listening to the prompt
        (RePaint's second, implicit role). Any object may still claim a row:
        rising editedness lifts both effects. rows/tokens aligned."""
        import torch as _t
        self._bg_rows = _t.tensor(rows, dtype=_t.long)
        self._bg_cols = _t.tensor(tokens, dtype=_t.long)  # ref col = own position
        self._bg_beta = float(beta)
        self._bg_txt = float(text_suppress)
        self._bg_halo = int(halo)
        self._bg_scale = _t.zeros(len(rows))
        # anchor must exist at step 0 (layout!) — seed from the specs' priors
        self._update_bg_scale()

    def _update_bg_scale(self):
        """Anchor scale = 1 - editedness. With halo > 0, the editedness map is
        grid-dilated (max-pool) first, leaving a growth band around forming
        objects — otherwise an object's size is capped by its first nucleus
        (the anchor pins every neighboring row)."""
        if getattr(self, "_bg_rows", None) is None:
            return
        e = self.row_editedness()
        if self._bg_halo > 0:
            G = int(self._N ** 0.5)
            eg = torch.zeros(self._N)
            eg[self.gen_t] = e
            k = 2 * self._bg_halo + 1
            eg = F.max_pool2d(eg.view(1, 1, G, G), kernel_size=k,
                              stride=1, padding=self._bg_halo).view(-1)
            self._bg_scale = (1.0 - eg[self._bg_cols]).clamp(0.0, 1.0)
        else:
            self._bg_scale = (1.0 - e[self._bg_rows]).clamp(0.0, 1.0)

    def add_dynamic_identity(self, rows, ref_cols, span, beta, **kw):
        self.add_dynamic(rows, span, ref_cols, beta, in_ref=True, **kw)

    def row_editedness(self):
        """Per-gen-row editedness in [0,1]: max dynamic gain over all specs.
        Used by the dynamic-canvas RePaint to re-anchor untouched rows."""
        import torch as _t
        e = _t.zeros(len(self.gen_t))
        for sp in self.dyn_specs:
            e[sp["rows"]] = _t.maximum(e[sp["rows"]], sp["gain"])
        return e

    def step_end(self):
        """Fold this step's harvested text-attention into the per-row gain
        (EMA over steps). Called from the step-end callback."""
        for sp in self.dyn_specs:
            if sp["n_acc"] == 0:
                continue
            acc = sp.pop("acc_dev", {})
            a = sum(t.cpu() for t in acc.values()) / sp["n_acc"]
            a = a / (a.max() + 1e-8)
            g_new = a.clamp(0, 1) ** self.dyn_gamma
            g_new = self.dyn_floor + (1 - self.dyn_floor) * g_new
            sp["gain"] = self.dyn_ema * sp["gain"] + (1 - self.dyn_ema) * g_new
            sp["n_acc"] = 0
            sp["_gain_dev"] = {}  # invalidate per-device gain cache
        self._update_bg_scale()
        self._bg_dev = {}

    def _get(self, device: torch.device):
        key = str(device)
        if key not in self._cache:
            irb = (self.identity_ref_bias.to(device)
                   if self.identity_ref_bias is not None else None)
            self._cache[key] = (
                self.gen_t.to(device),
                self.text_bias.to(device),
                self.ref_bias.to(device),
                irb,
            )
        return self._cache[key]

    def _active(self, attn) -> tuple[bool, bool]:
        do_s, do_i = self._roles.get(id(attn), (True, self.has_identity))
        s_active = do_s and self.current_step <= self.apply_until
        i_active = (do_i and self.has_identity
                    and self.id_from <= self.current_step <= self.id_until)
        return s_active, i_active

    def _apply(self, scores, gs, rs, T, gen_t, tb, rb, irb,
               s_active: bool, i_active: bool):
        # scores: [1, H, n_gen, total_len]
        if s_active:
            scores[:, :, :, :T] = scores[:, :, :, :T] + tb[None, None, :, :T]
            scores[:, :, :, rs:rs + self._N] = scores[:, :, :, rs:rs + self._N] + rb[None, None, :, :]
            if getattr(self, "_bg_rows", None) is not None:
                dev = scores.device
                key = str(dev)
                cached = getattr(self, "_bg_dev", None) or {}
                if key not in cached:
                    cached[key] = (self._bg_rows.to(dev),
                                   self._bg_cols.to(dev),
                                   self._bg_scale.to(dev, torch.float32))
                    self._bg_dev = cached
                br, bc0, sc = cached[key]
                scores[:, :, br, rs + bc0] += (sc * self._bg_beta)[None, None]
                if self._bg_txt > 0:
                    # unclaimed rows stop listening to the prompt
                    scores[:, :, br, :T] -= (sc * self._bg_txt)[None, None, :, None]
        if i_active:
            ramp = self._ramp()
            if irb is not None:
                scores[:, :, :, rs:rs + self._N] = scores[:, :, :, rs:rs + self._N] + ramp * irb[None, None, :, :]
            dev = scores.device
            key = str(dev)
            for si, sp in enumerate(self.dyn_specs):
                if sp["beta"] == 0.0:
                    continue  # harvest-only spec (editedness tracking)
                cached = sp.setdefault("_gain_dev", {})
                if key not in cached:
                    cached[key] = (sp["rows"].to(dev),
                                   sp["bias_cols"].to(dev),
                                   sp["gain"].to(dev, torch.float32))
                rows, cols, gain = cached[key]
                # ramp only the identity (in_ref) transport, not target text bias
                m = sp["beta"] * (ramp if sp["in_ref"] else 1.0)
                if sp["in_ref"] and sp["topk"] > 0:
                    # correspondence-sharpened: boost each row's own top-k
                    # matches among the object's ref columns (emergent
                    # correspondence), not the whole region uniformly.
                    # gain is renormalized robustly (90th pct, clamped): the
                    # max-normalized silhouette gain starves the transport —
                    # most object rows sit at ~0.2 and get e^0.5 (nothing).
                    cols_abs = rs + cols
                    k = min(sp["topk"], cols_abs.numel())
                    q = torch.quantile(gain, 0.9) + 1e-8
                    g = (gain / q).clamp(0.0, 1.0)
                    # [n_gen, C] head-mean scores over object ref cols
                    sub = scores[0, :, :, cols_abs].mean(0)
                    idx = sub[rows].topk(k, dim=1).indices        # [R, k]
                    tgt_cols = cols_abs[idx]                       # [R, k]
                    rows_e = rows[:, None].expand(-1, k)           # [R, k]
                    bias = (g[:, None] * m).expand(-1, k)          # [R, k]
                    scores[0, :, rows_e, tgt_cols] += bias
                    # correspondence-gated semantic routing: rows whose matched
                    # source parts are attribute-free (e.g. bare legs) stop
                    # hearing the attribute words — text prior cannot paint
                    # sleeves where the source instance has none
                    if self._garm is not None and si in self._garm:
                        gr = self._garm[si][idx].mean(1)           # [R] matched garment-ness
                        gspan = self.sem_gate["span"].to(dev)
                        sup = (-self.sem_gate["kappa"] * (1.0 - gr) * g)
                        scores[0, :, rows[:, None], gspan[None, :]] += sup[:, None][None]
                    continue
                if sp["in_ref"]:
                    cols = rs + cols
                bias = gain[:, None] * m
                scores[:, :, rows[:, None], cols[None, :]] += bias[None, None]
        return scores

    def _compute_garm(self, q_p, k_p, ci, rs):
        """Per object ref column: affinity of that SOURCE part to the gated
        attribute phrase (e.g. garment words). Rank-normalized to [0,1]."""
        if self.sem_gate is None:
            self._garm = None
            return
        dev = q_p.device
        span = self.sem_gate["span"].to(dev)
        self._garm = {}
        for si, sp in enumerate(self.dyn_specs):
            if not sp["in_ref"]:
                continue
            cols = rs + sp["bias_cols"].to(dev)
            q_ref = q_p[ci, :, cols, :].float()      # [H, C, D]
            k_g = k_p[ci, :, span, :].float()        # [H, G, D]
            aff = torch.einsum("hcd,hgd->hcg", q_ref, k_g).mean(dim=(0, 2))  # [C]
            a = aff - aff.min()
            self._garm[si] = a / (a.max() + 1e-8)

    def _ot_transport(self, out_gm, scores, v_all, rs):
        """Barycentric OT value injection:  o_i <- (1-λg_i) o_i + λg_i (P V)_i.

        P is an entropic-OT plan (Sinkhorn) between the object's gen rows and
        the object's ref columns: row sums = 1 (no averaging), column sums
        encouraged uniform (mass conservation — every source part, including
        garment-free skin, must land somewhere). Delivered in VALUE space, so
        it does not compete with text columns inside the softmax."""
        if not (self.id_from <= self.current_step <= self.id_until):
            return out_gm
        for sp in self.dyn_specs:
            ot = sp.get("ot")
            if not (sp["in_ref"] and ot):
                continue
            dev = scores.device
            key = str(dev)
            cached = sp.setdefault("_gain_dev", {})
            if key not in cached:
                cached[key] = (sp["rows"].to(dev),
                               sp["bias_cols"].to(dev),
                               sp["gain"].to(dev, torch.float32))
            rows, cols, gain = cached[key]
            q90 = torch.quantile(gain, 0.9) + 1e-8
            g = (gain / q90).clamp(0.0, 1.0)
            sel = g > 0.3                      # transport only object rows
            if int(sel.sum()) < 4:
                continue
            r_idx = rows[sel]
            cols_abs = rs + cols
            S = scores[0, :, :, cols_abs].mean(0)[r_idx].float()   # [R', C]
            K = ((S - S.max()) / ot["tau"]).exp() + 1e-12
            Rn, Cn = K.shape
            u = torch.ones(Rn, device=dev)
            c_marg = torch.full((Cn,), Rn / Cn, device=dev)
            for _ in range(ot["iters"]):
                v = c_marg / (K.t() @ u).clamp_min(1e-12)
                u = 1.0 / (K @ v).clamp_min(1e-12)
            P = u[:, None] * K * v[None, :]
            P = P / P.sum(1, keepdim=True).clamp_min(1e-12)        # rows sum 1
            v_obj = v_all[0, :, cols_abs, :].float()               # [H, C, D]
            t = torch.einsum("rc,hcd->hrd", P, v_obj)              # [H, R', D]
            lam = (ot["lam"] * g[sel]).to(out_gm.dtype)            # [R']
            o = out_gm[0, :, r_idx, :]
            out_gm[0, :, r_idx, :] = ((1 - lam)[None, :, None] * o
                                      + lam[None, :, None] * t.to(out_gm.dtype))
        return out_gm

    def _harvest(self, attn_w, i_active: bool):
        """Accumulate per-dest-row attention mass to each preserved object's
        text span from this layer's softmaxed weights [1, H, n_gen, total].

        Slice FIRST, cast after (casting the full matrix costs GBs per call);
        accumulate on-device and merge to CPU once per step in step_end()."""
        if not (i_active and self.dyn_specs):
            return
        dev = attn_w.device
        key = str(dev)
        for sp in self.dyn_specs:
            rows = sp["rows"].to(dev)
            span = sp["span"].to(dev)
            m = attn_w[0][:, rows][:, :, span].float().mean(0).sum(-1)  # [R]
            acc = sp.setdefault("acc_dev", {})
            if key in acc:
                acc[key] += m
            else:
                acc[key] = m
            sp["n_acc"] += 1
            if self.capture_entropy and sp["in_ref"]:
                # within-object attention stats over the object's ref columns:
                # how MUCH mass goes to the object (obj_mass) and how DIFFUSE
                # it is within the object (normalized entropy; 1 = uniform =
                # averaging = only low-frequency identity transports)
                rs = self._T_full + self._N
                cols_abs = rs + sp["bias_cols"].to(dev)
                # cols-first slice keeps the copy small ([H, n_gen, C] bf16)
                p = attn_w[0][:, :, cols_abs].float()[:, rows].mean(0)  # [R, C]
                obj_mass = p.sum(1)                                     # [R]
                pn = p / (obj_mass[:, None] + 1e-8)
                ent = -(pn * (pn + 1e-12).log()).sum(1)
                ent_norm = ent / float(torch.log(torch.tensor(float(p.shape[1]))))
                t8 = pn.topk(min(8, p.shape[1]), dim=1).values.sum(1)
                # restrict to rows the transport actually targets (gain>0.5):
                # all-row averages are diluted by background rows
                g = sp["gain"].to(ent_norm.device)
                hi = g > 0.5
                n_hi = int(hi.sum())
                if n_hi > 0:
                    sp["_ent"].append((self.current_step,
                                       float(ent_norm[hi].mean()),
                                       float(obj_mass[hi].mean()),
                                       float(t8[hi].mean()),
                                       n_hi, float(g.mean()), float(g.max())))
                else:
                    sp["_ent"].append((self.current_step,
                                       float(ent_norm.mean()),
                                       float(obj_mass.mean()),
                                       float(t8.mean()), 0,
                                       float(g.mean()), float(g.max())))

    def entropy_summary(self):
        """Aggregate per-spec within-object attention stats (capture_entropy)."""
        out = []
        for i, sp in enumerate(self.dyn_specs):
            if not sp["_ent"]:
                continue
            import numpy as _np
            a = _np.array(sp["_ent"])  # step, ent, obj_mass, top8, n_hi, g_mean, g_max
            half = a[:, 0].max() / 2 if len(a) else 0
            early = a[a[:, 0] <= half]
            late = a[a[:, 0] > half]
            out.append({
                "spec": i, "topk": sp["topk"], "beta": sp["beta"],
                "ent_norm": float(a[:, 1].mean()),
                "obj_mass": float(a[:, 2].mean()),
                "top8_share": float(a[:, 3].mean()),
                "ent_early": float(early[:, 1].mean()) if len(early) else None,
                "ent_late": float(late[:, 1].mean()) if len(late) else None,
                "n_hi_rows": float(a[:, 4].mean()),
                "gain_mean": float(a[:, 5].mean()),
                "gain_max": float(a[:, 6].mean()),
            })
        return out

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        from diffusers.models.transformers.transformer_flux import (
            _get_qkv_projections,
            apply_rotary_emb,
        )

        # ── Single-stream block ─────────────────────────────────
        if encoder_hidden_states is None:
            query, key, value, _, _, _ = _get_qkv_projections(attn, hidden_states, None)
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            qp = query.permute(0, 2, 1, 3)
            kp = key.permute(0, 2, 1, 3)
            vp = value.permute(0, 2, 1, 3)

            s_active, i_active = self._active(attn)
            apply = (
                (s_active or i_active)
                and len(self.gen_t) > 0
                and self._T_full is not None
            )
            B = qp.shape[0]
            if apply and B == 1:
                # gen rows are recomputed manually below — fused SDPA only for
                # the remaining rows (their fused output would be discarded)
                ng = self._nongen_rows(self._T_full, self._N, qp.device)
                full_out = torch.empty_like(qp)
                full_out[:, :, ng, :] = F.scaled_dot_product_attention(
                    qp[:, :, ng, :], kp, vp)
            else:
                full_out = F.scaled_dot_product_attention(qp, kp, vp)

            if apply:
                T, N = self._T_full, self._N
                gs = T
                rs = T + N
                device = qp.device
                gen_t, tb, rb, irb = self._get(device)
                gen_rows = gs + gen_t

                ci = B - 1
                q_gm = qp[ci:ci + 1, :, gen_rows, :]
                k_all = kp[ci:ci + 1]
                v_all = vp[ci:ci + 1]
                D_h = q_gm.shape[-1]
                scale = D_h ** -0.5
                scores = (q_gm.float() @ k_all.float().transpose(-2, -1)) * scale
                self._compute_garm(qp, kp, ci, rs)
                scores = self._apply(scores, gs, rs, T, gen_t, tb, rb, irb,
                                     s_active, i_active)
                attn_w = torch.softmax(scores, dim=-1).to(qp.dtype)
                self._harvest(attn_w, i_active)
                out_gm = attn_w @ v_all
                if i_active:
                    out_gm = self._ot_transport(out_gm, scores, v_all, rs)
                full_out[ci:ci + 1, :, gen_rows, :] = out_gm

            return full_out.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)

        # ── Double-stream block ─────────────────────────────────
        self._T_full = encoder_hidden_states.shape[1]

        query, key, value, eq, ek, ev = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        for t in [query, key, value, eq, ek, ev]:
            t.__class__ = torch.Tensor

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        eq = eq.unflatten(-1, (attn.heads, -1))
        ek = ek.unflatten(-1, (attn.heads, -1))
        ev = ev.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        eq = attn.norm_added_q(eq)
        ek = attn.norm_added_k(ek)

        jq = torch.cat([eq, query], dim=1)
        jk = torch.cat([ek, key], dim=1)
        jv = torch.cat([ev, value], dim=1)
        if image_rotary_emb is not None:
            jq = apply_rotary_emb(jq, image_rotary_emb, sequence_dim=1)
            jk = apply_rotary_emb(jk, image_rotary_emb, sequence_dim=1)

        jq_p = jq.permute(0, 2, 1, 3)
        jk_p = jk.permute(0, 2, 1, 3)
        jv_p = jv.permute(0, 2, 1, 3)

        s_active, i_active = self._active(attn)
        apply = (
            (s_active or i_active)
            and len(self.gen_t) > 0
        )
        T_full = encoder_hidden_states.shape[1]
        N_img = hidden_states.shape[1] // 2
        B = jq_p.shape[0]
        if apply and B == 1:
            ng = self._nongen_rows(T_full, N_img, jq_p.device)
            full_out = torch.empty_like(jq_p)
            full_out[:, :, ng, :] = F.scaled_dot_product_attention(
                jq_p[:, :, ng, :], jk_p, jv_p)
        else:
            full_out = F.scaled_dot_product_attention(jq_p, jk_p, jv_p)

        if apply:
            gs = T_full
            rs = T_full + N_img
            device = jq_p.device
            gen_t, tb, rb, irb = self._get(device)
            gen_rows = gs + gen_t

            ci = B - 1
            k_all = jk_p[ci:ci + 1]
            v_all = jv_p[ci:ci + 1]
            q_gm = jq_p[ci:ci + 1, :, gen_rows, :]
            D_h = q_gm.shape[-1]
            scale = D_h ** -0.5
            scores = (q_gm.float() @ k_all.float().transpose(-2, -1)) * scale
            self._compute_garm(jq_p, jk_p, ci, rs)
            scores = self._apply(scores, gs, rs, T_full, gen_t, tb, rb, irb,
                                 s_active, i_active)
            attn_w = torch.softmax(scores, dim=-1).to(jq_p.dtype)
            self._harvest(attn_w, i_active)
            out_gm = attn_w @ v_all
            if i_active:
                out_gm = self._ot_transport(out_gm, scores, v_all, rs)
            full_out[ci:ci + 1, :, gen_rows, :] = out_gm

        full_out = full_out.permute(0, 2, 1, 3).flatten(2, 3).to(jq.dtype)
        T_p = encoder_hidden_states.shape[1]
        enc_out = full_out[:, :T_p]
        img_out = full_out[:, T_p:]
        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        enc_out = attn.to_add_out(enc_out)
        return img_out, enc_out


def _layer_global_index(name: str) -> int:
    """Map layer name → global index in FLUX 57-layer ordering.

    Double-stream blocks are indexed 0..18, single-stream are 19..56.
    Ordering matches FreeFlux's paper convention.
    """
    import re
    m = re.search(r"transformer_blocks\.(\d+)\.attn", name)
    if not m:
        return -1
    idx = int(m.group(1))
    if "single_transformer_blocks" in name:
        return 19 + idx
    return idx


def install(pipe, processor, layer_filter: list[int] | None = None,
            identity_layer_filter: list[int] | None = None):
    """Install processor on all attention blocks, optionally filtered by
    global layer index (0..56).

    layer_filter          : layers where suppression biases apply (None = all).
    identity_layer_filter : layers where the identity boost applies (None = all);
                            only meaningful if processor.identity_ref_bias is set.

    A layer is installed if it belongs to either set; its per-mechanism role is
    registered on the processor so a single shared instance can behave
    differently per layer. Layers in NEITHER set keep their default processor.
    """
    has_identity = getattr(processor, "has_identity", None) or (
        getattr(processor, "identity_ref_bias", None) is not None)
    original = {}
    installed_sup, installed_id = [], []
    for name, module in pipe.transformer.named_modules():
        if not (name.endswith(".attn") and hasattr(module, "set_processor")):
            continue
        is_double = "transformer_blocks" in name and "single_transformer_blocks" not in name
        is_single = "single_transformer_blocks" in name
        if not (is_double or is_single):
            continue
        gidx = _layer_global_index(name)
        do_s = layer_filter is None or gidx in layer_filter
        do_i = has_identity and (identity_layer_filter is None
                                 or gidx in identity_layer_filter)
        if not (do_s or do_i):
            continue
        original[name] = module.processor
        module.set_processor(processor)
        if hasattr(processor, "_roles"):
            processor._roles[id(module)] = (do_s, do_i)
        if do_s:
            installed_sup.append(gidx)
        if do_i:
            installed_id.append(gidx)
    n_d = sum(1 for n in original if "single_transformer_blocks" not in n)
    n_s = len(original) - n_d
    print(f"[Regional] {len(original)} layers installed (double={n_d}, single={n_s})")
    if layer_filter is not None:
        print(f"[Regional]   suppression on {len(installed_sup)}: {sorted(installed_sup)}")
    if has_identity:
        print(f"[Regional]   identity boost on {len(installed_id)}: {sorted(installed_id)}")
    return original
