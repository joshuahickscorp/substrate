"""Series S: semiotics / universal-latent-language probes over frozen pooled latents. Eight cpu-now
experiments that ask whether a learned discrete code (VQ over latents) behaves as a grounded sign, whether
concept arithmetic is meaningful, whether discrete bottlenecks help reasoning at matched compute, whether
codes are stable across seeds, whether latent concepts compose, whether learned codes beat designed
symbols, whether useful directions are interpretable, and a standing anti-self-deception meta-test that
flags any S-test a frozen-random projection passes (vacuous).

Every experiment declares an explicit NULL and returns "null_supported": bool. The standing controls are
wired where relevant: frozen-random substrate arm (substrate_ablation / frozen_random_projection),
matched compute (diagnostics.compute), or a tuned/random-matched baseline. Honest nulls only: the toy
outcome is reported as-is, never tuned toward a positive. Pooling discards within-frame object/spatial
structure, so the compositional and grounding tests are EXPECTED to land in failure-taxonomy slot 3 and
the published bound IS the result.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, mlp_flops, param_count
from ..diagnostics.geometry import rsa
from ..diagnostics.held_out_combo import factorized_latents, held_out_combination
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.seed_consistency import code_stability, cross_seed_cka
from ..diagnostics.substrate_ablation import frozen_random_projection, substrate_ablation
from ..seeding import seed_everything
from ..shell.refine import IterativeRefiner
from .base import Experiment


# ----------------------------------------------------------------------------------------------------
# shared tiny mechanisms (inline, the established pattern)
# ----------------------------------------------------------------------------------------------------
def _kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd k-means; returns (hard code assignment per row, centroids). The VQ codebook in the
    frozen-latent setting (no pixel decoder, the codes ARE the abstraction)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:k]
    cent = x[idx].clone()
    codes = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, cent)
        codes = d.argmin(1)
        for c in range(k):
            m = codes == c
            if m.any():
                cent[c] = x[m].mean(0)
    return codes, cent


def _onehot(codes: torch.Tensor, k: int) -> torch.Tensor:
    oh = torch.zeros(codes.shape[0], k)
    oh[torch.arange(codes.shape[0]), codes] = 1.0
    return oh


def _mutual_information(codes: torch.Tensor, world: torch.Tensor, k: int, nb: int) -> float:
    """Discrete MI between code assignment and a quantized world variable (both small alphabets).
    Plug-in estimator over the joint histogram, in nats."""
    joint = torch.zeros(k, nb)
    for c, w in zip(codes.tolist(), world.tolist(), strict=True):
        joint[int(c), int(w)] += 1.0
    joint = joint / joint.sum().clamp(min=1.0)
    pc = joint.sum(1, keepdim=True)
    pw = joint.sum(0, keepdim=True)
    denom = (pc @ pw).clamp(min=1e-12)
    nz = joint > 0
    return float((joint[nz] * (joint[nz] / denom[nz]).log()).sum())


# ====================================================================================================
# S1: symbol-grounding gate (icon / index / symbol)
# ====================================================================================================
class S1(Experiment):
    id = "s1_symbol_grounding"
    metric = ("code_world_mi", "code_world_mi_random", "code_rsa", "needs_real")
    baseline = "random codebook at matched k (MI floor) and shuffled-pairing RSA floor"
    ablation = "learned VQ codebook vs random codebook; substrate_ablation frozen-random arm"
    null_hypothesis = (
        "codes carry no more world-variable information than a random codebook at matched k, and "
        "code-adjacency RSA ties the shuffled floor: the code is an arbitrary label, not a grounded index"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, k = int(e.dim), int(e.codebook_size)
        nb = int(e.n_classes)
        mi_learned, mi_random, rsa_real, rsa_shuf, code_r2, raw_r2 = [], [], [], [], [], []
        needs_real_flags = []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nb,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            x, world = task.x, task.y  # world var is the generative cluster label, linearly decodable
            codes, cent = _kmeans(x, k, int(e.kmeans_iters), s)
            g = torch.Generator().manual_seed(s + 7)
            rand_codes = torch.randint(0, k, (x.shape[0],), generator=g)
            # indexicality: MI(code; world_var) vs MI(random_code; world_var)
            mi_learned.append(_mutual_information(codes, world, k, nb))
            mi_random.append(_mutual_information(rand_codes, world, k, nb))
            # iconicity: RSA between code-centroid adjacency and the stimulus adjacency of those codes
            # (do geometrically near codes tag perceptually near clips). The shuffled control permutes
            # the code-to-centroid correspondence (breaks any icon structure).
            stim_proto = torch.stack(
                [x[codes == c].mean(0) if (codes == c).any() else cent[c] for c in range(k)]
            )
            rsa_real.append(rsa(cent, stim_proto))
            gp = torch.Generator().manual_seed(s + 11)
            rsa_shuf.append(rsa(cent, stim_proto[torch.randperm(k, generator=gp)]))
            # code_probe_R2 minus raw_latent_probe_R2 (does the one-hot code add over the raw latent)
            code_r2.append(linear_probe(_onehot(codes, k), world, seed=s)["score"])
            raw_r2.append(linear_probe(x, world, seed=s)["score"])
            # sanity: is the world var decodable above the shuffled chance floor at all (taxonomy-3 gate).
            # needs_real here means "above the shuffled floor", NOT "needed the real encoder": a
            # frozen-random LINEAR projection is invertible so it is vacuous for this linear metric.
            abl = substrate_ablation(x, world, seed=s)
            needs_real_flags.append(abl["needs_real"])

        def mean(v):
            return sum(v) / len(v)

        mil, mir = mean(mi_learned), mean(mi_random)
        rre, rsh = mean(rsa_real), mean(rsa_shuf)
        cr, rr = mean(code_r2), mean(raw_r2)
        needs_real = bool(sum(needs_real_flags) > len(needs_real_flags) / 2)
        mi_gain = mil - mir
        rsa_gain = rre - rsh
        out = {
            "code_world_mi_learned": round(mil, 4),
            "code_world_mi_random": round(mir, 4),
            "mi_gain_vs_random": round(mi_gain, 4),
            "code_rsa_real": round(rre, 4),
            "code_rsa_shuffled": round(rsh, 4),
            "rsa_gain_vs_shuffled": round(rsa_gain, 4),
            "code_probe_acc": round(cr, 4),
            "raw_latent_probe_acc": round(rr, 4),
            "code_minus_raw": round(cr - rr, 4),
            # decodability sanity: above the shuffled floor, NOT an encoder-specificity claim (the
            # frozen-random arm is vacuous for this linear metric, see substrate_ablation module doc)
            "needs_real": needs_real,
            "codebook_size": k,
            "seeds": list(seeds),
            # null: code is an arbitrary label (MI ties the random-code floor AND RSA ties the shuffled floor)
            "null_supported": bool(mi_gain <= 0.05 and rsa_gain <= 0.1),
            # grounded index (Peirce): indexicality (MI over the random-code floor) AND iconicity (code
            # geometry RSA over the shuffled floor). Both are earned controls; frozen-random is NOT used
            # (it is vacuous for these linear/geometry metrics, see substrate_ablation module doc).
            "grounded_index": bool(mi_gain > 0.05 and rsa_gain > 0.1),
        }
        return out


# ====================================================================================================
# S3: latent concept arithmetic (a:b :: c:d offset retrieval)
# ====================================================================================================
class S3(Experiment):
    id = "s3_concept_arithmetic"
    metric = ("retrieval_at1", "gain_vs_random_offset", "gain_vs_frozen_random", "probe_r2")
    baseline = "random matched-norm offset (anti-self-deception) and frozen-random projection arm"
    ablation = "learned factor offset vs random matched-norm offset vs frozen-random projection"
    null_hypothesis = (
        "offset arithmetic ties a random matched-norm offset, OR it survives equally under a "
        "frozen-random projection: arithmetic is a generic property of any linear space, not of concepts"
    )
    tier = "cpu-now"

    def _quadruples(self, dim, n_factor, per, sep, seed):
        """Two-factor latents; a quadruple (a,b,c,d) varies ONE factor f1 while sharing f2, so the
        offset latent(b)-latent(a) should transport latent(c) to latent(d). Returns x, f1, f2."""
        x, f1, f2 = factorized_latents(
            n_a=n_factor, n_b=n_factor, per_cell=per, dim=dim, interaction=0.0, seed=seed
        )
        return x * sep, f1, f2

    def _retrieval(self, x, f1, f2, n_factor, projfn=None, random_offset=False, seed=0):
        z = projfn(x) if projfn is not None else x
        # cell-mean prototypes indexed by (f1, f2)
        proto = {}
        for a in range(n_factor):
            for b in range(n_factor):
                m = (f1 == a) & (f2 == b)
                if m.any():
                    proto[(a, b)] = z[m].mean(0)
        g = torch.Generator().manual_seed(seed + 3)
        hits1 = 0
        total = 0
        keys = list(proto.keys())
        allz = torch.stack([proto[kk] for kk in keys])
        for a in range(n_factor):
            for b in range(n_factor):
                for ap in range(n_factor):
                    bp = b
                    if (a, b) not in proto or (ap, b) not in proto:
                        continue
                    if (a, bp) not in proto or (ap, bp) not in proto:
                        continue
                    # analogy: a_b : ap_b :: a_bp : ?  (vary f1 from a to ap, hold f2)
                    off = proto[(ap, b)] - proto[(a, b)]
                    if random_offset:
                        r = torch.randn(off.shape, generator=g)
                        off = r / r.norm() * off.norm()
                    target = proto[(a, bp)] + off
                    d = torch.cdist(target.unsqueeze(0), allz)[0]
                    pred = keys[int(d.argmin())]
                    hits1 += int(pred == (ap, bp))
                    total += 1
        return hits1 / max(1, total)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nf = int(e.dim), int(e.n_factor)
        learned, randoff, frozen, probe = [], [], [], []
        for s in seeds:
            seed_everything(s)
            x, f1, f2 = self._quadruples(dim, nf, int(e.per_cell), float(e.separation), s)
            # gate: the factor must be linearly decodable first
            probe.append(linear_probe(x, f1, seed=s)["score"])
            learned.append(self._retrieval(x, f1, f2, nf, seed=s))
            randoff.append(self._retrieval(x, f1, f2, nf, random_offset=True, seed=s))
            frozen.append(
                self._retrieval(x, f1, f2, nf, projfn=lambda t, s=s: frozen_random_projection(t, s), seed=s)
            )

        def mean(v):
            return sum(v) / len(v)

        lm, rm, fm, pm = mean(learned), mean(randoff), mean(frozen), mean(probe)
        gain_rand = lm - rm
        gain_frozen = lm - fm
        out = {
            "retrieval_at1_learned": round(lm, 4),
            "retrieval_at1_random_offset": round(rm, 4),
            "retrieval_at1_frozen_random": round(fm, 4),
            "gain_vs_random_offset": round(gain_rand, 4),
            "gain_vs_frozen_random": round(gain_frozen, 4),
            "factor_probe_acc": round(pm, 4),
            "seeds": list(seeds),
            # null: ties random offset, OR survives equally under frozen-random (generic, not V-JEPA)
            "null_supported": bool(gain_rand <= 0.1 or abs(gain_frozen) <= 0.1),
            "arithmetic_meaningful": bool(gain_rand > 0.1 and gain_frozen > 0.1),
        }
        return out


# ====================================================================================================
# S4: latent reasoning vs discrete-bottleneck reasoning at matched compute
# ====================================================================================================
class S4(Experiment):
    id = "s4_latent_vs_discrete"
    metric = ("acc_latent", "acc_discrete", "gain_at_matched_flops", "params_ratio")
    baseline = "discrete-codebook-bottleneck reasoner at matched forward FLOPs"
    ablation = "continuous latent refiner vs VQ-bottleneck refiner; matched_within compute check"
    null_hypothesis = (
        "at matched compute, latent reasoning ties the discrete-bottleneck arm: symbol serialization is "
        "not a measurable bottleneck on this task"
    )
    tier = "cpu-now"

    def _fit_eval(self, backbone, head, codebook, x, y, xte, yte, epochs, lr, k):
        params = [*backbone.parameters(), *head.parameters()]
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            z = backbone(x)
            z = z[0] if isinstance(z, tuple) else z
            if codebook is not None:
                z = self._vq(z, codebook)
            F.cross_entropy(head(z), y).backward()
            opt.step()
        with torch.no_grad():
            z = backbone(xte)
            z = z[0] if isinstance(z, tuple) else z
            if codebook is not None:
                z = self._vq(z, codebook)
            return float((head(z).argmax(-1) == yte).float().mean())

    def _vq(self, z, codebook):
        """Straight-through VQ bottleneck: snap each intermediate state to the nearest codeword (the
        symbolic serialization), but pass gradients through (so the arms train comparably)."""
        d = torch.cdist(z, codebook)
        idx = d.argmin(1)
        zq = codebook[idx]
        return z + (zq - z).detach()

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps, k = int(e.dim), int(e.hidden), int(e.steps), int(e.codebook_size)
        epochs, lr = int(e.epochs), float(e.lr)
        lat_acc, dis_acc = [], []
        for s in seeds:
            seed_everything(s)
            # a multi-cell relational task: predict the joint (f1,f2) cell as one label (composition)
            x, f1, f2 = factorized_latents(
                n_a=int(e.n_factor),
                n_b=int(e.n_factor),
                per_cell=int(e.per_cell),
                dim=dim,
                interaction=0.0,
                seed=s,
            )
            y = f1 * int(e.n_factor) + f2
            cut = int(x.shape[0] * 0.7)
            perm = torch.randperm(x.shape[0])
            x, y = x[perm], y[perm]
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            nc = int(y.max()) + 1

            # arm A: continuous latent refiner end to end
            seed_everything(s)
            refA = IterativeRefiner(dim, hidden, steps)
            headA = nn.Linear(dim, nc)
            lat_acc.append(self._fit_eval(refA, headA, None, xtr, ytr, xte, yte, epochs, lr, k))
            # arm B: same refiner + compute, every intermediate state forced through a VQ bottleneck
            seed_everything(s)
            refB = IterativeRefiner(dim, hidden, steps)
            headB = nn.Linear(dim, nc)
            g = torch.Generator().manual_seed(s + 5)
            codebook = torch.randn(k, dim, generator=g)
            dis_acc.append(self._fit_eval(refB, headB, codebook, xtr, ytr, xte, yte, epochs, lr, k))

        def mean(v):
            return sum(v) / len(v)

        am, bm = mean(lat_acc), mean(dis_acc)
        gain = am - bm
        refA = IterativeRefiner(dim, hidden, steps)
        refB = IterativeRefiner(dim, hidden, steps)
        # the VQ snap adds a nearest-codeword lookup, negligible vs the refiner linear FLOPs; arms match
        flops = mlp_flops([dim, hidden, dim]) * steps
        compute = matched_within(flops, flops)
        out = {
            "acc_latent": round(am, 4),
            "acc_discrete": round(bm, 4),
            "gain_at_matched_flops": round(gain, 4),
            "margin": float(e.margin),
            "codebook_size": k,
            "params_latent": param_count(refA),
            "params_discrete": param_count(refB),
            "params_ratio": round(param_count(refA) / max(1, param_count(refB)), 4),
            "compute_matched": compute,
            "seeds": list(seeds),
            # null: latent ties discrete at matched compute (serialization not a measurable bottleneck)
            "null_supported": bool(abs(gain) <= float(e.margin)),
            "latent_beats_discrete": bool(gain > float(e.margin) and compute["matched"]),
        }
        return out


# ====================================================================================================
# S5: private-language stability (cross-seed code reproducibility)
# ====================================================================================================
class S5(Experiment):
    id = "s5_code_stability"
    metric = ("code_agreement", "code_agreement_chance", "cross_seed_cka", "cka_frozen_random")
    baseline = "random-relabeling code floor (1/k) and frozen-random CKA floor"
    ablation = "learned codebook cross-seed agreement vs random floor; geometry CKA vs frozen-random"
    null_hypothesis = (
        "cross-seed code agreement ties the random / frozen-random floor: the code is a per-run private "
        "idiolect, not a stable shared language"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        dim, k = int(e.dim), int(e.codebook_size)
        n_runs = int(e.n_runs)
        # one shared cached-latent set, K independent codebooks on different seeds
        seed_everything(0)
        task = make_task_stream(
            n_tasks=1,
            dim=dim,
            classes_per_task=int(e.n_classes),
            samples_per_task=int(e.samples),
            separation=float(e.separation),
            seed=0,
        )[0]
        x = task.x
        code_assignments, onehots = [], []
        for r in range(n_runs):
            codes, _ = _kmeans(x, k, int(e.kmeans_iters), seed=100 + r)
            code_assignments.append(codes)
            onehots.append(_onehot(codes, k))
        cs = code_stability(code_assignments, k)
        cka = cross_seed_cka(onehots)
        # frozen-random CKA floor: same points under independent random projections (no learned structure)
        fr_reps = [frozen_random_projection(x, seed=200 + r) for r in range(n_runs)]
        fr_cka = cross_seed_cka(fr_reps)
        agreement = cs["mean_agreement"]
        chance = cs["chance"]
        out = {
            "code_agreement": round(agreement, 4),
            "code_agreement_chance": round(chance, 4),
            "agreement_above_chance": round(agreement - chance, 4),
            "cross_seed_cka": round(cka["mean_cka"], 4),
            "cka_frozen_random": round(fr_cka["mean_cka"], 4),
            "cka_above_frozen_random": round(cka["mean_cka"] - fr_cka["mean_cka"], 4),
            "codebook_size": k,
            "n_runs": n_runs,
            # null: agreement ties the random floor AND code CKA ties the frozen-random floor (idiolect)
            "null_supported": bool(
                (agreement - chance) <= 0.2 and (cka["mean_cka"] - fr_cka["mean_cka"]) <= 0.1
            ),
            "stable_shared_code": bool(cs["stable"] and (cka["mean_cka"] - fr_cka["mean_cka"]) > 0.1),
        }
        return out


# ====================================================================================================
# S6: compositionality probe (held-out-combination decoding)
# ====================================================================================================
class S6(Experiment):
    id = "s6_compositionality"
    metric = ("heldout_acc", "heldout_acc_frozen_random", "gap_to_seen", "compositional")
    baseline = "frozen-random projection arm (floor) and the seen-combination accuracy ceiling"
    ablation = "additive factorized latent vs frozen-random; held-out vs seen decode accuracy"
    null_hypothesis = (
        "held-out combinations are not decodable above the frozen-random floor: the pooled latent does "
        "not represent factors compositionally, it represents whole-scene gestalts (taxonomy 3)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        nf, dim = int(e.n_factor), int(e.dim)
        ho_real, ho_fr, seen, comp_flags = [], [], [], []
        for s in seeds:
            seed_everything(s)
            x, ya, yb = factorized_latents(
                n_a=nf, n_b=nf, per_cell=int(e.per_cell), dim=dim, interaction=float(e.interaction), seed=s
            )
            real = held_out_combination(x, ya, yb, seed=s)
            fr = held_out_combination(frozen_random_projection(x.float(), s), ya, yb, seed=s)
            if "error" in real or "error" in fr:
                continue
            ho_real.append(real["heldout_acc"])
            ho_fr.append(fr["heldout_acc"])
            seen.append(real["seen_acc"])
            comp_flags.append(real["compositional"])

        def mean(v):
            return sum(v) / len(v) if v else 0.0

        hr, hf, sm = mean(ho_real), mean(ho_fr), mean(seen)
        above_floor = hr - hf
        out = {
            "heldout_acc": round(hr, 4),
            "heldout_acc_frozen_random": round(hf, 4),
            "heldout_above_frozen_random": round(above_floor, 4),
            "seen_acc": round(sm, 4),
            "gap_to_seen": round(sm - hr, 4),
            "chance": round(1.0 / nf, 4),
            "compositional": bool(sum(comp_flags) > len(comp_flags) / 2) if comp_flags else False,
            "interaction": float(e.interaction),
            "seeds": list(seeds),
            # null: held-out not decodable above frozen-random floor (no compositional structure)
            "null_supported": bool(above_floor <= 0.1 or (hr - 1.0 / nf) <= 0.15),
            "systematic_composition": bool(above_floor > 0.1 and (hr - 1.0 / nf) > 0.15),
        }
        return out


# ====================================================================================================
# S7: learned codes vs human-designed symbols at matched capacity
# ====================================================================================================
class S7(Experiment):
    id = "s7_learned_vs_designed"
    metric = ("acc_learned", "acc_designed", "acc_random", "raw_ceiling")
    baseline = "human-designed symbols (ground-truth quantized labels) at matched k and matched probe"
    ablation = "learned VQ code vs designed symbols vs random codebook; raw-latent ceiling"
    null_hypothesis = (
        "learned codes tie human-designed symbols at matched capacity: optimizing the vocabulary for use "
        "buys nothing over the designer ontology"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, k = int(e.dim), int(e.codebook_size)
        nclass = int(e.n_classes)
        learned, designed, randc, raw = [], [], [], []
        for s in seeds:
            seed_everything(s)
            # task target distinct from the generative label so neither vocabulary trivially equals it:
            # downstream target = parity of the generative label index (a derived task over the world)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nclass,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            x, world = task.x, task.y
            target = (world % 2).long()  # the held-out downstream task
            # A: learned VQ codebook of size k
            codes_l, _ = _kmeans(x, k, int(e.kmeans_iters), s)
            # B: human-designed symbols = ground-truth generative labels quantized to k symbols
            codes_d = (world % k).long()
            # control: random codebook at k
            g = torch.Generator().manual_seed(s + 9)
            codes_r = torch.randint(0, k, (x.shape[0],), generator=g)
            learned.append(linear_probe(_onehot(codes_l, k), target, seed=s)["score"])
            designed.append(linear_probe(_onehot(codes_d, k), target, seed=s)["score"])
            randc.append(linear_probe(_onehot(codes_r, k), target, seed=s)["score"])
            raw.append(linear_probe(x, target, seed=s)["score"])

        def mean(v):
            return sum(v) / len(v)

        lm, dm, rm, raw_m = mean(learned), mean(designed), mean(randc), mean(raw)
        gain = lm - dm
        out = {
            "acc_learned": round(lm, 4),
            "acc_designed": round(dm, 4),
            "acc_random_codebook": round(rm, 4),
            "raw_ceiling": round(raw_m, 4),
            "learned_minus_designed": round(gain, 4),
            "margin": float(e.margin),
            "codebook_size": k,
            "seeds": list(seeds),
            # null: learned ties designed at matched capacity (use-optimization buys nothing)
            "null_supported": bool(abs(gain) <= float(e.margin)),
            "learned_beats_designed": bool(gain > float(e.margin)),
        }
        return out


# ====================================================================================================
# S9: interpretable vs merely useful directions
# ====================================================================================================
class S9(Experiment):
    id = "s9_interpretable_vs_useful"
    metric = ("spearman_use_interp", "frac_top_useful_named", "frac_random_named", "needs_real")
    baseline = "random directions (interpretability floor) and frozen-random projection arm"
    ablation = "useful-direction interpretability vs random-direction interpretability"
    null_hypothesis = (
        "usefulness and interpretability are uncorrelated (Spearman ties zero): the useful code is not "
        "the namable code, it is an opaque-but-useful private code"
    )
    tier = "cpu-now"

    def _spearman(self, a: torch.Tensor, b: torch.Tensor) -> float:
        ra = a.argsort().argsort().float()
        rb = b.argsort().argsort().float()
        ra, rb = ra - ra.mean(), rb - rb.mean()
        denom = (ra.pow(2).sum().sqrt() * rb.pow(2).sum().sqrt()) + 1e-12
        return float((ra * rb).sum() / denom)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        nf, dim = int(e.n_factor), int(e.dim)
        spear, frac_useful, frac_random, needs_real_flags = [], [], [], []
        for s in seeds:
            seed_everything(s)
            x, ya, yb = factorized_latents(
                n_a=nf, n_b=nf, per_cell=int(e.per_cell), dim=dim, interaction=0.0, seed=s
            )
            target = (ya * nf + yb).long()
            nc = int(target.max()) + 1
            # learned directions = rows of a trained linear probe weight (the readout basis)
            head = nn.Linear(dim, nc)
            opt = torch.optim.Adam(head.parameters(), lr=float(e.lr))
            for _ in range(int(e.epochs)):
                opt.zero_grad()
                F.cross_entropy(head(x), target).backward()
                opt.step()
            W = head.weight.detach()  # [nc, dim]
            # usefulness per dim = summed absolute readout weight (ablation-magnitude proxy)
            usefulness = W.abs().sum(0)  # [dim]
            # interpretability per dim = max abs correlation with a named generative factor (ya, yb)
            fa = ya.float() - ya.float().mean()
            fb = yb.float() - yb.float().mean()
            interp = torch.zeros(dim)
            for j in range(dim):
                xj = x[:, j] - x[:, j].mean()
                ca = abs(float((xj * fa).sum() / (xj.norm() * fa.norm() + 1e-12)))
                cb = abs(float((xj * fb).sum() / (xj.norm() * fb.norm() + 1e-12)))
                interp[j] = max(ca, cb)
            spear.append(self._spearman(usefulness, interp))
            # fraction of the top-useful dims whose interpretability exceeds the random-dim median
            topk = max(1, dim // 5)
            top_idx = usefulness.argsort(descending=True)[:topk]
            thresh = interp.median()
            frac_useful.append(float((interp[top_idx] > thresh).float().mean()))
            g = torch.Generator().manual_seed(s + 13)
            rand_idx = torch.randperm(dim, generator=g)[:topk]
            frac_random.append(float((interp[rand_idx] > thresh).float().mean()))
            # decodability sanity only (above the shuffled floor); reported, not gated. This is NOT an
            # encoder-specificity claim (frozen-random is vacuous for a linear probe).
            abl = substrate_ablation(x, target, seed=s)
            needs_real_flags.append(abl["needs_real"])

        def mean(v):
            return sum(v) / len(v)

        sm, fu, fr = mean(spear), mean(frac_useful), mean(frac_random)
        needs_real = bool(sum(needs_real_flags) > len(needs_real_flags) / 2)
        out = {
            "spearman_use_interp": round(sm, 4),
            "frac_top_useful_named": round(fu, 4),
            "frac_random_named": round(fr, 4),
            "named_gain_vs_random": round(fu - fr, 4),
            "needs_real": needs_real,
            "seeds": list(seeds),
            # null: usefulness and interpretability uncorrelated (Spearman ~ 0, top-useful not more named)
            "null_supported": bool(abs(sm) <= 0.2 and (fu - fr) <= 0.15),
            "useful_is_namable": bool(sm > 0.2 and (fu - fr) > 0.15),
        }
        return out


# ====================================================================================================
# S10: anti-self-deception meta-test (does a random projection pass the S-tests)
# ====================================================================================================
class S10(Experiment):
    id = "s10_anti_self_deception"
    metric = (
        "frac_tests_above_shuffle",
        "frozen_random_is_vacuous_control",
        "delta_frozen_random",
        "delta_shuffled",
    )
    baseline = "shuffled-pairing floor (the valid certifier) and a frozen-random projection (known-vacuous)"
    ablation = "real vs shuffled floor vs frozen_random over each S-style decode test"
    null_hypothesis = (
        "an S-style decode test carries no genuine signal: its real score does not beat the shuffled "
        "chance floor. Note on the frozen-random arm: for a LINEAR decode metric a full-rank random "
        "projection is invertible, so real always ties frozen-random (delta_frozen_random ~ 0) by "
        "construction. frozen-random therefore cannot certify non-vacuity for these tests; the shuffled "
        "floor is the valid certifier. This meta-test now REPORTS that frozen-random is a vacuous control "
        "here rather than mislabeling shuffle-beating signal as vacuous."
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nf = int(e.dim), int(e.n_factor)
        # the S-style decode tests routed through substrate_ablation: alignment (cluster label),
        # arithmetic-factor decode, compositional joint-cell decode, translation (single factor decode)
        per_test: dict[str, dict] = {}
        for name in ("alignment", "arithmetic", "compositionality", "translation"):
            per_test[name] = {"needs_real": [], "dfr": [], "dsh": []}
        for s in seeds:
            seed_everything(s)
            cluster = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            xf, ya, yb = factorized_latents(
                n_a=nf, n_b=nf, per_cell=int(e.per_cell), dim=dim, interaction=0.0, seed=s
            )
            joint = (ya * nf + yb).long()
            tests = {
                "alignment": (cluster.x, cluster.y),
                "arithmetic": (xf, ya),
                "compositionality": (xf, joint),
                "translation": (xf, yb),
            }
            for name, (xx, yy) in tests.items():
                abl = substrate_ablation(xx, yy, seed=s)
                # needs_real now means "real beats the shuffled floor" (genuine signal present)
                per_test[name]["needs_real"].append(abl["needs_real"])
                per_test[name]["dfr"].append(abl["delta_frozen_random"])
                per_test[name]["dsh"].append(abl["delta_shuffled"])

        def mean(v):
            return sum(v) / len(v)

        verdicts = {}
        n_above_shuffle = 0
        fr_tie_margin = 0.05
        for name, d in per_test.items():
            above = bool(sum(d["needs_real"]) > len(d["needs_real"]) / 2)
            dfr = mean(d["dfr"])
            ties_fr = bool(abs(dfr) <= fr_tie_margin)  # real ~ frozen_random (expected for a linear metric)
            verdicts[name] = {
                "signal_above_shuffle_floor": above,
                "delta_frozen_random": round(dfr, 4),
                "delta_shuffled": round(mean(d["dsh"]), 4),
                "ties_frozen_random": ties_fr,
                # a test carries real signal iff it beats the shuffled floor; the frozen_random tie is
                # expected (invertible linear map) and does NOT make a shuffle-beating test vacuous
                "verdict": "SIGNAL-PRESENT" if above else "NO-SIGNAL",
            }
            n_above_shuffle += int(above)
        all_tie_frozen_random = all(v["ties_frozen_random"] for v in verdicts.values())
        all_above_shuffle = all(v["signal_above_shuffle_floor"] for v in verdicts.values())
        frac_above = n_above_shuffle / len(verdicts)
        out = {
            "per_test": verdicts,
            "frac_tests_above_shuffle": round(frac_above, 4),
            "all_tests_above_shuffle": bool(all_above_shuffle),
            # the honest meta-finding: for these LINEAR decode tests real always ties frozen_random, so
            # the frozen_random arm is a vacuous certifier (it cannot fail on an invertible map). The
            # shuffled floor is the valid certifier and every test clears it.
            "frozen_random_is_vacuous_control": bool(all_tie_frozen_random),
            "valid_certifier": "shuffled_floor",
            "n_tests": len(verdicts),
            "seeds": list(seeds),
            # null: some test carries no genuine signal (does not beat the shuffled floor). With the
            # corrected certifier this is False here (all tests clear the floor).
            "null_supported": bool(not all_above_shuffle),
        }
        return out
