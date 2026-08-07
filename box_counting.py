"""COMP3710 Lab 1, Part 3 -- box-counting dimension of an IFS attractor.

The idea: cover the attractor with a grid of square boxes of side e and count
how many boxes N(e) it touches. For a fractal, N(e) ~ e^(-D) over a range of
scales, so D is the slope of log N(e) against log(1/e).

Why this script also does the Sierpinski triangle
-------------------------------------------------
Box counting is easy to get wrong: too coarse and the whole object sits in one
box, too fine and you are measuring how many points you happened to sample
rather than the geometry. Both failures still produce a straight-ish line and a
plausible number.

So the pipeline is first run on a fractal whose dimension is known exactly. The
Sierpinski triangle is built from three similarities of ratio 1/2 satisfying the
open set condition, so Moran's equation 3*(1/2)^D = 1 gives D = log3/log2 =
1.58496... exactly. If the code recovers that, the fern measurement is credible.

The fern gets no such closed form: its four maps are affine but NOT similarities
(they shear, and they compress by different amounts along different directions),
and their images overlap, so the Moran equation does not apply. Box counting is
the honest way to get a number.

Run:
    python box_counting.py --system all
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ifs import SYSTEMS, bounds, square_view, rasterise


def occupancy_grid(ifs, device, grid, points, steps, burn_in, seed):
    """Boolean grid of visited cells, on a SQUARE window with square cells."""
    lo, hi = bounds(ifs, device, seed=seed)
    view = square_view(lo, hi)
    counts = rasterise(ifs, view, grid, grid, points, steps, burn_in,
                       device, seed=seed, per_map=False)
    cell = (view[1] - view[0]) / grid
    return counts > 0, cell, view


def box_counts(occ, scales):
    """N(e) for each box size, in pixels, via max-pooling the occupancy grid."""
    x = occ.float()[None, None]
    out = []
    for s in scales:
        pooled = x if s == 1 else F.max_pool2d(x, kernel_size=s, stride=s)
        out.append(int(pooled.sum().item()))
    return out


def analyse(name, device, args):
    ifs = SYSTEMS[name]
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")

    t0 = time.time()
    occ, cell, view = occupancy_grid(ifs, device, args.grid, args.points,
                                     args.steps, args.burn_in, args.seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = args.points * (args.steps - args.burn_in)
    print(f"window   : x [{view[0]:.3f}, {view[1]:.3f}]  (square, {args.grid}^2 cells)")
    print(f"cell size: {cell:.3e}")
    print(f"sampled  : {samples:,} points in {time.time() - t0:.2f} s")
    print(f"occupied : {int(occ.sum().item()):,} cells at the finest scale")

    scales = [1 << k for k in range(args.max_pow + 1) if (1 << k) <= args.grid // 4]
    counts = box_counts(occ, scales)
    eps = np.array([s * cell for s in scales], dtype=float)
    N = np.array(counts, dtype=float)

    keep = N > 0
    eps, N, scales = eps[keep], N[keep], [s for s, k in zip(scales, keep) if k]

    logx = np.log(1.0 / eps)
    logy = np.log(N)

    # slope between neighbouring scales -- shows where the power law actually holds
    local = np.diff(logy) / np.diff(logx)

    print(f"\n{'box (px)':>9} {'e':>12} {'N(e)':>12} {'local slope':>12}")
    for i, (s, e, n) in enumerate(zip(scales, eps, N)):
        ls = f"{local[i]:.4f}" if i < len(local) else ""
        print(f"{s:>9} {e:>12.3e} {int(n):>12,} {ls:>12}")

    lo_i, hi_i = args.fit_lo, len(scales) - args.fit_hi
    if hi_i - lo_i < 2:
        lo_i, hi_i = 0, len(scales)
    slope, intercept = np.polyfit(logx[lo_i:hi_i], logy[lo_i:hi_i], 1)

    resid = logy[lo_i:hi_i] - (slope * logx[lo_i:hi_i] + intercept)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((logy[lo_i:hi_i] - logy[lo_i:hi_i].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nfit over box sizes {scales[lo_i]}..{scales[hi_i - 1]} px "
          f"({hi_i - lo_i} points)")
    print(f"  box-counting dimension D = {slope:.4f}   (R^2 = {r2:.5f})")
    if ifs["exact_dim"] is not None:
        exact = ifs["exact_dim"]
        print(f"  exact (Moran)            = {exact:.4f}")
        print(f"  error                    = {abs(slope - exact):.4f} "
              f"({abs(slope - exact) / exact * 100:.2f}%)")

    # ---- plot ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(logx, logy, "o", ms=5, label="measured")
    xs = np.linspace(logx[lo_i], logx[hi_i - 1], 2)
    ax1.plot(xs, slope * xs + intercept, "-", lw=2,
             label=f"fit: D = {slope:.4f}")
    ax1.axvspan(logx[lo_i], logx[hi_i - 1], alpha=0.08, color="tab:green")
    ax1.set_xlabel("log(1/e)")
    ax1.set_ylabel("log N(e)")
    ax1.set_title(f"{name}: box counting")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(logx[:-1], local, "o-", ms=4, label="local slope")
    ax2.axhline(slope, ls="--", color="tab:orange", label=f"fitted D = {slope:.4f}")
    if ifs["exact_dim"] is not None:
        ax2.axhline(ifs["exact_dim"], ls=":", color="tab:red",
                    label=f'exact = {ifs["exact_dim"]:.4f}')
    ax2.axvspan(logx[lo_i], logx[hi_i - 1], alpha=0.08, color="tab:green")
    ax2.set_xlabel("log(1/e)")
    ax2.set_ylabel("slope between adjacent scales")
    ax2.set_title("where the power law holds")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(args.outdir, f"{name}_boxcount.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved:", path)
    return slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all",
                    choices=list(SYSTEMS) + ["all"])
    ap.add_argument("--grid", type=int, default=4096,
                    help="finest grid, must be a power of two")
    ap.add_argument("--points", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--burn-in", type=int, default=20)
    ap.add_argument("--max-pow", type=int, default=10)
    ap.add_argument("--fit-lo", type=int, default=2,
                    help="drop this many of the SMALLEST boxes (sampling limit)")
    ap.add_argument("--fit-hi", type=int, default=2,
                    help="drop this many of the LARGEST boxes (saturation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("PyTorch Version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device         :", device)
    if torch.cuda.is_available():
        print("GPU            :", torch.cuda.get_device_name(0))

    names = list(SYSTEMS) if args.system == "all" else [args.system]
    # control first, so a bad pipeline is caught before the fern is reported
    names.sort(key=lambda n: SYSTEMS[n]["exact_dim"] is None)

    results = {n: analyse(n, device, args) for n in names}

    print(f"\n{'=' * 62}\nsummary\n{'=' * 62}")
    for n, d in results.items():
        exact = SYSTEMS[n]["exact_dim"]
        ref = f"   exact {exact:.4f}" if exact is not None else "   (no closed form)"
        print(f"  {n:<18} D = {d:.4f}{ref}")


if __name__ == "__main__":
    main()
