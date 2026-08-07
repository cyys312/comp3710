"""COMP3710 Lab 1, Part 3 -- render the Barnsley fern three different ways.

  1. log-density        how often the chaos game visits each pixel
  2. classic green      the picture people recognise as a fern
  3. by-transform       each pixel coloured by WHICH of the four affine maps put
                        the point there -- this one is the explanation: you can
                        literally see f2 building leaflet after leaflet

Run:
    python render_fern.py                       # fern, 1M orbits
    python render_fern.py --system sierpinski   # the control fractal
"""

import argparse
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")            # compute nodes have no display
import matplotlib.pyplot as plt

from ifs import SYSTEMS, bounds, rasterise

# one colour per affine map, in the order they are listed in ifs.py
MAP_COLOURS = np.array([
    [0.97, 0.83, 0.25],          # f1  stem      -- yellow
    [0.18, 0.68, 0.32],          # f2  leaflets  -- green
    [0.25, 0.55, 0.95],          # f3  left      -- blue
    [0.92, 0.36, 0.45],          # f4  right     -- red
    [0.70, 0.45, 0.85],          # spare
])


def padded_view(lo, hi, margin=0.05):
    """Bounding box of the attractor with a little air around it."""
    dx, dy = hi[0] - lo[0], hi[1] - lo[1]
    return (lo[0] - margin * dx, hi[0] + margin * dx,
            lo[1] - margin * dy, hi[1] + margin * dy)


def normalised_log(counts):
    """Hit counts -> [0, 1] brightness. Log because the density spans orders of
    magnitude: the stem gets 1% of the draws, the leaflet map gets 85%."""
    a = torch.log1p(counts.float())
    peak = a.max()
    return (a / peak).cpu().numpy() if peak > 0 else a.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="barnsley-fern", choices=list(SYSTEMS))
    ap.add_argument("--points", type=int, default=1_000_000,
                    help="number of orbits played simultaneously")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--burn-in", type=int, default=20)
    ap.add_argument("--height", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ifs = SYSTEMS[args.system]

    print("PyTorch Version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device         :", device)
    if torch.cuda.is_available():
        print("GPU            :", torch.cuda.get_device_name(0))

    lo, hi = bounds(ifs, device, seed=args.seed)
    view = padded_view(lo, hi)
    # keep pixels square so the picture is not stretched
    width = max(1, int(round(args.height * (view[1] - view[0]) / (view[3] - view[2]))))
    print(f"Attractor bbox : x [{lo[0]:.3f}, {hi[0]:.3f}]  y [{lo[1]:.3f}, {hi[1]:.3f}]")
    print(f"Raster         : {width} x {args.height}")

    samples = args.points * (args.steps - args.burn_in)
    print(f"Orbits         : {args.points:,} in parallel x {args.steps} steps "
          f"({samples:,} points plotted)")

    t0 = time.time()
    per_map = rasterise(ifs, view, args.height, width, args.points,
                        args.steps, args.burn_in, device, seed=args.seed,
                        per_map=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"Chaos game     : {dt:.2f} s  ({samples / dt / 1e6:.1f} M points/s)")

    total = per_map.sum(dim=0)
    intensity = normalised_log(total)
    extent = [view[0], view[1], view[2], view[3]]

    def save(fig, name):
        path = os.path.join(args.outdir, f"{args.system}_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        print("Saved:", path)

    # --- 1. log-density ---------------------------------------------------
    fig = plt.figure(figsize=(6, 6 * args.height / width))
    plt.imshow(intensity, origin="lower", extent=extent, cmap="viridis")
    plt.axis("off")
    plt.tight_layout(pad=0)
    save(fig, "density")

    # --- 2. classic green -------------------------------------------------
    green = np.zeros(intensity.shape + (3,))
    green[..., 0] = 0.10 * intensity
    green[..., 1] = 0.95 * intensity
    green[..., 2] = 0.25 * intensity
    fig = plt.figure(figsize=(6, 6 * args.height / width))
    plt.imshow(green, origin="lower", extent=extent)
    plt.axis("off")
    plt.tight_layout(pad=0)
    save(fig, "green")

    # --- 3. coloured by which map produced the point ----------------------
    winner = per_map.argmax(dim=0).cpu().numpy()
    rgb = MAP_COLOURS[winner] * intensity[..., None]
    fig = plt.figure(figsize=(7, 6 * args.height / width))
    plt.imshow(rgb, origin="lower", extent=extent)
    plt.axis("off")
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                          color=MAP_COLOURS[i],
                          label=f'f{i+1} {ifs["labels"][i]}  p={ifs["p"][i]:.2f}')
               for i in range(len(ifs["p"]))]
    plt.legend(handles=handles, loc="upper left", framealpha=0.15,
               labelcolor="white", fontsize=8)
    plt.tight_layout(pad=0)
    save(fig, "by_transform")

    # how the probability mass actually landed
    shares = (per_map.sum(dim=(1, 2)).float() / total.sum().float()).cpu().tolist()
    print("\nShare of plotted points per map (should track p):")
    for i, (lab, p, s) in enumerate(zip(ifs["labels"], ifs["p"], shares)):
        print(f"  f{i+1} {lab:<10} p={p:.2f}   measured={s:.3f}")


if __name__ == "__main__":
    main()
