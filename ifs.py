"""Iterated Function Systems and a GPU chaos game, in PyTorch.

An IFS is a finite set of contraction maps w_i(v) = A_i v + b_i, each chosen with
probability p_i. Hutchinson's theorem says the system has a unique compact
attractor -- and the "chaos game" finds it: start anywhere, repeatedly apply a
randomly chosen w_i, and the orbit converges onto the attractor.

The textbook program (Fractals for the Classroom, sec. 6.6) plays this game with
ONE point for many iterations. Here we play it with N points at once: every
iteration is a handful of batched tensor ops over the whole population, so a
million independent orbits advance in lockstep on the GPU. That is where the
parallelism lives -- there is no Python loop over points, only over steps.
"""

import torch

# --- Barnsley fern -------------------------------------------------------
# Four affine maps. Read them as: f1 draws the stem, f2 is the self-similar
# "shrink the whole fern and place it one leaflet up" map (hence p = 0.85, it
# does most of the work), f3 and f4 spawn the largest left/right leaflets.
BARNSLEY_FERN = {
    "name": "barnsley-fern",
    "A": [[[0.00,  0.00], [ 0.00, 0.16]],    # f1  stem
          [[0.85,  0.04], [-0.04, 0.85]],    # f2  successively smaller leaflets
          [[0.20, -0.26], [ 0.23, 0.22]],    # f3  largest left leaflet
          [[-0.15, 0.28], [ 0.26, 0.24]]],   # f4  largest right leaflet
    "b": [[0.00, 0.00], [0.00, 1.60], [0.00, 1.60], [0.00, 0.44]],
    "p": [0.01, 0.85, 0.07, 0.07],
    "labels": ["stem", "leaflets", "left", "right"],
    "exact_dim": None,        # no closed form: the maps are affine, not similarities
}

# --- Sierpinski triangle -------------------------------------------------
# Included as a CONTROL. Its three maps are genuine similarities with ratio 1/2
# satisfying the open set condition, so the Moran equation 3 * (1/2)^D = 1 gives
# the dimension exactly: D = log 3 / log 2. Measuring a known answer is how we
# check the box-counting code before trusting it on the fern.
_H = 0.8660254037844386          # sqrt(3)/2
SIERPINSKI = {
    "name": "sierpinski",
    "A": [[[0.5, 0.0], [0.0, 0.5]]] * 3,
    "b": [[0.0, 0.0], [0.5, 0.0], [0.25, _H / 2]],
    "p": [1 / 3, 1 / 3, 1 / 3],
    "labels": ["A", "B", "C"],
    "exact_dim": 1.5849625007211562,          # log(3)/log(2)
}

SYSTEMS = {s["name"]: s for s in (BARNSLEY_FERN, SIERPINSKI)}


def _as_tensors(ifs, device):
    A = torch.tensor(ifs["A"], dtype=torch.float32, device=device)   # (M, 2, 2)
    b = torch.tensor(ifs["b"], dtype=torch.float32, device=device)   # (M, 2)
    p = torch.tensor(ifs["p"], dtype=torch.float32, device=device)   # (M,)
    cum = torch.cumsum(p / p.sum(), dim=0)
    cum[-1] = 1.0                       # guard against float drift at the top end
    return A, b, cum


def chaos_game(ifs, n_points, n_steps, burn_in, device, seed=0):
    """Yield (points, transform_index) after each post-burn-in step.

    points : (n_points, 2) float32 -- one live orbit per row
    index  : (n_points,)   int64   -- which map produced this position

    Burn-in matters: the orbits start at the origin, which is generally NOT on
    the attractor. The maps are contractions, so the distance to the attractor
    decays geometrically; discarding the first `burn_in` steps throws away the
    visible "approach trail" that would otherwise smear the image.
    """
    A, b, cum = _as_tensors(ifs, device)
    n_maps = cum.numel()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    pts = torch.zeros(n_points, 2, device=device)

    for step in range(n_steps):
        # one uniform draw per orbit -> per-orbit choice of map
        u = torch.rand(n_points, device=device, generator=gen)
        idx = torch.searchsorted(cum, u).clamp_(max=n_maps - 1)

        # gather this step's matrices and offsets, then apply them all at once
        pts = torch.einsum("nij,nj->ni", A[idx], pts) + b[idx]

        if step >= burn_in:
            yield pts, idx


def bounds(ifs, device, n_points=200_000, n_steps=120, burn_in=20, seed=0):
    """Empirical bounding box of the attractor (min/max over sampled orbits)."""
    lo = torch.full((2,), float("inf"), device=device)
    hi = torch.full((2,), float("-inf"), device=device)
    for pts, _ in chaos_game(ifs, n_points, n_steps, burn_in, device, seed):
        lo = torch.minimum(lo, pts.amin(dim=0))
        hi = torch.maximum(hi, pts.amax(dim=0))
    return lo.cpu().tolist(), hi.cpu().tolist()


def square_view(lo, hi, margin=0.02):
    """Smallest square window containing [lo, hi], plus a margin.

    Box counting requires SQUARE boxes in world coordinates. Forcing the window
    square (rather than fitting the fern's tall bounding box) lets us use a
    power-of-two pixel grid whose cells are square, so pooling by 2, 4, 8, ...
    corresponds to genuinely square boxes of side 2e, 4e, 8e.
    """
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 * (1 + margin)
    return (cx - half, cx + half, cy - half, cy + half)


def rasterise(ifs, view, height, width, n_points, n_steps, burn_in,
              device, seed=0, per_map=False):
    """Play the chaos game and bin every visited point into a density grid.

    Returns an int64 tensor of hit counts, shaped (height, width) or
    (n_maps, height, width) when per_map=True.
    """
    xmin, xmax, ymin, ymax = view
    n_maps = len(ifs["p"])
    shape = (n_maps, height, width) if per_map else (height, width)
    hist = torch.zeros(shape, dtype=torch.int64, device=device)

    sx = width / (xmax - xmin)
    sy = height / (ymax - ymin)

    for pts, idx in chaos_game(ifs, n_points, n_steps, burn_in, device, seed):
        ix = ((pts[:, 0] - xmin) * sx).long()
        iy = ((pts[:, 1] - ymin) * sy).long()
        keep = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        flat = iy[keep] * width + ix[keep]

        if per_map:
            sel = idx[keep]
            for m in range(n_maps):
                f = flat[sel == m]
                if f.numel():
                    hist[m] += torch.bincount(f, minlength=height * width).view(height, width)
        else:
            hist += torch.bincount(flat, minlength=height * width).view(height, width)

    return hist
