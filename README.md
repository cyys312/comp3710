# COMP3710 — Barnsley Fern via a GPU Chaos Game

Pattern Analysis, Lab 1 Part 3. An iterated function system (IFS) rendered with
PyTorch, where a million orbits of the chaos game advance in parallel on the GPU,
plus a box-counting measurement of the attractor's fractal dimension.

Source: *Fractals for the Classroom*, section 6.6 — "Program of the Chapter:
Chaos Game for the Fern", one of the chapters listed in the lab sheet.

## The mathematics

An IFS is a finite set of contraction maps on the plane,

```
w_i(v) = A_i v + b_i        chosen with probability p_i
```

Hutchinson's theorem guarantees such a system has a unique compact **attractor**:
the one set that is exactly the union of its own images under all the maps. The
fern is that attractor for these four affine maps:

| map | A | b | p | what it draws |
|-----|---|---|---|---|
| f₁ | `[[0, 0], [0, 0.16]]` | `(0, 0)` | 0.01 | the stem |
| f₂ | `[[0.85, 0.04], [−0.04, 0.85]]` | `(0, 1.60)` | 0.85 | the whole fern, shrunk ~15% and nudged up one leaflet |
| f₃ | `[[0.20, −0.26], [0.23, 0.22]]` | `(0, 1.60)` | 0.07 | the largest left-hand leaflet |
| f₄ | `[[−0.15, 0.28], [0.26, 0.24]]` | `(0, 0.44)` | 0.07 | the largest right-hand leaflet |

f₂ is where the self-similarity comes from: it maps the entire fern onto itself
one leaflet higher, so every leaflet is a slightly shrunken, slightly rotated
copy of the whole plant, forever.

The **chaos game** finds the attractor without ever solving for it: start at any
point, repeatedly apply a randomly chosen `w_i`, and the orbit is drawn onto the
attractor geometrically fast (every map is a contraction, so the distance to the
attractor shrinks by a constant factor each step). The probabilities do not
change *what* the attractor is — only how densely each region gets sampled, so
they are chosen roughly in proportion to how much area each map covers.

The first `--burn-in` steps are discarded: the orbits start at the origin, which
is not on the attractor, and plotting the approach would smear the image.

## Where the GPU parallelism is

The textbook program plays the chaos game with **one** point for many thousands
of iterations — an inherently sequential loop, because each position depends on
the previous one.

The parallelism here comes from playing **N independent games at once**. Each
iteration of `ifs.chaos_game` is a handful of batched tensor operations over the
entire population:

```python
u   = torch.rand(n_points, device=device, generator=gen)   # one draw per orbit
idx = torch.searchsorted(cum, u)                           # per-orbit map choice
pts = torch.einsum("nij,nj->ni", A[idx], pts) + b[idx]     # all orbits advance
```

There is no Python loop over points anywhere — only over steps, and the step
count is small (a few hundred) while the point count is large (10⁶). The
`A[idx]` gather builds a per-orbit stack of 2×2 matrices and `einsum` applies a
million different matrix–vector products in one kernel. Binning is the same
story: `torch.bincount` on flattened pixel indices accumulates a million hits per
step without a loop.

`render_fern.py` prints the throughput it achieves (points plotted per second),
which makes the GPU's contribution concrete rather than asserted.

## Fractal dimension

`box_counting.py` covers the attractor with square boxes of side `e` and counts
how many are touched. For a fractal `N(e) ~ e^(−D)`, so `D` is the slope of
`log N(e)` against `log(1/e)`.

**The script measures a known answer first.** Box counting fails quietly: too
coarse and everything sits in one box, too fine and you measure your sampling
density rather than the geometry — and both still produce a straight-ish line.
So the pipeline is run on the **Sierpinski triangle**, whose three maps are
genuine similarities of ratio ½ satisfying the open set condition. Moran's
equation `3·(½)^D = 1` gives its dimension exactly:

```
D = log 3 / log 2 = 1.584962...
```

If the code recovers that, the fern number is credible.

The fern gets **no closed form**. Its maps are affine but not similarities — they
shear, and compress by different amounts in different directions — and their
images overlap, so Moran's equation does not apply. Box counting is the honest
route, and the second panel of the output plot shows the local slope at every
scale so the fitted range is visible rather than assumed.

There is still a useful cross-check. Feeding Moran's equation the geometric-mean
contraction of each map, `rᵢ = √|det Aᵢ|`, gives an estimate that ignores both the
anisotropy and the overlap — approximate by construction, but arrived at from a
completely different direction than counting boxes. `moran_estimate()` computes
it. For the fern the two agree to 0.5%, which is a much stronger statement than
either number alone.

(f₁ is skipped in that estimate: `det = 0`, because the stem map collapses the
plane onto a line segment. A line cannot raise a dimension that is already
above 1.)

## Output

`render_fern.py` writes three views, because the same data answers three
different questions:

| file | what it shows |
|---|---|
| `*_density.png` | visit frequency (log scale — the stem gets 1% of the draws, f₂ gets 85%) |
| `*_green.png` | the picture people recognise as a fern |
| `*_by_transform.png` | each pixel coloured by **which map** put the point there |

The third is the explanatory one: f₂'s territory is visibly the whole fern minus
the two big leaflets, which is exactly the self-similarity stated above.

## Running it

Locally (CPU works, just slower):

```bash
pip install torch numpy matplotlib
```

```bash
python render_fern.py && python box_counting.py --system all
```

On Rangpur — create the log directory first, Slurm will not:

```bash
mkdir -p ~/comp3710/logs && cd ~/comp3710 && sbatch slurm/job_fern.sh
```

Check `sinfo -s` before submitting: `comp3710`, `a100` and `a100-grind` are the
same ten `a100-[0-9]` nodes and are frequently full, while `a100-test`
(`a100-a`, `a100-b`) is usually free but capped at 20 minutes of wall time by its
QOS, and rejects `--mem` requests.

## Results

Measured on Rangpur, one NVIDIA A100-PCIE-40GB, PyTorch 2.13.0+cu130.

| quantity | value |
|---|---|
| orbits × steps | 1,000,000 × 200 |
| points plotted | 180,000,000 |
| chaos-game time | **0.71 s** |
| throughput | **253.3 M points/s** |
| attractor bounding box | x ∈ [−2.182, 2.656], y ∈ [0.003, 9.998] |

Sampling fidelity — the share of plotted points landing under each map matches
the probabilities it was given, to three decimals:

| map | p | measured |
|---|---|---|
| f₁ stem | 0.01 | 0.010 |
| f₂ leaflets | 0.85 | 0.850 |
| f₃ left | 0.07 | 0.070 |
| f₄ right | 0.07 | 0.070 |

Dimension, from a 4096² occupancy grid:

| attractor | box counting | reference | gap |
|---|---|---|---|
| Sierpinski triangle (control) | **1.5944** | 1.5850 exact (`log3/log2`) | 0.59% |
| Barnsley fern | **1.8245** | 1.8336 self-affine Moran estimate | 0.50% |

The control lands within 0.6% of an exactly known answer, so the pipeline is
doing what it claims. The fern's 1.8245 then agrees to 0.5% with an estimate
derived from the map determinants rather than from counting anything — two
independent routes to the same number.

A dimension near 1.82 is high but not surprising once you look at the
`by_transform` image: the leaflets pack against each other densely enough that
the attractor comes much closer to covering area than the thin outline
suggests.

## Files

| file | purpose |
|---|---|
| `ifs.py` | IFS definitions and the GPU chaos game engine |
| `render_fern.py` | the three visualisations |
| `box_counting.py` | dimension measurement, validated on Sierpinski |
| `slurm/job_fern.sh` | Slurm batch job for Rangpur |
| `AI_PROMPTS.md` | record of AI use, as the lab sheet requires |
