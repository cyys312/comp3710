# Record of AI use — COMP3710 Lab 1

The lab sheet requires that all prompts, outputs and model reasoning be
documented when AI models are used, and states that projects generated from a
single prompt may lose significant marks. This file is that record.

**Assistant used:** Claude (Anthropic), via Claude Code in a terminal session.

---

## Part 3 — Barnsley fern

### How the fractal was chosen

Rather than asking for "a fractal", the selection was made against the lab
sheet's constraints, which rules out most of the obvious answers:

- must not be close to the Mandelbrot set, and must not be one of the fractals
  already in the lab sheet
- must use PyTorch/TF/JAX parallelism in a *major component* of the algorithm
- must support substantial analysis, not just a picture

Four candidates were compared on those criteria:

| candidate | mechanism | verdict |
|---|---|---|
| Lyapunov fractal | Lyapunov exponent of a periodically forced logistic map | viable; per-pixel, same shape of parallelism as the lab sheet code |
| **IFS chaos game (Barnsley fern)** | random affine maps, point cloud | **chosen** — mechanism unrelated to escape-time, parallelism is over *orbits* not pixels, and it is explicitly on the lab sheet's approved list (textbook §6.6) |
| Newton fractal | Newton iteration on z³−1, coloured by root basin | rejected — still per-pixel complex iteration, real risk of being judged too close to Mandelbrot |
| Apollonian gasket | circle inversion / Möbius transformations | viable but the highest implementation cost |

**Correction made during the discussion.** The assistant initially claimed the
fern's fractal dimension "can be computed analytically from the affine
coefficients". That is wrong for this fractal: the Moran equation `Σ rᵢ^D = 1`
requires the maps to be *similarities* satisfying the open set condition. The
fern's maps shear, compress anisotropically, and overlap, so no closed form
applies. This is why `box_counting.py` measures the Sierpinski triangle (where
the formula *is* exact) as a control before reporting a number for the fern.

*[Add your own note here: did you check this claim yourself? How?]*

### Prompts used

**Prompt 1 — implementation**

> [paste the prompt you used]

Output: `ifs.py`, `render_fern.py`, `box_counting.py`.

Notable choices the assistant made and the stated reasons:

- **N orbits in parallel rather than one orbit for N steps.** The textbook
  program is a sequential loop; the parallel version is what justifies the GPU.
- **Burn-in discarded.** Orbits start at the origin, which is not on the
  attractor; without burn-in the approach trail is visible in the image.
- **Square window for box counting.** Boxes must be square in world coordinates,
  so the pixel grid is square and a power of two, letting `max_pool2d` at stride
  2, 4, 8 … correspond to genuinely square boxes.
- **Local-slope panel in the plot.** Makes the fitted range visible instead of
  assumed, since box counting produces a plausible straight line even when it is
  measuring sampling density rather than geometry.

**Prompt 2, 3, … — your own iterations**

> [record every follow-up: parameter changes, bugs you hit, things you asked it
>  to explain, things you disagreed with]

### What still needs to be *yours*

The demo asks you to explain how the fractal is formed and justify the use of
parallelism. Things worth doing yourself before the demo, and recording here:

- [ ] Change one map's coefficients and predict the effect before running it
      (e.g. raise f₂'s rotation and watch the fern curl)
- [ ] Run with `--burn-in 0` and see the approach trail — confirms why it matters
- [ ] Run with `--points 1000 --steps 200000`-style settings on CPU and compare
      the wall time to the parallel version. This is the number that proves the
      parallelism claim.
- [ ] Vary `--fit-lo` / `--fit-hi` and see how much the reported dimension moves.
      A dimension quoted without that sensitivity is not a measurement.
- [ ] Check the measured Sierpinski dimension against log3/log2 yourself

---

## Part 1 — AI Tasks

### Replicating the 2D Gaussian in NumPy, then converting to PyTorch

**Prompt**

> Generate a Python script to plot a 2D Gaussian function using Numpy and Matplotlib

*[paste the output summary and any follow-up prompts]*

**Problems observed:** *[fill in]*

Common failure modes to check for and record if they occurred:
- generated code defines `device` but never calls `.to(device)`, so nothing runs
  on the GPU — this happened in `task1.py`
- `plt.imshow()` called on a CUDA tensor without `.cpu().numpy()`
- missing `extent` / `origin='lower'`, so the axes are pixel indices and the
  image is vertically flipped

### 2D sine and the Gabor filter

*[fill in]* — key point: the sine's angle must depend on **both** x and y, or the
stripes come out vertical instead of oriented.

---

## Part 2 — AI Tasks

Lab sheet question: *how good is the AI model at generating a Mandelbrot set in
PyTorch that runs fast on the GPU, in just a few prompts? Were there issues, did
you need to modify the code, and what were they?*

**Prompts used:** *[fill in]*

**Issues actually encountered in this project:**

| issue | symptom | fix |
|---|---|---|
| copying code out of the lab-sheet PDF inserts spaces between tokens | `""" ` became `" " "` → `SyntaxError: unterminated string literal`; `'cuda'` became `' cuda '` → `RuntimeError: Invalid device string` | retyped rather than pasted |
| PDF copy also strips leading indentation | function and loop bodies at column 0 | rewrote the file |
| `ns = torch.zeros_like(z)` makes the counter complex | `ComplexWarning: Casting complex values to real discards the imaginary part` | harmless — the imaginary part is identically zero; image unaffected |
| diverged points keep iterating | values overflow to `inf`/`nan`, noisy at deep zoom | `torch.where` to freeze escaped points |
| `plt.show()` on a compute node | job "succeeds" but produces no file | `plt.savefig()` |
| editing in `nano` while commands are typed ahead | queued `cp` re-ran on exit and overwrote the edited file with the original | verify with `sed -n` after every edit |
| iteration count not raised with zoom depth | boundary detail collapses into solid black | 200 → 800 iterations |

**Conclusion:** *[write a few sentences in your own words]*

---

## Rangpur / HPC notes worth recording

- `comp3710`, `a100` and `a100-grind` are three queue names for the **same** ten
  `a100-[0-9]` nodes — switching between them does not shorten the wait
- `a100-test` (`a100-a`, `a100-b`) is usually free, but its QOS caps wall time at
  20 minutes and the node config rejects `--mem`
- a queued interactive `srun` blocks the terminal; anything typed goes to its
  stdin and is lost. `sbatch` returns immediately and survives disconnection
- Slurm does not create the directory named in `--output`
