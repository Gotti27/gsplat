# Using gsplat as a dependency

This fork is meant to be consumed by other research projects. Four supported
ways, ordered from "just reference the repo" to "fully prebuilt".

| Method | Needs `nvcc` | Compile time | Isolation must be off | Notes |
| --- | --- | --- | --- | --- |
| uv project, git source | yes | ~45 min once per env | yes | pin a tag or commit |
| `uv pip install "gsplat @ git+ssh://…"` | yes | ~45 min once per env | yes | quick one-off |
| Prebuilt wheel (GitHub Release) | no | none | n/a | tied to torch/CUDA/python/arch |
| Local path, editable | yes | incremental | n/a | co-development |

Source installs on a CUDA 12.x toolkit need no extra flags: the build system
pre-includes the `cuda::ceil_div` compatibility header automatically (see
[CUDA12.md](CUDA12.md)). `NVCC_FLAGS` remains available as an override.

## Published releases

| Release | `torch` | Wheel |
| --- | --- | --- |
| `v1.6.0+leo1` | `2.9.1+cu126` | `gsplat-1.6.0+leo1-cp314-cp314-linux_x86_64.whl` |
| `v1.6.0+leo2` | `2.14.1+cu126` | `gsplat-1.6.0+leo2-cp314-cp314-linux_x86_64.whl` |

Both carry the same CUDA 12.x compatibility shim, are tagged `cp314` /
`linux_x86_64`, and contain `sm_80` device code. Pick the release whose torch
minor matches your environment; the examples below use `+leo2`.

## 1. uv project, git source

```toml
# pyproject.toml
[project]
dependencies = ["gsplat"]

[tool.uv.sources]
gsplat = { git = "ssh://git@github.com/Gotti27/gsplat.git", tag = "v1.6.0+leo2" }
```

```bash
UV_NO_BUILD_ISOLATION=1 uv sync
```

Pin a commit instead of a moving branch when reproducibility matters:

```toml
gsplat = { git = "ssh://git@github.com/Gotti27/gsplat.git", rev = "<40-char sha>" }
```

### Why `UV_NO_BUILD_ISOLATION=1` is required

`pyproject.toml` declares `torch>=2.7` as a build requirement. With isolation
enabled, uv installs a *fresh* torch from PyPI into a temporary build
environment and links `gsplat/csrc.so` against it. The resulting extension then
fails to import in your environment, which uses a different torch build (for
example `torch==2.14.1+cu126` from the PyTorch index). Disabling isolation makes
the build use your installed torch.

The same applies to `uv pip`:

```bash
UV_NO_BUILD_ISOLATION=1 uv pip install \
  "gsplat @ git+ssh://git@github.com/Gotti27/gsplat.git@v1.6.0+leo2"
```

### glm submodule

`gsplat/cuda/csrc/third_party/glm` is a submodule and is required by the CUDA
headers. uv and pip clone submodules for VCS installs; if you copy the tree
manually, initialise it first:

```bash
git submodule update --init gsplat/cuda/csrc/third_party/glm
```

## 2. Prebuilt wheel

Download the wheel attached to the matching GitHub Release and reference it
directly (no `nvcc`, no compile, no isolation concerns):

```bash
uv add "gsplat @ https://github.com/Gotti27/gsplat/releases/download/v1.6.0+leo2/gsplat-1.6.0+leo2-cp314-cp314-linux_x86_64.whl"
```

A wheel is **not** universal:

- it is linked against one **torch minor** (e.g. 2.14) and one **CUDA major**
  (e.g. 12.6);
- it is tagged for one **Python minor** (e.g. cp314);
- it contains device code for the architectures in `TORCH_CUDA_ARCH_LIST` at
  build time (e.g. `8.0` for A100).

For a wheel that also runs on newer GPUs without a rebuild, build with
`TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"` or `"8.0+PTX"`; the trailing PTX makes the
driver JIT the kernels for an unknown compute capability on first use.

### Install order: torch first, and do not use `--no-deps`

The extension inside the wheel is linked against one specific torch build, so
install torch from the index you intend to use *before* the wheel:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.14.1 torchvision==0.29.1
uv pip install "https://github.com/Gotti27/gsplat/releases/download/v1.6.0+leo2/gsplat-1.6.0+leo2-cp314-cp314-linux_x86_64.whl"
```

Installing torch first is sufficient: uv does not revisit an already satisfied
requirement, so this ordering alone protects the `+cu126` build from being
replaced by a PyPI one.

Do **not** add `--no-deps` to the wheel install. It also skips gsplat's own
runtime requirements (`jaxtyping`, `ninja`, `nvtx`, `rich`), and the failure
surfaces much later — as `ModuleNotFoundError: No module named 'jaxtyping'`
raised from inside `nerfview`, which looks unrelated to gsplat. Keeping the
exact torch pins in your own requirements file is the right guard, not
`--no-deps`.

## 3. Local path / co-development

For projects that need to modify the CUDA sources:

```toml
[tool.uv.sources]
gsplat = { path = "../gsplat", editable = true }
```

Editable installs do not recompile native modules on import, so after changing
CUDA code run:

```bash
python setup.py build_ext --inplace   # incremental (build/ is reused)
python setup.py develop --no-deps
```

or use the JIT flavour described in [CUDA12.md](CUDA12.md), which recompiles on
demand into `~/.cache/torch_extensions/`.

## Using the example trainer and viewer

The wheel contains only the `gsplat` package. `simple_trainer.py` and
`simple_viewer.py` live in `examples/` and import sibling modules, so a
downstream project that wants them has to vendor:

```
simple_trainer.py   simple_viewer.py   gsplat_viewer.py   utils.py   exif.py
datasets/{__init__,colmap,traj,normalize}.py
```

Keep them together in one directory and run them from there, so the sibling
imports resolve.

Requirements beyond the wheel's own (`pycolmap`, `tyro`, `torchmetrics`,
`tensorboard`, `imageio[ffmpeg]`, `opencv-python-headless`, `piexif`,
`scikit-learn`, `matplotlib`, `pyyaml`):

- `viser` and `nerfview` (pin the commit used by `examples/requirements.txt`);
- **`splines`** — `nerfview` imports it at module scope but does not declare it,
  so it has to be listed explicitly.

Two practical notes:

- the trainer exports a `.ply` only with `--save_ply`; `--ply_steps` on its own
  produces nothing, because the export is gated on that flag;
- `gsplat/scene` ships as source only in the wheel, but importing and using it
  does not trigger a JIT build, so no compiler is needed for the viewer.

The viewer serves HTTP, so on a cluster it needs an interactive allocation and
an SSH tunnel to the compute node; it cannot be used from a batch job.

## Building a wheel

```bash
# on a node with nvcc; can also be submitted with sbatch (see script header)
bash scripts/build_wheel.sh
```

The wheel lands in `dist/`. Attach it to a GitHub Release, then reference the
asset URL (or use `uv pip install --find-links <dir-or-url> gsplat`).

## Verify any install

```bash
python - <<'PY'
import torch, gsplat
from gsplat.cuda._backend import _C
print("gsplat", gsplat.__version__, "at", gsplat.__file__)
print("ext loaded:", _C is not None)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert "site-packages" in gsplat.__file__, gsplat.__file__
PY
```

If `ext loaded` is `False`, the native extension was not built for this
environment: check that the wheel matches your torch/CUDA/Python, or rebuild
from source with `UV_NO_BUILD_ISOLATION=1`.

Check `gsplat.__file__` as well. A source checkout that appears earlier on
`sys.path` shadows the installed package; its `csrc.so` was built against
whatever torch that checkout was last compiled with, so it fails to load and
gsplat **silently falls back to a JIT build** — no error, just tens of minutes
of compiling before the first real work.

## Validated consumer setup

A separate project installed `+leo2` from its release URL into a clean venv
(Python 3.14, `torch 2.14.1+cu126`), then ran the bundled trainer on mip360
`garden` (factor 4, 2000 steps):

| install | PSNR | SSIM |
| --- | --- | --- |
| wheel `+leo2`, torch 2.14.1 | 23.962 dB | 0.6720 |
| source build, torch 2.14.1 | 23.915 dB | 0.6714 |
| source build, torch 2.9.1 | 23.925 dB | 0.6714 |

The extension loaded from `site-packages` with no JIT fallback, and the web
viewer rendered the resulting checkpoint on an A100. The differences between the
rows are run-to-run variation, not a regression.
