# Environment

You need [uv](https://docs.astral.sh/uv/). One Python 3.11 environment runs all the code in this repository. It needs an NVIDIA GPU, with a driver that supports CUDA 12.4 or newer.

## Install

Run this in the repository root:

```bash
uv venv --python 3.11 --seed .venv && source .venv/bin/activate
uv pip install -r environment/requirements.lock --index-strategy unsafe-best-match
```

**Tip for cluster users:** `uv` keeps its cache in your home folder. If your home has a size or file count limit, set the cache to a folder with enough space before you install: `export UV_CACHE_DIR=<folder>`.

## Files

| File | What it is | When you use it |
|---|---|---|
| `requirements.in` | The short list of the packages that we select, with their version limits. | Edit it to change a package. |
| `requirements.lock` | The exact version of every package, made from `requirements.in`. | Install it. Do not edit it directly. |
| `experiment_envs/` | The separate environments that we used for the paper (`jit`, `lightdit`, `dgm-eval`). | Use them if your results differ. |

## Change a package

Edit `requirements.in`, then make the lock file again:

```bash
uv pip compile environment/requirements.in -o environment/requirements.lock \
    --python-version 3.11 --index-strategy unsafe-best-match
```

- `--python-version 3.11` selects package versions for Python 3.11.
- `--index-strategy unsafe-best-match` lets `uv` use both PyPI and the PyTorch index. Without it, the install and the compile fail, because the PyTorch index has an old `certifi`.

This command selects the newest versions that the limits allow on that day, so other packages can also change. Test the code after you make the lock file again.
