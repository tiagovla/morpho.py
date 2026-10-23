# morpho.py

`morpho.py` computes photonic crystal band structures with the plane-wave
expansion (PWE) method. It supports one-, two-, and three-dimensional periodic
geometries, with scalar TE/TM solvers in 1D and 2D and a vector Maxwell solver
in 3D.

Start with the [quick start](quickstart.md), browse the [API reference](api.md),
or run one of the [examples](examples.md).

## Install

The project uses [uv](https://docs.astral.sh/uv/). From the repository root,
install the package and development tools with:

```console
uv sync
```

## Documentation

To preview these pages locally, run:

```console
uv run mkdocs serve
```

Build a static site with:

```console
uv run mkdocs build --strict
```
