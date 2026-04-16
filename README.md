# marimo_bias_variance_app

Interactive marimo app for Lecture 3 of aDAV: choosing model complexity, understanding overfitting, and connecting validation and cross-validation to the bias-variance story.

## Run locally

```bash
uv sync
sh ./run.sh
```

## Files

- `app.py`: the single-file marimo app
- `pyproject.toml`: local dependencies for running and exporting the app
- `docs/`: GitHub Pages / Pyodide export generated with marimo

## Export to WASM

```bash
uv run python -m marimo export html-wasm app.py -o docs --mode run -f
```

## Live app

After GitHub Pages is enabled for this repository, the app should be available at:

`https://personalwebsite.github.io/marimo_bias_variance_app/`
