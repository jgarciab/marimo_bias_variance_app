# marimo_bias_variance_app

Interactive marimo app for Lecture 3 of aDAV: choosing model complexity, understanding overfitting, and connecting validation and cross-validation to the bias-variance story.

## Run locally

```bash
uv sync
uv run marimo run app.py
```

## Files

- `app.py`: the single-file marimo app
- `pyproject.toml`: local dependencies for running and exporting the app
- `.github/workflows/deploy.yml`: builds the WASM app and deploys it to GitHub Pages
- `build/`: optional local WASM export generated with marimo; ignored by git

## Export to WASM locally

```bash
bash ./export_wasm.sh
```

## Live app

GitHub Actions builds and deploys the WASM app on every push to `main`. After GitHub Pages is enabled with source set to GitHub Actions, the app should be available at:

`https://personalwebsite.github.io/marimo_bias_variance_app/`
