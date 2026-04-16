#!/bin/sh
set -eu

if [ ! -x ".venv/bin/python" ]; then
    if [ -e ".venv" ]; then
        backup_name=".venv_broken_$(date +%Y%m%d_%H%M%S)"
        mv ".venv" "${backup_name}"
    fi
    uv sync
fi

exec uv run python -m marimo run app.py
