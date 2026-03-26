"""Unified entrypoint for V-JEPA2 inference container.

Usage:
    python -m app serve          # start API server (default)
    python -m app infer [FILE]   # run inference
    python -m app download       # download model
"""
from app.cli import main

main()
