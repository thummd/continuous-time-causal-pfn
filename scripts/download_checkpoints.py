#!/usr/bin/env python
"""Download pretrained checkpoints from the Hugging Face model repo.

Checkpoints are hosted on Hugging Face rather than in this git repository:
    https://huggingface.co/thummd/continuous-time-causal-pfn

Examples
--------
    python scripts/download_checkpoints.py                 # everything
    python scripts/download_checkpoints.py --subset grid_v4  # one subset
    python scripts/download_checkpoints.py --out ./checkpoints
"""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

REPO_ID = "thummd/continuous-time-causal-pfn"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--subset", default=None,
        help="Only fetch a subdirectory (e.g. 'grid_v4'). Default: all.",
    )
    ap.add_argument(
        "--out", default="checkpoints",
        help="Local output directory (default: ./checkpoints).",
    )
    args = ap.parse_args()

    allow = [f"{args.subset}/*"] if args.subset else None
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        allow_patterns=allow,
        local_dir=args.out,
    )
    print(f"Downloaded checkpoints to: {path}")


if __name__ == "__main__":
    main()
