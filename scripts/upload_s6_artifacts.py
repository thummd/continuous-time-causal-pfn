#!/usr/bin/env python
"""Upload s6 checkpoints as wandb artifacts to each run.

One-off script. For future runs, the trainer will call log_artifact
automatically (Part a of the plan).
"""

import os
import sys
import wandb

REPO_DIR = os.path.expanduser("~/repos/do-over-time-pfn")
PROJECT = "dot-pfn/do-over-time-pfn"

RUNS = [
    ("s6_bd_nolag_interp_causal", "f5i8g5n1"),
    ("s6_bd_nolag_interp_obs",    "t546slrm"),
    ("s6_fd_nolag_interp_causal", "h15alxa7"),
    ("s6_fd_nolag_interp_obs",    "r6hz4pba"),
]


def upload_one(name: str, run_id: str) -> None:
    ckpt_path = os.path.join(REPO_DIR, "checkpoints", name, "do_over_time_pfn_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"MISSING: {ckpt_path}")
        return

    print(f"\n=== {name} (run {run_id}) ===")
    print(f"  ckpt: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:.1f} MB)")

    # Resume the finished run so the artifact is attached to it
    run = wandb.init(
        project="do-over-time-pfn",
        entity="dot-pfn",
        id=run_id,
        resume="must",
    )

    artifact_name = f"{name}_best"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"Best checkpoint for {name} (s6 quick-validation)",
        metadata={
            "run_id": run_id,
            "run_name": name,
            "experiment": "s6",
            "fixes": "2a+2b+2c+2d",
            "steps": 1000,
            "n_queries": 10,
            "sim_device": "cpu",
        },
    )
    artifact.add_file(ckpt_path, name="do_over_time_pfn_best.pt")

    # Also add step_losses.csv for diagnostics
    step_losses = os.path.join(REPO_DIR, "checkpoints", name, "step_losses.csv")
    if os.path.exists(step_losses):
        artifact.add_file(step_losses, name="step_losses.csv")

    run.log_artifact(artifact, aliases=["best", "s6"])
    run.finish()
    print(f"  uploaded as artifact '{artifact_name}'")


def main():
    for name, run_id in RUNS:
        try:
            upload_one(name, run_id)
        except Exception as e:
            print(f"ERROR uploading {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
