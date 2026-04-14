#!/usr/bin/env python
"""Plot learning curves from training log files.

Parses the step-log lines written by trainer.py and produces a multi-panel
figure with train loss, eval loss, and eval RMSE over training steps.

Usage
-----
    python scripts/plot_learning_curves.py \
        --logs results/sanity_bd_causal/train.log "BD Causal" \
              results/sanity_bd_obs_only/train.log "BD Obs-Only" \
        --out figures/sanity_bd_curves.png

    # Or for multiple experiments:
    python scripts/plot_learning_curves.py \
        --logs results/sanity_bd_causal/train.log   "BD Causal" \
              results/sanity_bd_obs_only/train.log  "BD Obs-Only" \
              results/sanity_fd_causal/train.log    "FD Causal" \
              results/sanity_fd_obs_only/train.log  "FD Obs-Only" \
        --out figures/sanity_learning_curves.png
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(path):
    """Extract (step, train_loss, eval_loss, rmse, lr) from a trainer log."""
    pattern = re.compile(
        r'Step\s+(\d+)/\d+.*?Train:\s+([\d.]+).*?Eval:\s+([-\d.]+).*?RMSE:\s+([\d.]+).*?LR:\s+([\d.eE+-]+)'
    )
    steps, trains, evals, rmses, lrs = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                trains.append(float(m.group(2)))
                evals.append(float(m.group(3)))
                rmses.append(float(m.group(4)))
                lrs.append(float(m.group(5)))
    return {
        'step': np.array(steps),
        'train_loss': np.array(trains),
        'eval_loss': np.array(evals),
        'eval_rmse': np.array(rmses),
        'lr': np.array(lrs),
    }


def parse_step_csv(path):
    """Parse per-step CSV from trainer (step_losses.csv)."""
    steps, losses, ymaxs, lrs = [], [], [], []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                steps.append(int(parts[0]))
                losses.append(float(parts[1]))
                ymaxs.append(float(parts[2]))
                if len(parts) >= 4:
                    lrs.append(float(parts[3]))
    return {
        'step': np.array(steps),
        'loss': np.array(losses),
        'y_max': np.array(ymaxs),
        'lr': np.array(lrs) if lrs else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', required=True,
                        help='Alternating: log_path label log_path label ...')
    parser.add_argument('--out', type=str, default='figures/learning_curves.png')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--per-step', action='store_true',
                        help='If set, treat --logs as step_losses.csv files (per-iteration)')
    parser.add_argument('--smooth', type=int, default=10,
                        help='Smoothing window for per-step plots (default: 10)')
    args = parser.parse_args()

    # Parse --logs pairs
    paths_labels = []
    it = iter(args.logs)
    for path in it:
        label = next(it, os.path.basename(os.path.dirname(path)))
        paths_labels.append((path, label))

    if args.per_step:
        # Per-iteration loss plots from step_losses.csv
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        colors = plt.cm.tab10.colors
        for i, (path, label) in enumerate(paths_labels):
            data = parse_step_csv(path)
            if data['step'].size == 0:
                print(f"WARNING: no data in {path}")
                continue
            color = colors[i % len(colors)]
            # Raw loss (transparent) + smoothed
            axes[0].plot(data['step'], data['loss'], color=color, alpha=0.15, linewidth=0.5)
            if data['step'].size >= args.smooth:
                kernel = np.ones(args.smooth) / args.smooth
                smoothed = np.convolve(data['loss'], kernel, mode='valid')
                axes[0].plot(data['step'][args.smooth-1:], smoothed,
                             color=color, alpha=0.9, linewidth=1.5, label=label)
            else:
                axes[0].plot(data['step'], data['loss'], color=color, alpha=0.9, label=label)
            # y_max (max |Y_true_norm| per batch)
            axes[1].plot(data['step'], data['y_max'], color=color, alpha=0.4, linewidth=0.5, label=label)
        axes[0].set_ylabel('Loss (per step)')
        axes[1].set_ylabel('max |Y_true_norm|')
        for ax in axes:
            ax.set_xlabel('Step')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
        colors = plt.cm.tab10.colors

        for i, (path, label) in enumerate(paths_labels):
            data = parse_log(path)
            if data['step'].size == 0:
                print(f"WARNING: no data in {path}")
                continue
            color = colors[i % len(colors)]
            axes[0].plot(data['step'], data['train_loss'], label=label, color=color, alpha=0.85)
            axes[1].plot(data['step'], data['eval_loss'], label=label, color=color, alpha=0.85)
            axes[2].plot(data['step'], data['eval_rmse'], label=label, color=color, alpha=0.85)

        axes[0].set_ylabel('Train Loss')
        axes[1].set_ylabel('Eval Loss')
        axes[2].set_ylabel('Eval RMSE')
        for ax in axes:
            ax.set_xlabel('Step')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    if args.title:
        fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if args.title else None)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
