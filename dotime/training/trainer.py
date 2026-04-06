"""Training pipeline for Do-Over-Time-PFN.

On-the-fly data generation with bar distribution loss,
AdamW optimizer, cosine LR with warmup.
"""

import os
import time
import math
import torch
import torch.nn as nn
from typing import List, Optional

from dotime.model.do_over_time_pfn import DoOverTimePFN
from dotime.model.bar_head import calibrate_bar_distribution
from dotime.data.temporal_dataloader import TemporalInterventionDataLoader


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, dataloader, device, tau_levels=None):
    """Evaluate on a set of batches, returning mean loss, RMSE, and optionally pinball loss."""
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_pinball = 0
    n_batches = 0

    head = model.head

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = model(batch)
        loss = head.loss(output, batch['Y_true_norm'])
        total_loss += loss.item()

        mean_pred = head.predict_mean(output)
        rmse = torch.sqrt(torch.mean((mean_pred - batch['Y_true_norm']) ** 2))
        total_rmse += rmse.item()

        if tau_levels is not None and hasattr(head, 'compute_pinball_loss'):
            pb_loss = head.compute_pinball_loss(
                output, batch['Y_true_norm'], tau_levels)
            total_pinball += pb_loss.item()

        n_batches += 1

    model.train()
    results = (total_loss / max(n_batches, 1), total_rmse / max(n_batches, 1))
    if tau_levels is not None:
        results = results + (total_pinball / max(n_batches, 1),)
    return results


def train(
    # Model config
    n_max: int = 41,
    embed_size: int = 512,
    n_heads: int = 4,
    n_encoder_layers: int = 10,
    n_cross_attn_heads: int = 4,
    n_buckets: int = 1000,
    encoder_backend: str = "transformer",
    encoder_config: Optional[dict] = None,
    # Prior config
    n_max_prior: int = 10,
    t_range: tuple = (50, 200),
    burn_in: int = 50,
    downstream_prob: float = 0.7,
    # Training config
    batch_size: int = 64,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    grad_clip: float = 100.0,
    eval_every: int = 500,
    bucket_calibration_samples: int = 10000,
    seed: int = 42,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    pinball_weight: float = 0.0,
    pinball_quantiles: Optional[List[float]] = None,
    num_workers: int = 0,
    prefetch: int = 2,
    head_type: str = "bar",
    tau_levels: Optional[List[float]] = None,
    target_key: str = "Y_true",
    observational_only: bool = False,
    n_queries: int = 1,
    query_mode: str = "single",
    n_mixer_layers: int = 1,
    intervention_source: str = "prior",
):
    """Full training pipeline for Do-Over-Time-PFN."""

    print("=" * 70)
    print("Do-Over-Time-PFN Training")
    print("=" * 70)
    print(f"   Head type: {head_type}")
    print(f"   Target: {target_key}")
    if n_queries > 1:
        print(f"   Dense supervision: {n_queries} queries per trajectory")
    if observational_only:
        print("   ABLATION: observational-only (intervention context zeroed)")

    # 1. Calibrate bar distribution (only for bar head)
    borders = None
    if head_type == "bar":
        print(f"\n1. Calibrating bar distribution ({bucket_calibration_samples} samples, {n_buckets} buckets)...")
        cal_steps = max(bucket_calibration_samples // batch_size, 10)
        cal_loader = TemporalInterventionDataLoader(
            num_steps=cal_steps, batch_size=batch_size,
            n_max=n_max, n_max_prior=n_max_prior,
            t_range=t_range, burn_in=burn_in,
            downstream_prob=downstream_prob, seed=seed + 1000,
        )
        bar_dist, borders = calibrate_bar_distribution(cal_loader, n_buckets=n_buckets)
        print(f"   Borders range: [{borders.min():.2f}, {borders.max():.2f}]")
    else:
        print("\n1. Skipping bar calibration (quantile head).")

    # 2. Create model
    print(f"\n2. Creating model (embed={embed_size}, layers={n_encoder_layers}, backend={encoder_backend})...")
    model = DoOverTimePFN(
        n_max=n_max, embed_size=embed_size, n_heads=n_heads,
        n_encoder_layers=n_encoder_layers, n_cross_attn_heads=n_cross_attn_heads,
        n_buckets=n_buckets, encoder_backend=encoder_backend,
        encoder_config=encoder_config,
        head_type=head_type,
        tau_levels=tau_levels,
        n_mixer_layers=n_mixer_layers,
    )
    if head_type == "bar":
        model.bar_head.set_bar_distribution(bar_dist, borders.to(device))
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {num_params:,}")

    # Pinball loss setup
    tau_levels = None
    if pinball_weight > 0:
        if pinball_quantiles is None:
            pinball_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        tau_levels = torch.tensor(pinball_quantiles, device=device, dtype=torch.float32)
        print(f"   Pinball loss: weight={pinball_weight}, quantiles={pinball_quantiles}")

    # 3. Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.98), eps=1e-6,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 4. Training dataloader
    print(f"\n3. Starting training ({total_steps} steps, batch_size={batch_size})...")
    train_loader = TemporalInterventionDataLoader(
        num_steps=total_steps, batch_size=batch_size,
        n_max=n_max, n_max_prior=n_max_prior,
        t_range=t_range, burn_in=burn_in,
        downstream_prob=downstream_prob, seed=seed,
        device=device,
        num_workers=num_workers,
        prefetch=prefetch,
        target_key=target_key,
        n_queries=n_queries,
        query_mode=query_mode,
        intervention_source=intervention_source,
    )

    # Eval loader (fixed seed for consistent evaluation)
    eval_loader = TemporalInterventionDataLoader(
        num_steps=20, batch_size=batch_size,
        n_max=n_max, n_max_prior=n_max_prior,
        t_range=t_range, burn_in=burn_in,
        downstream_prob=downstream_prob, seed=seed + 2000,
        device=device,
    )

    # 5. Training loop
    model.train()
    best_eval_loss = float('inf')
    start_time = time.time()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Observational-only ablation: zero out intervention context
        if observational_only:
            batch['intervention_target'] = torch.zeros_like(batch['intervention_target'])
            batch['intervention_type'] = torch.zeros_like(batch['intervention_type'])
            batch['intervention_value'] = torch.zeros_like(batch['intervention_value'])
            batch['intervention_time_start'] = torch.zeros_like(batch['intervention_time_start'])
            batch['intervention_time_end'] = torch.zeros_like(batch['intervention_time_end'])

        # Encoder caching: compute h_vars once per trajectory, reuse for all queries
        if '_traj_idx' in batch:
            h_vars = model.encode(batch)  # (B, N_max, E) — B unique trajectories
            traj_idx = batch['_traj_idx']  # (B_total,) maps queries to trajectories
            h_vars_expanded = h_vars[traj_idx]  # (B_total, N_max, E)
            # Expand variable_mask to match queries
            query_batch = {k: v for k, v in batch.items() if k not in ('X_obs_norm', '_traj_idx')}
            query_batch['variable_mask'] = batch['variable_mask'][traj_idx]
            output = model.query(h_vars_expanded, query_batch)
        else:
            output = model(batch)

        # Skip batches with extreme normalized targets (normalization edge case)
        y_max = batch['Y_true_norm'].abs().max().item()
        if y_max > 20:
            print(f"   Step {step}: extreme Y_true_norm={y_max:.1f}, skipping")
            continue

        if head_type == "quantile":
            loss = model.quantile_head.loss(output, batch['Y_true_norm'])
        else:
            bar_loss = model.bar_head.loss(output, batch['Y_true_norm'])
            if tau_levels is not None:
                pb_loss = model.bar_head.compute_pinball_loss(
                    output, batch['Y_true_norm'], tau_levels)
                loss = bar_loss + pinball_weight * pb_loss
            else:
                loss = bar_loss

        if not torch.isfinite(loss):
            print(f"   Step {step}: NaN/Inf loss, skipping")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Log + evaluate
        if (step + 1) % eval_every == 0:
            avg_train_loss = running_loss / eval_every
            running_loss = 0.0

            eval_results = evaluate(model, eval_loader, device, tau_levels=tau_levels)
            eval_loss, eval_rmse = eval_results[0], eval_results[1]

            elapsed = time.time() - start_time
            lr_now = scheduler.get_last_lr()[0]
            log_msg = (f"   Step {step+1}/{total_steps} ({elapsed:.0f}s) | "
                       f"Train: {avg_train_loss:.4f} | Eval: {eval_loss:.4f} | "
                       f"RMSE: {eval_rmse:.4f} | LR: {lr_now:.2e}")
            if tau_levels is not None:
                eval_pinball = eval_results[2]
                log_msg += f" | Pinball: {eval_pinball:.4f}"
            print(log_msg)

            # Save best
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "do_over_time_pfn_best.pt")
                torch.save({
                    'step': step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_loss': eval_loss,
                    'eval_rmse': eval_rmse,
                    'borders': borders,
                    'head_type': head_type,
                    'config': {
                        'n_max': n_max, 'embed_size': embed_size,
                        'n_heads': n_heads, 'n_encoder_layers': n_encoder_layers,
                        'n_cross_attn_heads': n_cross_attn_heads,
                        'n_buckets': n_buckets, 'encoder_backend': encoder_backend,
                        'head_type': head_type,
                        'tau_levels': tau_levels,
                        'n_mixer_layers': n_mixer_layers,
                    },
                }, save_path)

    total_time = time.time() - start_time
    print(f"\n4. Training complete! Total time: {total_time:.0f}s")
    print(f"   Best eval loss: {best_eval_loss:.4f}")

    return model
