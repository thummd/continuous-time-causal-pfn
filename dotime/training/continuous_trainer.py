"""Training pipeline for :class:`ContinuousDoOverTimePFN`.

Parallel to :mod:`dotime.training.trainer` but replaces
:class:`TemporalInterventionDataLoader` with
:class:`ContinuousTemporalInterventionDataLoader` and
:class:`DoOverTimePFN` with :class:`ContinuousDoOverTimePFN`.

Scoping differences from the discrete trainer
---------------------------------------------
- **Quantile head only.**  Bar-distribution calibration assumes a
  fixed discretisation of the outcome space sampled from the prior;
  integrating that with the continuous prior would require an extra
  calibration pass at every schedule change.  The quantile head with
  pinball loss is a strictly simpler alternative and is what we use in
  the workshop experiments.
- **No observational-only ablation.**  The ablation is trivial to add
  (zero the intervention spec in the batch) but we omit it for a
  smaller surface area; callers who need it can wrap the training loop
  themselves.
- **Structure-only training.**  :class:`ContinuousExtendedPrior` only
  supports named TSCM structures today (no random graph sampler).  The
  discrete trainer's full CTP path isn't mirrored here.

Everything else -- cosine-with-warmup LR, AdamW, gradient clipping,
background prefetch, early stopping, Wandb logging -- is the same.
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional

import torch
import torch.nn as nn

from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.bar_head import calibrate_bar_distribution
from dotime.model.continuous import ContinuousDoOverTimePFN

try:
    import wandb  # noqa: F401
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _evaluate(model, dataloader, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0
    for batch in dataloader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        output = model(batch)
        loss = model.head.loss(output, batch["Y_true_norm"])
        mean_pred = model.head.predict_mean(output)
        rmse = torch.sqrt(((mean_pred - batch["Y_true_norm"]) ** 2).mean())
        total_loss += float(loss.item())
        total_rmse += float(rmse.item())
        n_batches += 1
    model.train()
    return {
        "eval_loss": total_loss / max(1, n_batches),
        "eval_rmse": total_rmse / max(1, n_batches),
    }


def train_continuous(
    # Model config
    n_max: int = 16,
    embed_size: int = 256,
    n_heads: int = 4,
    n_encoder_layers: int = 4,
    n_cross_attn_heads: int = 4,
    encoder_backend: str = "transformer",
    encoder_config: Optional[dict] = None,
    context_window: int = 128,
    n_mixer_layers: int = 1,
    num_time_frequencies: int = 64,
    time_min_freq: float = 0.01,
    time_max_freq: float = 10.0,
    tau_levels: Optional[List[float]] = None,
    head_type: str = "quantile",
    n_buckets: int = 1000,
    bucket_calibration_samples: int = 10000,
    # Prior config
    prior_mode: str = "tscm",
    n_min_prior: int = 3,
    n_max_prior: int = 10,
    edge_prob: float = 0.3,
    hidden_prob: float = 0.0,
    regime_prob: float = 0.0,
    regime_count_range: tuple = (2, 3),
    sticky_alpha: float = 9.0,
    other_alpha: float = 0.5,
    mechanism_kind: str = "linear",
    p_neural: float = 0.0,
    neural_hidden_dim: int = 8,
    neural_out_scale_range: tuple = (0.5, 2.0),
    num_substeps: int = 1,
    tscm_structure: str = "back_door",
    schedule: str = "regular",
    dt: float = 1.0,
    jitter: float = 0.3,
    exp_rate: float = 1.0,
    pair_mode: str = "counterfactual",
    t_range: tuple = (50, 200),
    intervention_kind_probs: tuple = (1.0, 0.0, 0.0),
    intervention_source: str = "prior",
    intervention_value_scale: float = 2.0,
    intervention_window_frac: tuple = (0.1, 0.3),
    soft_shift_scale: float = 1.0,
    time_varying_profile: str = "random",
    theta_range: tuple = (0.5, 2.0),
    sigma_range: tuple = (0.2, 0.6),
    weight_scale: float = 0.5,
    # Training config
    batch_size: int = 32,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    total_steps: int = 5000,
    grad_clip: float = 1.0,
    eval_every: int = 500,
    seed: int = 42,
    device: str = "cpu",
    save_dir: str = "checkpoints/ct",
    prefetch: int = 2,
    target_key: str = "Y_true",
    n_queries: int = 1,
    query_mode: str = "single",
    eval_num_steps: int = 20,
    early_stop_patience: int = 0,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> ContinuousDoOverTimePFN:
    """Train a :class:`ContinuousDoOverTimePFN` on a synthetic continuous-time prior.

    Returns the trained model.  The best checkpoint (by eval loss) is
    saved to ``{save_dir}/continuous_do_over_time_pfn_best.pt``.
    """
    use_wandb = bool(wandb_project) and _WANDB_AVAILABLE
    if wandb_project and not _WANDB_AVAILABLE:
        print("WARNING: wandb_project set but wandb package not installed. Skipping.")
    if use_wandb:
        import wandb  # noqa: F811
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={
                "n_max": n_max, "embed_size": embed_size,
                "n_encoder_layers": n_encoder_layers, "n_heads": n_heads,
                "n_cross_attn_heads": n_cross_attn_heads,
                "context_window": context_window, "n_mixer_layers": n_mixer_layers,
                "num_time_frequencies": num_time_frequencies,
                "tscm_structure": tscm_structure, "schedule": schedule,
                "dt": dt, "jitter": jitter, "exp_rate": exp_rate,
                "pair_mode": pair_mode, "t_range": list(t_range),
                "intervention_kind_probs": list(intervention_kind_probs),
                "intervention_source": intervention_source,
                "intervention_value_scale": intervention_value_scale,
                "intervention_window_frac": list(intervention_window_frac),
                "soft_shift_scale": soft_shift_scale,
                "time_varying_profile": time_varying_profile,
                "theta_range": list(theta_range), "sigma_range": list(sigma_range),
                "weight_scale": weight_scale,
                "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                "warmup_steps": warmup_steps, "total_steps": total_steps,
                "grad_clip": grad_clip, "eval_every": eval_every,
                "target_key": target_key, "n_queries": n_queries,
                "query_mode": query_mode, "seed": seed,
            },
        )

    print("=" * 70)
    print("Continuous-time Do-Over-Time-PFN Training")
    print("=" * 70)
    if prior_mode == "random":
        regime_msg = ""
        if regime_prob > 0.0:
            regime_msg = (
                f", regime_prob={regime_prob} (R in "
                f"[{regime_count_range[0]}, {regime_count_range[1]}])"
            )
        print(
            f"   Prior: random-graph (N in [{n_min_prior}, {n_max_prior}], "
            f"edge_prob={edge_prob}, hidden_prob={hidden_prob}{regime_msg})"
            f"  |  schedule: {schedule}  |  pair_mode: {pair_mode}"
        )
    else:
        print(f"   Prior: {tscm_structure}  |  schedule: {schedule}  |  pair_mode: {pair_mode}")
    print(f"   Intervention kind probs: {intervention_kind_probs}  |  source: {intervention_source}")

    # 1. Model.  Head choice supports "quantile" (default, no calibration)
    #    or "bar" (classification-as-regression, needs bucket borders
    #    sampled from the prior before training starts).
    if head_type not in ("quantile", "bar"):
        raise ValueError(f"invalid head_type: {head_type!r}")
    tau_levels = tau_levels or [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"   Head type: {head_type}")
    model = ContinuousDoOverTimePFN(
        n_max=n_max,
        embed_size=embed_size,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_cross_attn_heads=n_cross_attn_heads,
        encoder_backend=encoder_backend,
        encoder_config=encoder_config,
        head_type=head_type,
        tau_levels=tau_levels,
        n_buckets=n_buckets,
        n_mixer_layers=n_mixer_layers,
        context_window=context_window,
        num_time_frequencies=num_time_frequencies,
        time_min_freq=time_min_freq,
        time_max_freq=time_max_freq,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {num_params:,}")

    # 1b. Bucket-border calibration for bar head.
    borders = None
    if head_type == "bar":
        cal_steps = max(bucket_calibration_samples // batch_size, 10)
        print(f"   Calibrating {n_buckets} bar-distribution buckets "
              f"from {cal_steps} prior batches...")
        cal_loader = ContinuousTemporalInterventionDataLoader(
            num_steps=cal_steps,
            seed=seed + 5_000,
            device="cpu",  # calibration only reads Y_true_norm scalars
            prefetch=0,
            **dict(
                batch_size=batch_size,
                prior_mode=prior_mode,
                tscm_structure=tscm_structure,
                n_min_prior=n_min_prior,
                n_max_prior=n_max_prior,
                edge_prob=edge_prob,
                hidden_prob=hidden_prob,
                regime_prob=regime_prob,
                regime_count_range=regime_count_range,
                sticky_alpha=sticky_alpha,
                other_alpha=other_alpha,
                mechanism_kind=mechanism_kind,
                p_neural=p_neural,
                neural_hidden_dim=neural_hidden_dim,
                neural_out_scale_range=neural_out_scale_range,
                num_substeps=num_substeps,
                schedule=schedule,
                dt=dt,
                jitter=jitter,
                exp_rate=exp_rate,
                pair_mode=pair_mode,
                t_range=t_range,
                n_max=n_max,
                normalize=True,
                target_key=target_key,
                n_queries=n_queries,
                query_mode=query_mode,
                theta_range=theta_range,
                sigma_range=sigma_range,
                weight_scale=weight_scale,
                intervention_value_scale=intervention_value_scale,
                intervention_window_frac=intervention_window_frac,
            ),
        )
        bar_dist, borders = calibrate_bar_distribution(cal_loader, n_buckets=n_buckets)
        model.bar_head.set_bar_distribution(bar_dist, borders.to(device))
        print(f"   Border range: [{borders.min():.3f}, {borders.max():.3f}]")

    # 2. Data loaders (train + eval with separate seeds)
    loader_kwargs = dict(
        batch_size=batch_size,
        prior_mode=prior_mode,
        tscm_structure=tscm_structure,
        n_min_prior=n_min_prior,
        n_max_prior=n_max_prior,
        edge_prob=edge_prob,
        hidden_prob=hidden_prob,
        regime_prob=regime_prob,
        regime_count_range=regime_count_range,
        sticky_alpha=sticky_alpha,
        other_alpha=other_alpha,
        mechanism_kind=mechanism_kind,
        p_neural=p_neural,
        neural_hidden_dim=neural_hidden_dim,
        neural_out_scale_range=neural_out_scale_range,
        num_substeps=num_substeps,
        schedule=schedule,
        dt=dt,
        jitter=jitter,
        exp_rate=exp_rate,
        pair_mode=pair_mode,
        t_range=t_range,
        n_max=n_max,
        normalize=True,
        target_key=target_key,
        n_queries=n_queries,
        query_mode=query_mode,
        theta_range=theta_range,
        sigma_range=sigma_range,
        weight_scale=weight_scale,
        intervention_value_scale=intervention_value_scale,
        intervention_window_frac=intervention_window_frac,
    )

    train_loader = ContinuousTemporalInterventionDataLoader(
        num_steps=total_steps,
        seed=seed,
        device=device,
        prefetch=prefetch,
        **loader_kwargs,
    )
    eval_loader = ContinuousTemporalInterventionDataLoader(
        num_steps=eval_num_steps,
        seed=seed + 10_000,
        device=device,
        prefetch=0,
        **loader_kwargs,
    )

    # 3. Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    scheduler = _cosine_with_warmup(optimizer, warmup_steps, total_steps)

    # 4. Training loop
    os.makedirs(save_dir, exist_ok=True)
    step_log_path = os.path.join(save_dir, "ct_step_losses.csv")
    step_log_file = open(step_log_path, "w")
    step_log_file.write("step,loss,y_max,lr\n")

    model.train()
    best_eval_loss = float("inf")
    n_evals_since_best = 0
    start_time = time.time()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch)

        y_max = float(batch["Y_true_norm"].abs().max().item())
        if y_max > 20:  # extreme normalization edge case
            continue

        loss = model.head.loss(output, batch["Y_true_norm"])
        if not torch.isfinite(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        running_loss += loss_val
        lr_now = scheduler.get_last_lr()[0]
        step_log_file.write(f"{step},{loss_val:.6f},{y_max:.4f},{lr_now:.2e}\n")
        if step % 100 == 0:
            step_log_file.flush()

        if use_wandb:
            import wandb  # noqa: F811
            wandb.log({"step_loss": loss_val, "y_max": y_max, "lr": lr_now}, step=step)

        if (step + 1) % eval_every == 0:
            avg_train_loss = running_loss / eval_every
            running_loss = 0.0

            metrics = _evaluate(model, eval_loader, device)
            eval_loss = metrics["eval_loss"]
            eval_rmse = metrics["eval_rmse"]
            elapsed = time.time() - start_time
            print(
                f"   Step {step+1}/{total_steps} ({elapsed:.0f}s)  |  "
                f"train {avg_train_loss:.4f}  eval {eval_loss:.4f}  "
                f"rmse {eval_rmse:.4f}  lr {lr_now:.2e}"
            )
            if use_wandb:
                import wandb  # noqa: F811
                wandb.log(
                    {
                        "train_loss": avg_train_loss,
                        "eval_loss": eval_loss,
                        "eval_rmse": eval_rmse,
                    },
                    step=step + 1,
                )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                n_evals_since_best = 0
                save_path = os.path.join(save_dir, "continuous_do_over_time_pfn_best.pt")
                torch.save(
                    {
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "eval_loss": eval_loss,
                        "eval_rmse": eval_rmse,
                        "borders": borders,
                        "config": {
                            "n_max": n_max,
                            "embed_size": embed_size,
                            "n_encoder_layers": n_encoder_layers,
                            "n_cross_attn_heads": n_cross_attn_heads,
                            "encoder_backend": encoder_backend,
                            "context_window": context_window,
                            "n_mixer_layers": n_mixer_layers,
                            "num_time_frequencies": num_time_frequencies,
                            "time_min_freq": time_min_freq,
                            "time_max_freq": time_max_freq,
                            "tau_levels": tau_levels,
                            "head_type": head_type,
                            "n_buckets": n_buckets,
                            "tscm_structure": tscm_structure,
                            "schedule": schedule,
                            "pair_mode": pair_mode,
                        },
                    },
                    save_path,
                )
            else:
                n_evals_since_best += 1

            if early_stop_patience > 0 and n_evals_since_best >= early_stop_patience:
                print(
                    f"   Early stopping at step {step+1} "
                    f"(no improvement for {n_evals_since_best} evals; "
                    f"best_eval_loss={best_eval_loss:.4f})"
                )
                break

    total_time = time.time() - start_time
    step_log_file.close()
    print(f"\nTraining complete in {total_time:.0f}s.  Best eval loss: {best_eval_loss:.4f}")

    if use_wandb:
        import wandb  # noqa: F811
        wandb.summary["best_eval_loss"] = best_eval_loss
        wandb.summary["total_time_s"] = total_time
        wandb.finish()

    return model
