"""Zero-shot CausalChamber evaluation for :class:`ContinuousDoOverTimePFN`.

Companion to :mod:`continuous_pk_eval` that targets the
CausalChamber (Gamella et al. 2024) physical system benchmark.
CausalChamber provides episodes at roughly 10 Hz where an actuator
setpoint changes at a known time -- a natural fit for the
continuous-time model's ``(intervention_target, intervention_value,
t_int_start)`` contract.

Reports both per-episode and aggregate RMSE / MAE / Pearson r between
the model's predicted query-variable trajectory and the true
post-intervention observations, together with lift over a naive
baseline that simply holds the last pre-intervention value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from dotime.data.causal_chamber_ct import build_causal_chamber_batch
from dotime.eval.continuous_pk_eval import _pearson_r, _denormalise_prediction
from dotime.model.continuous import ContinuousDoOverTimePFN


@dataclass
class EpisodeMetrics:
    """Per-episode zero-shot chamber metrics."""

    experiment: str
    changepoint: int
    query_var: str
    n_queries: int
    rmse: float
    mae: float
    pearson_r: float
    mean_pred: float
    mean_gt: float


@dataclass
class ChamberAggregateMetrics:
    """Aggregate metrics across all evaluated episodes."""

    n_episodes: int
    n_queries: int
    macro_rmse: float
    macro_mae: float
    pooled_rmse: float
    pooled_mae: float
    naive_pooled_rmse: float
    lift_over_naive: float
    mean_pearson_r: float


def _naive_prediction_from_episode(
    episode: Dict,
    query_var: str,
    n_queries: int,
) -> torch.Tensor:
    """Last pre-intervention value of ``query_var``, broadcast to n_queries."""
    var_names: List[str] = list(episode["var_names"])
    idx = var_names.index(query_var)
    last_obs = float(episode["X_obs"][-1, idx])
    return torch.full((n_queries,), last_obs, dtype=torch.float32)


@torch.no_grad()
def evaluate_episode(
    model: ContinuousDoOverTimePFN,
    episode: Dict,
    query_var: str,
    device: str = "cpu",
    **adapter_kwargs,
) -> Dict[str, torch.Tensor]:
    """Run the model on one chamber episode and return metrics + raw predictions."""
    batch = build_causal_chamber_batch(
        episode, query_var=query_var, **adapter_kwargs,
    )
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    model.eval()
    output = model(batch)
    pred_norm = model.head.predict_mean(output)
    pred_raw = _denormalise_prediction(pred_norm, batch).cpu()
    gt_raw = batch["Y_true"].cpu()

    err = pred_raw - gt_raw
    metrics = EpisodeMetrics(
        experiment=str(episode.get("experiment", "")),
        changepoint=int(episode.get("changepoint", -1)),
        query_var=query_var,
        n_queries=int(pred_raw.numel()),
        rmse=float(err.pow(2).mean().sqrt().item()),
        mae=float(err.abs().mean().item()),
        pearson_r=_pearson_r(pred_raw, gt_raw),
        mean_pred=float(pred_raw.mean().item()),
        mean_gt=float(gt_raw.mean().item()),
    )

    return {
        "metrics": metrics,
        "predictions": pred_raw,
        "ground_truth": gt_raw,
        "query_times": batch["t_query"].cpu(),
    }


@torch.no_grad()
def evaluate_episodes(
    model: ContinuousDoOverTimePFN,
    episodes: Iterable[Dict],
    query_var: str,
    device: str = "cpu",
    **adapter_kwargs,
) -> Dict[str, object]:
    """Evaluate on every episode and aggregate."""
    model.eval()
    per_episode: List[EpisodeMetrics] = []
    pred_all: List[torch.Tensor] = []
    gt_all: List[torch.Tensor] = []
    naive_all: List[torch.Tensor] = []
    predictions_dump: List[Dict[str, torch.Tensor]] = []

    for ep in episodes:
        r = evaluate_episode(model, ep, query_var=query_var, device=device, **adapter_kwargs)
        per_episode.append(r["metrics"])
        pred_all.append(r["predictions"])
        gt_all.append(r["ground_truth"])
        naive_all.append(_naive_prediction_from_episode(ep, query_var, r["metrics"].n_queries))
        predictions_dump.append(
            {
                "experiment": r["metrics"].experiment,
                "changepoint": r["metrics"].changepoint,
                "predictions": r["predictions"],
                "ground_truth": r["ground_truth"],
                "query_times": r["query_times"],
            }
        )

    if not per_episode:
        raise ValueError("no episodes evaluated")

    pred_pooled = torch.cat(pred_all)
    gt_pooled = torch.cat(gt_all)
    naive_pooled = torch.cat(naive_all)

    pooled_err = pred_pooled - gt_pooled
    naive_err = naive_pooled - gt_pooled

    pooled_rmse = float(pooled_err.pow(2).mean().sqrt().item())
    pooled_mae = float(pooled_err.abs().mean().item())
    naive_rmse = float(naive_err.pow(2).mean().sqrt().item())

    aggregate = ChamberAggregateMetrics(
        n_episodes=len(per_episode),
        n_queries=int(pred_pooled.numel()),
        macro_rmse=float(sum(m.rmse for m in per_episode) / len(per_episode)),
        macro_mae=float(sum(m.mae for m in per_episode) / len(per_episode)),
        pooled_rmse=pooled_rmse,
        pooled_mae=pooled_mae,
        naive_pooled_rmse=naive_rmse,
        lift_over_naive=(naive_rmse - pooled_rmse) / max(naive_rmse, 1e-6),
        mean_pearson_r=float(
            sum((m.pearson_r if m.pearson_r == m.pearson_r else 0.0) for m in per_episode)
            / len(per_episode)
        ),
    )

    return {
        "per_episode": per_episode,
        "aggregate": aggregate,
        "per_episode_predictions": predictions_dump,
    }


def metrics_to_dict(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "per_episode": [asdict(m) for m in result["per_episode"]],
        "aggregate": asdict(result["aggregate"]),
    }


def format_summary(result: Dict[str, object]) -> str:
    agg = result["aggregate"]
    lines = [
        "=" * 60,
        "Zero-shot CausalChamber evaluation",
        "=" * 60,
        f"  Episodes            : {agg.n_episodes}",
        f"  Queries             : {agg.n_queries}",
        f"  Macro-avg RMSE      : {agg.macro_rmse:.3f}",
        f"  Pooled RMSE         : {agg.pooled_rmse:.3f}",
        f"  Naive-baseline RMSE : {agg.naive_pooled_rmse:.3f}",
        f"  Lift over naive     : {agg.lift_over_naive:+.1%}",
        f"  Mean Pearson r      : {agg.mean_pearson_r:+.3f}",
    ]
    return "\n".join(lines)
