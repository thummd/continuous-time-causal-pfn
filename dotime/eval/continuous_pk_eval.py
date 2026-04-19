"""Zero-shot Theophylline evaluation for :class:`ContinuousDoOverTimePFN`.

Thin orchestration layer on top of
:mod:`dotime.data.pk_pd.theophylline_adapter`: takes a trained model
and the Theophylline dataset, produces per-subject and aggregate
metrics, and optionally writes a JSON results file.

Reported metrics
----------------
Computed in *original* mg/L units (after denormalising the model's
quantile outputs with the per-subject ``_norm_means`` /
``_norm_stds``):

- Per subject: RMSE, MAE, mean prediction, mean ground truth, Pearson
  correlation between prediction and GT over that subject's post-dose
  queries.
- Aggregate: macro-averaged RMSE / MAE (mean over subjects), combined
  RMSE (pooled over all queries), mean absolute dose-response slope
  error, and a naive baseline RMSE (predict the subject's mean
  post-dose concentration).  The naive baseline gives a minimum bar
  the trained model is expected to beat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import torch

from dotime.data.pk_pd.theophylline import TheophSubject
from dotime.data.pk_pd.theophylline_adapter import build_theophylline_batch
from dotime.model.continuous import ContinuousDoOverTimePFN


@dataclass
class SubjectMetrics:
    """Per-subject zero-shot PK evaluation metrics."""

    subject_id: int
    n_queries: int
    rmse: float
    mae: float
    pearson_r: float
    mean_pred: float
    mean_gt: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all evaluated subjects."""

    n_subjects: int
    n_queries: int
    macro_rmse: float
    macro_mae: float
    pooled_rmse: float
    pooled_mae: float
    naive_pooled_rmse: float
    lift_over_naive: float
    mean_pearson_r: float


def _denormalise_prediction(
    normalised_pred: torch.Tensor,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Undo the adapter's z-score normalisation for the concentration target."""
    query_target = batch["query_target"]
    means = batch["_norm_means"]
    stds = batch["_norm_stds"]
    batch_idx = torch.arange(query_target.shape[0], device=query_target.device)
    target_mean = means[batch_idx, query_target]
    target_std = stds[batch_idx, query_target]
    return normalised_pred * target_std + target_mean


def _pearson_r(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation.  Returns ``nan`` if either vector is constant."""
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom.item() < 1e-12:
        return float("nan")
    return float((a * b).sum().item() / denom.item())


@torch.no_grad()
def evaluate_subject(
    model: ContinuousDoOverTimePFN,
    subject: TheophSubject,
    device: str = "cpu",
    **adapter_kwargs,
) -> Dict[str, torch.Tensor]:
    """Run the model on one subject and return a dict of metrics + raw predictions."""
    batch = build_theophylline_batch(subject, **adapter_kwargs)
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    model.eval()
    output = model(batch)
    pred_norm = model.head.predict_mean(output)
    pred_mg_per_l = _denormalise_prediction(pred_norm, batch).cpu()

    gt_mg_per_l = batch["Y_true"].cpu()
    err = pred_mg_per_l - gt_mg_per_l
    metrics = SubjectMetrics(
        subject_id=int(batch["_subject_id"][0].item()),
        n_queries=int(pred_mg_per_l.numel()),
        rmse=float(err.pow(2).mean().sqrt().item()),
        mae=float(err.abs().mean().item()),
        pearson_r=_pearson_r(pred_mg_per_l, gt_mg_per_l),
        mean_pred=float(pred_mg_per_l.mean().item()),
        mean_gt=float(gt_mg_per_l.mean().item()),
    )

    return {
        "metrics": metrics,
        "predictions": pred_mg_per_l,
        "ground_truth": gt_mg_per_l,
        "query_times": batch["_query_time_hours"].cpu(),
    }


@torch.no_grad()
def evaluate_dataset(
    model: ContinuousDoOverTimePFN,
    subjects: Iterable[TheophSubject],
    device: str = "cpu",
    **adapter_kwargs,
) -> Dict[str, object]:
    """Evaluate on every subject and aggregate.

    Returns a dict with ``per_subject`` (list of :class:`SubjectMetrics`)
    and ``aggregate`` (:class:`AggregateMetrics`), plus a raw
    ``per_subject_predictions`` list for downstream plotting.
    """
    model.eval()
    per_subject: List[SubjectMetrics] = []
    pred_all: List[torch.Tensor] = []
    gt_all: List[torch.Tensor] = []
    predictions_dump: List[Dict[str, torch.Tensor]] = []

    for subj in subjects:
        r = evaluate_subject(model, subj, device=device, **adapter_kwargs)
        per_subject.append(r["metrics"])
        pred_all.append(r["predictions"])
        gt_all.append(r["ground_truth"])
        predictions_dump.append(
            {
                "subject_id": r["metrics"].subject_id,
                "predictions": r["predictions"],
                "ground_truth": r["ground_truth"],
                "query_times": r["query_times"],
            }
        )

    pred_pooled = torch.cat(pred_all)
    gt_pooled = torch.cat(gt_all)

    # Naive baseline: predict each subject's mean post-dose concentration.
    naive_pred_parts: List[torch.Tensor] = []
    for gt in gt_all:
        naive_pred_parts.append(torch.full_like(gt, fill_value=float(gt.mean().item())))
    naive_pred_pooled = torch.cat(naive_pred_parts)

    pooled_err = pred_pooled - gt_pooled
    naive_err = naive_pred_pooled - gt_pooled

    pooled_rmse = float(pooled_err.pow(2).mean().sqrt().item())
    pooled_mae = float(pooled_err.abs().mean().item())
    naive_rmse = float(naive_err.pow(2).mean().sqrt().item())

    aggregate = AggregateMetrics(
        n_subjects=len(per_subject),
        n_queries=int(pred_pooled.numel()),
        macro_rmse=float(sum(m.rmse for m in per_subject) / max(len(per_subject), 1)),
        macro_mae=float(sum(m.mae for m in per_subject) / max(len(per_subject), 1)),
        pooled_rmse=pooled_rmse,
        pooled_mae=pooled_mae,
        naive_pooled_rmse=naive_rmse,
        lift_over_naive=(naive_rmse - pooled_rmse) / max(naive_rmse, 1e-6),
        mean_pearson_r=float(
            sum((m.pearson_r if m.pearson_r == m.pearson_r else 0.0) for m in per_subject)
            / max(len(per_subject), 1)
        ),
    )

    return {
        "per_subject": per_subject,
        "aggregate": aggregate,
        "per_subject_predictions": predictions_dump,
    }


def metrics_to_dict(result: Dict[str, object]) -> Dict[str, object]:
    """Serialise the nested dataclass results to plain dicts for JSON dumping."""
    return {
        "per_subject": [asdict(m) for m in result["per_subject"]],
        "aggregate": asdict(result["aggregate"]),
    }


def format_summary(result: Dict[str, object]) -> str:
    """Human-readable summary string for CLI output."""
    agg = result["aggregate"]
    lines = [
        "=" * 60,
        "Zero-shot Theophylline PK evaluation",
        "=" * 60,
        f"  Subjects            : {agg.n_subjects}",
        f"  Queries             : {agg.n_queries}",
        f"  Macro-avg RMSE      : {agg.macro_rmse:.3f} mg/L",
        f"  Pooled RMSE         : {agg.pooled_rmse:.3f} mg/L",
        f"  Naive-baseline RMSE : {agg.naive_pooled_rmse:.3f} mg/L",
        f"  Lift over naive     : {agg.lift_over_naive:+.1%}",
        f"  Mean Pearson r      : {agg.mean_pearson_r:+.3f}",
        "",
        "Per subject:",
    ]
    for m in result["per_subject"]:
        lines.append(
            f"  subject {m.subject_id:>2d}: RMSE={m.rmse:5.2f}  "
            f"MAE={m.mae:5.2f}  r={m.pearson_r:+.2f}  "
            f"pred={m.mean_pred:5.2f}  gt={m.mean_gt:5.2f}"
        )
    return "\n".join(lines)
