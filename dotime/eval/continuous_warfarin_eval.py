"""Zero-shot Warfarin PK/PD evaluation for :class:`ContinuousDoOverTimePFN`.

Companion to :mod:`continuous_pk_eval` that targets the three-variable
Warfarin benchmark (dose -> concentration cp -> PCA).  Every subject
produces two streams of queries -- one for the PK mediator
(concentration, canonical variable 1) and one for the PD outcome
(PCA, canonical variable 2) -- and we report metrics separately for
each stream plus a combined aggregate.

Uses :func:`continuous_pk_eval._denormalise_prediction` and
:func:`continuous_pk_eval._pearson_r` for consistency with the
Theophylline evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

import torch

from dotime.data.pk_pd.warfarin import WarfarinSubject
from dotime.data.pk_pd.warfarin_adapter import build_warfarin_batch
from dotime.eval.continuous_pk_eval import _denormalise_prediction, _pearson_r
from dotime.model.continuous import ContinuousDoOverTimePFN


@dataclass
class WarfarinSubjectMetrics:
    """Per-subject Warfarin evaluation metrics split by query variable."""

    subject_id: int
    n_cp: int
    n_pca: int
    rmse_cp: float
    rmse_pca: float
    mae_cp: float
    mae_pca: float
    pearson_r_cp: float
    pearson_r_pca: float
    mean_pred_cp: float
    mean_gt_cp: float
    mean_pred_pca: float
    mean_gt_pca: float


@dataclass
class WarfarinAggregateMetrics:
    """Aggregate metrics over all evaluated Warfarin subjects."""

    n_subjects: int
    n_cp_queries: int
    n_pca_queries: int
    pooled_rmse_cp: float
    pooled_rmse_pca: float
    pooled_mae_cp: float
    pooled_mae_pca: float
    naive_pooled_rmse_cp: float
    naive_pooled_rmse_pca: float
    lift_over_naive_cp: float
    lift_over_naive_pca: float
    mean_pearson_r_cp: float
    mean_pearson_r_pca: float


def _pooled_rmse(err: torch.Tensor) -> float:
    if err.numel() == 0:
        return float("nan")
    return float(err.pow(2).mean().sqrt().item())


def _pooled_mae(err: torch.Tensor) -> float:
    if err.numel() == 0:
        return float("nan")
    return float(err.abs().mean().item())


def _safe_mean(xs: List[float]) -> float:
    valid = [x for x in xs if x == x]  # filter NaN
    return float(sum(valid) / len(valid)) if valid else float("nan")


@torch.no_grad()
def evaluate_warfarin_subject(
    model: ContinuousDoOverTimePFN,
    subject: WarfarinSubject,
    device: str = "cpu",
    **adapter_kwargs,
) -> Dict[str, object]:
    """Run the model on one subject and return split-by-variable metrics."""
    batch = build_warfarin_batch(subject, **adapter_kwargs)
    # Strip Python-list fields before moving to device; they stay attached
    # to the returned dict via the local ``var_names`` variable.
    var_names = batch.pop("_query_variable_name")
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    model.eval()
    output = model(batch)
    pred_norm = model.head.predict_mean(output)
    pred_raw = _denormalise_prediction(pred_norm, batch).cpu()
    gt_raw = batch["Y_true"].cpu()

    is_cp = torch.tensor([n == "cp" for n in var_names], dtype=torch.bool)
    is_pca = torch.tensor([n == "pca" for n in var_names], dtype=torch.bool)

    cp_err = pred_raw[is_cp] - gt_raw[is_cp]
    pca_err = pred_raw[is_pca] - gt_raw[is_pca]

    metrics = WarfarinSubjectMetrics(
        subject_id=subject.subject_id,
        n_cp=int(is_cp.sum().item()),
        n_pca=int(is_pca.sum().item()),
        rmse_cp=_pooled_rmse(cp_err),
        rmse_pca=_pooled_rmse(pca_err),
        mae_cp=_pooled_mae(cp_err),
        mae_pca=_pooled_mae(pca_err),
        pearson_r_cp=_pearson_r(pred_raw[is_cp], gt_raw[is_cp]),
        pearson_r_pca=_pearson_r(pred_raw[is_pca], gt_raw[is_pca]),
        mean_pred_cp=float(pred_raw[is_cp].mean().item()) if is_cp.any() else float("nan"),
        mean_gt_cp=float(gt_raw[is_cp].mean().item()) if is_cp.any() else float("nan"),
        mean_pred_pca=float(pred_raw[is_pca].mean().item()) if is_pca.any() else float("nan"),
        mean_gt_pca=float(gt_raw[is_pca].mean().item()) if is_pca.any() else float("nan"),
    )

    return {
        "metrics": metrics,
        "pred_cp": pred_raw[is_cp],
        "gt_cp": gt_raw[is_cp],
        "pred_pca": pred_raw[is_pca],
        "gt_pca": gt_raw[is_pca],
        "query_times_cp": batch["t_query"][is_cp].cpu(),
        "query_times_pca": batch["t_query"][is_pca].cpu(),
    }


@torch.no_grad()
def evaluate_warfarin_dataset(
    model: ContinuousDoOverTimePFN,
    subjects: Iterable[WarfarinSubject],
    device: str = "cpu",
    **adapter_kwargs,
) -> Dict[str, object]:
    """Evaluate on every subject and aggregate."""
    model.eval()
    per_subject: List[WarfarinSubjectMetrics] = []
    pred_cp_all: List[torch.Tensor] = []
    gt_cp_all: List[torch.Tensor] = []
    pred_pca_all: List[torch.Tensor] = []
    gt_pca_all: List[torch.Tensor] = []

    # Naive baseline: predict each subject's mean cp / pca over their own
    # observation window (mirrors the Theophylline naive baseline).
    naive_cp_all: List[torch.Tensor] = []
    naive_pca_all: List[torch.Tensor] = []

    for subj in subjects:
        r = evaluate_warfarin_subject(model, subj, device=device, **adapter_kwargs)
        per_subject.append(r["metrics"])
        pred_cp_all.append(r["pred_cp"])
        gt_cp_all.append(r["gt_cp"])
        pred_pca_all.append(r["pred_pca"])
        gt_pca_all.append(r["gt_pca"])

        if r["gt_cp"].numel() > 0:
            naive_cp_all.append(
                torch.full_like(r["gt_cp"], fill_value=float(r["gt_cp"].mean().item()))
            )
        if r["gt_pca"].numel() > 0:
            naive_pca_all.append(
                torch.full_like(r["gt_pca"], fill_value=float(r["gt_pca"].mean().item()))
            )

    if not per_subject:
        raise ValueError("no Warfarin subjects evaluated")

    def _cat_or_empty(parts: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts) if parts else torch.empty(0)

    pred_cp = _cat_or_empty(pred_cp_all)
    gt_cp = _cat_or_empty(gt_cp_all)
    pred_pca = _cat_or_empty(pred_pca_all)
    gt_pca = _cat_or_empty(gt_pca_all)
    naive_cp = _cat_or_empty(naive_cp_all)
    naive_pca = _cat_or_empty(naive_pca_all)

    pooled_rmse_cp = _pooled_rmse(pred_cp - gt_cp)
    pooled_rmse_pca = _pooled_rmse(pred_pca - gt_pca)
    naive_rmse_cp = _pooled_rmse(naive_cp - gt_cp)
    naive_rmse_pca = _pooled_rmse(naive_pca - gt_pca)

    def _lift(model_rmse: float, naive_rmse: float) -> float:
        if naive_rmse != naive_rmse or naive_rmse < 1e-6:
            return float("nan")
        return (naive_rmse - model_rmse) / naive_rmse

    aggregate = WarfarinAggregateMetrics(
        n_subjects=len(per_subject),
        n_cp_queries=int(pred_cp.numel()),
        n_pca_queries=int(pred_pca.numel()),
        pooled_rmse_cp=pooled_rmse_cp,
        pooled_rmse_pca=pooled_rmse_pca,
        pooled_mae_cp=_pooled_mae(pred_cp - gt_cp),
        pooled_mae_pca=_pooled_mae(pred_pca - gt_pca),
        naive_pooled_rmse_cp=naive_rmse_cp,
        naive_pooled_rmse_pca=naive_rmse_pca,
        lift_over_naive_cp=_lift(pooled_rmse_cp, naive_rmse_cp),
        lift_over_naive_pca=_lift(pooled_rmse_pca, naive_rmse_pca),
        mean_pearson_r_cp=_safe_mean([m.pearson_r_cp for m in per_subject]),
        mean_pearson_r_pca=_safe_mean([m.pearson_r_pca for m in per_subject]),
    )

    return {"per_subject": per_subject, "aggregate": aggregate}


def metrics_to_dict(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "per_subject": [asdict(m) for m in result["per_subject"]],
        "aggregate": asdict(result["aggregate"]),
    }


def format_summary(result: Dict[str, object]) -> str:
    agg = result["aggregate"]
    lines = [
        "=" * 62,
        "Zero-shot Warfarin PK/PD evaluation",
        "=" * 62,
        f"  Subjects                  : {agg.n_subjects}",
        f"  Concentration queries (cp): {agg.n_cp_queries}",
        f"  PCA queries                : {agg.n_pca_queries}",
        "",
        "  Concentration (PK mediator, mg/L)",
        f"    Pooled RMSE             : {agg.pooled_rmse_cp:.3f}",
        f"    Naive-baseline RMSE     : {agg.naive_pooled_rmse_cp:.3f}",
        f"    Lift over naive         : {agg.lift_over_naive_cp:+.1%}",
        f"    Mean Pearson r          : {agg.mean_pearson_r_cp:+.3f}",
        "",
        "  PCA (PD outcome, % activity)",
        f"    Pooled RMSE             : {agg.pooled_rmse_pca:.3f}",
        f"    Naive-baseline RMSE     : {agg.naive_pooled_rmse_pca:.3f}",
        f"    Lift over naive         : {agg.lift_over_naive_pca:+.1%}",
        f"    Mean Pearson r          : {agg.mean_pearson_r_pca:+.3f}",
    ]
    return "\n".join(lines)
