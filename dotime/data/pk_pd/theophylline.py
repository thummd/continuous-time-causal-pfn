"""Theophylline pharmacokinetic dataset loader.

The ``datasets::Theoph`` dataset from R is a classic pharmacokinetic
benchmark: 12 subjects each receive a single oral dose of theophylline
and have their serum concentration measured at 11 irregularly-spaced
time points over ~24 hours.  Both the per-subject 1-compartment PK
model and the dataset itself are textbook.

Schema
------
Each row in the CSV has columns:

- ``Subject`` (1-12) : subject identifier.
- ``Wt`` (kg)        : body weight.
- ``Dose`` (mg/kg)   : oral dose, delivered at ``Time == 0``.
- ``Time`` (hr)      : time since dose.
- ``conc`` (mg/L)    : serum theophylline concentration.

Why we use it
-------------
As the primary zero-shot real-world benchmark for the ICML FMSD 2026
workshop paper: it is small, irregularly-sampled, has a genuinely
continuous underlying dynamics, and the compartmental model maps
naturally onto the Ornstein-Uhlenbeck SCM prior used for training
(concentration follows a linear ODE driven by the dose).

The CSV is shipped with the repo at ``data/pk_pd/theophylline.csv`` so
this loader works offline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


#: Location of the bundled CSV, relative to the repo root.
_DEFAULT_CSV_PATH = Path(__file__).resolve().parents[3] / "data" / "pk_pd" / "theophylline.csv"


@dataclass
class TheophSubject:
    """One subject's PK record.

    Attributes
    ----------
    subject_id : int
        Identifier from the original dataset (1 to 12).
    weight_kg : float
        Body weight in kilograms.
    dose_mg_per_kg : float
        Oral dose, applied as an instantaneous intervention at ``time ==
        0``.
    times : torch.Tensor
        Shape ``(T,)``; observation times in hours relative to dose.
        The first entry is typically ``0.0`` and the last is around 24.
    concentrations : torch.Tensor
        Shape ``(T,)``; serum concentrations in mg/L at the
        corresponding ``times``.
    """

    subject_id: int
    weight_kg: float
    dose_mg_per_kg: float
    times: torch.Tensor
    concentrations: torch.Tensor

    @property
    def n_observations(self) -> int:
        return int(self.times.numel())

    def __post_init__(self) -> None:
        if self.times.dim() != 1 or self.concentrations.dim() != 1:
            raise ValueError("times and concentrations must be 1-D")
        if self.times.numel() != self.concentrations.numel():
            raise ValueError(
                f"length mismatch: times={self.times.numel()} vs "
                f"concentrations={self.concentrations.numel()}"
            )
        if self.times.numel() < 2:
            raise ValueError("need at least 2 observation times")
        if not (self.times.diff() > 0).all():
            raise ValueError("observation times must be strictly increasing")


def load_theophylline(
    csv_path: Optional[Path] = None,
) -> List[TheophSubject]:
    """Load the dataset and split it by subject.

    Parameters
    ----------
    csv_path : Path, optional
        Override the bundled CSV location.  Useful for unit tests and
        for users who keep a custom copy of the dataset.

    Returns
    -------
    list of TheophSubject
        12 subjects, each with 11 time points (one subject has 10 or
        11 depending on CSV provenance; the loader accepts any count
        >= 2).
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Theophylline CSV not found at {path}. Fetch it once from "
            "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv"
        )

    rows_by_subject: Dict[int, List[dict]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = int(row["Subject"])
            rows_by_subject.setdefault(subj, []).append(row)

    subjects: List[TheophSubject] = []
    for subj_id in sorted(rows_by_subject.keys()):
        rows = sorted(rows_by_subject[subj_id], key=lambda r: float(r["Time"]))
        times = torch.tensor([float(r["Time"]) for r in rows], dtype=torch.float32)
        concs = torch.tensor([float(r["conc"]) for r in rows], dtype=torch.float32)
        subjects.append(
            TheophSubject(
                subject_id=subj_id,
                weight_kg=float(rows[0]["Wt"]),
                dose_mg_per_kg=float(rows[0]["Dose"]),
                times=times,
                concentrations=concs,
            )
        )
    return subjects
