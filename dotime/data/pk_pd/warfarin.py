"""Warfarin pharmacokinetic / pharmacodynamic dataset loader.

The ``warfarin`` dataset from the ``nlmixr2data`` R package is a
three-variable PK/PD benchmark: an oral warfarin dose produces a
concentration trajectory (``cp``) which in turn drives a
pharmacodynamic response (``pca`` = prothrombin complex activity,
essentially an anticoagulation effect marker).

Structure (per subject)
-----------------------
- Single oral dose at ``time = 0`` with ``amt`` mg (varies by
  subject; ``evid == 1`` marks the dosing row).
- Concentration observations (``dvid == "cp"``) at irregular times
  through ~120 h.
- PCA observations (``dvid == "pca"``) at irregular times, often at
  different time points than the concentration observations.
- Covariates: weight (kg), age (years), sex.

Why we use it
-------------
As a three-variable real-world PK/PD benchmark for the ICML FMSD 2026
workshop paper. Theophylline exercises the continuous-time model on
the two-variable ``dose -> concentration`` setup; Warfarin adds the
pharmacodynamic mediator step ``concentration -> PCA`` and therefore
exercises the model's front-door-style reasoning on real data.

Schema
------
CSV columns: ``id, time, amt, dv, dvid, evid, wt, age, sex``.

- ``id`` (int) : subject identifier.
- ``time`` (float, h) : observation time since dose.
- ``amt`` (float, mg) : dose amount (only on ``evid == 1`` rows).
- ``dv`` (float) : dependent variable value (concentration or PCA).
- ``dvid`` ({"cp", "pca"}) : which dependent variable the row measures.
- ``evid`` ({0, 1}) : 0 = observation, 1 = dose.
- ``wt``, ``age``, ``sex`` : covariates (not consumed by the adapter).

The bundled CSV at ``data/pk_pd/warfarin.csv`` works offline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch


_DEFAULT_CSV_PATH = Path(__file__).resolve().parents[3] / "data" / "pk_pd" / "warfarin.csv"


def _extract_obs(rows: List[dict], dvid: str) -> tuple:
    """Return (times, values) for all observations of one dvid, averaging duplicates.

    A handful of Warfarin subjects have replicate measurements at the
    same timestamp.  We collapse them with an arithmetic mean so the
    downstream adapter can use a strictly-increasing time axis.
    """
    pairs: Dict[float, List[float]] = {}
    for r in rows:
        if int(r["evid"]) == 0 and r["dvid"] == dvid:
            pairs.setdefault(float(r["time"]), []).append(float(r["dv"]))
    if not pairs:
        return (torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32))
    sorted_times = sorted(pairs.keys())
    values = [sum(pairs[t]) / len(pairs[t]) for t in sorted_times]
    return (
        torch.tensor(sorted_times, dtype=torch.float32),
        torch.tensor(values, dtype=torch.float32),
    )


@dataclass
class WarfarinSubject:
    """One subject's PK + PD record.

    Attributes
    ----------
    subject_id : int
    dose_mg : float
        Total dose in milligrams, administered at ``time == 0``.
    weight_kg : float
    age_years : int
    sex : str
        ``"male"`` or ``"female"`` in the bundled dataset.
    cp_times : torch.Tensor
        Shape ``(n_cp,)`` observation times (hours) for concentration.
    cp_values : torch.Tensor
        Shape ``(n_cp,)`` concentration values (mg/L).
    pca_times : torch.Tensor
        Shape ``(n_pca,)`` observation times for PCA.
    pca_values : torch.Tensor
        Shape ``(n_pca,)`` PCA values (% of baseline activity).
    """

    subject_id: int
    dose_mg: float
    weight_kg: float
    age_years: int
    sex: str
    cp_times: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    cp_values: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    pca_times: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    pca_values: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self) -> None:
        for name, t in (("cp_times", self.cp_times),
                        ("cp_values", self.cp_values),
                        ("pca_times", self.pca_times),
                        ("pca_values", self.pca_values)):
            if t.dim() != 1:
                raise ValueError(f"{name} must be 1-D, got shape {tuple(t.shape)}")
        if self.cp_times.numel() != self.cp_values.numel():
            raise ValueError(
                f"cp length mismatch: {self.cp_times.numel()} times vs "
                f"{self.cp_values.numel()} values"
            )
        if self.pca_times.numel() != self.pca_values.numel():
            raise ValueError(
                f"pca length mismatch: {self.pca_times.numel()} times vs "
                f"{self.pca_values.numel()} values"
            )
        if self.cp_times.numel() > 1 and not (self.cp_times.diff() > 0).all():
            raise ValueError("cp_times must be strictly increasing")
        if self.pca_times.numel() > 1 and not (self.pca_times.diff() > 0).all():
            raise ValueError("pca_times must be strictly increasing")

    @property
    def n_cp(self) -> int:
        return int(self.cp_times.numel())

    @property
    def n_pca(self) -> int:
        return int(self.pca_times.numel())

    def union_times(self) -> torch.Tensor:
        """Sorted union of ``cp_times`` and ``pca_times`` plus ``0.0``.

        ``0.0`` is always included so the adapter can anchor the
        intervention window at the dosing time.
        """
        all_times = torch.cat(
            [torch.tensor([0.0], dtype=torch.float32), self.cp_times, self.pca_times]
        )
        return torch.unique(all_times)  # torch.unique returns sorted


def load_warfarin(
    csv_path: Optional[Path] = None,
) -> List[WarfarinSubject]:
    """Load the Warfarin CSV and split it by subject.

    Parameters
    ----------
    csv_path : Path, optional
        Override the bundled CSV location.

    Returns
    -------
    list of WarfarinSubject
        Ordered by ``subject_id``.
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Warfarin CSV not found at {path}. Fetch it once via the "
            "nlmixr2data R package (data/warfarin.rda in the repo) or "
            "from github.com/nlmixr2/nlmixr2data."
        )

    rows_by_subject: Dict[int, List[dict]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_subject.setdefault(int(row["id"]), []).append(row)

    subjects: List[WarfarinSubject] = []
    for subj_id in sorted(rows_by_subject.keys()):
        rows = rows_by_subject[subj_id]
        dose_rows = [r for r in rows if int(r["evid"]) == 1]
        if not dose_rows:
            continue
        dose_row = dose_rows[0]
        dose_mg = float(dose_row["amt"])
        wt = float(dose_row["wt"])
        age = int(dose_row["age"])
        sex = str(dose_row["sex"])

        cp_times, cp_values = _extract_obs(rows, dvid="cp")
        pca_times, pca_values = _extract_obs(rows, dvid="pca")

        subjects.append(
            WarfarinSubject(
                subject_id=subj_id,
                dose_mg=dose_mg,
                weight_kg=wt,
                age_years=age,
                sex=sex,
                cp_times=cp_times,
                cp_values=cp_values,
                pca_times=pca_times,
                pca_values=pca_values,
            )
        )
    return subjects
