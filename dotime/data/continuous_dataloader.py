"""Continuous-time on-the-fly temporal intervention dataloader.

Analogue of :class:`dotime.data.temporal_dataloader.TemporalInterventionDataLoader`
that pulls samples from :class:`ContinuousExtendedPrior` instead of
``ExtendedCausalTimePrior``.

Batch contract
--------------
Every field returned by the discrete loader is still present with the
same semantics, plus two new tensors:

- ``times``: ``(B, T)`` float tensor of absolute observation times.
- ``dts``: ``(B, T - 1)`` float tensor of inter-observation gaps.

``t_int_start``, ``t_int_end``, ``t_query`` are in the same absolute
units as ``times``, so downstream encoders can compute ``times -
t_int_start`` directly.

Normalisation
-------------
The usual :func:`dotime.data.normalization.normalize_batch` is applied
so that ``X_obs_norm`` / ``Y_true_norm`` are available alongside the
raw ``X_obs`` / ``Y_true`` tensors.  It does not touch ``times`` or
``dts`` (they don't share the per-variable scale of the trajectory
tensors), so no extra changes are needed there.
"""

from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import Dict, Iterator, Optional

import torch

from dotime.data.normalization import normalize_batch
from dotime.prior.continuous.extended_prior import ContinuousExtendedPrior


class ContinuousTemporalInterventionDataLoader:
    """Infinite dataloader for continuous-time causal PFN training.

    Parameters
    ----------
    num_steps : int
        Number of batches per iteration over the loader.
    batch_size : int
        Batch size.
    tscm_structure : str
        Named :class:`TSCMStructure` value (``back_door``, ``front_door``, ...).
    schedule : {"regular", "jittered", "exponential"}
        Observation schedule family; see :mod:`time_schedule`.
    pair_mode : {"counterfactual", "interventional"}
        Paired-sample semantics.  The workshop paper defaults to
        ``counterfactual`` (shared noise).
    t_range : tuple of int
        Uniform prior on the number of observations per trajectory.
    dt, jitter, exp_rate : forwarded to :class:`ContinuousExtendedPrior`.
    n_max : int
        Variable-axis padding (should match the model's ``n_max``).
    seed : int
        Base seed for the prior's RNG.
    normalize : bool
        Apply per-variable z-score normalisation to ``X_obs`` and
        ``Y_true`` (producing ``X_obs_norm`` / ``Y_true_norm``).
    device : str
        Device to move the returned batch tensors to.
    prefetch : int
        Background-prefetch queue depth.  ``0`` disables prefetch.
    target_key : str
        Which field of the raw batch to use as the regression target
        (typically ``"Y_true"`` or ``"Y_causal_effect"``).
    n_queries : int
        Number of (variable, time) query points per trajectory.
    query_mode : {"single", "all_pairs"}
        See :meth:`ContinuousExtendedPrior.generate_sample`.
    theta_range, sigma_range, weight_scale, intervention_value_scale, intervention_window_frac :
        Forwarded to the prior.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        tscm_structure: str = "back_door",
        schedule: str = "regular",
        pair_mode: str = "counterfactual",
        t_range: tuple = (50, 200),
        dt: float = 1.0,
        jitter: float = 0.3,
        exp_rate: float = 1.0,
        n_max: int = 41,
        seed: int = 42,
        normalize: bool = True,
        device: str = "cpu",
        prefetch: int = 0,
        target_key: str = "Y_true",
        n_queries: int = 1,
        query_mode: str = "single",
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.6),
        weight_scale: float = 0.5,
        intervention_value_scale: float = 2.0,
        intervention_window_frac: tuple = (0.1, 0.3),
    ) -> None:
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.prefetch = prefetch
        self.target_key = target_key
        self.n_queries = n_queries
        self.query_mode = query_mode

        self.prior = ContinuousExtendedPrior(
            tscm_structure=tscm_structure,
            n_max=n_max,
            t_range=t_range,
            schedule=schedule,
            dt=dt,
            jitter=jitter,
            exp_rate=exp_rate,
            pair_mode=pair_mode,
            intervention_value_scale=intervention_value_scale,
            intervention_window_frac=intervention_window_frac,
            theta_range=theta_range,
            sigma_range=sigma_range,
            weight_scale=weight_scale,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.prefetch > 0:
            yield from self._iter_prefetch()
        else:
            for _ in range(self.num_steps):
                yield self._generate_batch()

    def _iter_prefetch(self) -> Iterator[Dict[str, torch.Tensor]]:
        queue: Queue = Queue(maxsize=self.prefetch)
        sentinel = object()

        def _fill():
            for _ in range(self.num_steps):
                queue.put(self._generate_batch())
            queue.put(sentinel)

        thread = Thread(target=_fill, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item
        thread.join(timeout=5)

    def _generate_batch(self) -> Dict[str, torch.Tensor]:
        batch = self.prior.generate_batch(
            batch_size=self.batch_size,
            n_queries=self.n_queries,
            query_mode=self.query_mode,
        )
        if self.normalize:
            batch = normalize_batch(batch, target_key=self.target_key)
        if self.device != "cpu":
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        return batch
