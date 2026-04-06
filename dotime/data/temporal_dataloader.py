"""On-the-fly temporal intervention dataloader.

Generates batches by sampling from the extended CausalTimePrior,
following the pattern of Do-PFN's ObservationalDataLoader.

Supports background prefetching to overlap data generation with GPU compute.
"""

import torch
from threading import Thread
from queue import Queue
from typing import Dict, Iterator, Optional

from dotime.prior.extended_prior import ExtendedCausalTimePrior
from dotime.data.normalization import normalize_batch


class TemporalInterventionDataLoader:
    """Infinite dataloader that generates temporal intervention batches on-the-fly."""

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        n_max: int = 41,
        n_max_prior: int = 10,
        t_range: tuple = (50, 200),
        burn_in: int = 50,
        downstream_prob: float = 0.7,
        seed: int = 42,
        normalize: bool = True,
        device: str = "cpu",
        num_workers: int = 0,
        prefetch: int = 2,
        target_key: str = "Y_true",
        n_queries: int = 1,
        query_mode: str = "single",
        intervention_source: str = "prior",
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.target_key = target_key
        self.n_queries = n_queries
        self.query_mode = query_mode

        self.prior = ExtendedCausalTimePrior(
            n_max=n_max,
            n_max_prior=n_max_prior,
            t_range=t_range,
            burn_in=burn_in,
            downstream_prob=downstream_prob,
            seed=seed,
            intervention_source=intervention_source,
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
        """Generate batches with background prefetching."""
        queue: Queue = Queue(maxsize=self.prefetch)
        sentinel = object()

        def _fill():
            for _ in range(self.num_steps):
                batch = self._generate_batch()
                queue.put(batch)
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
        """Generate a single batch."""
        batch = self.prior.generate_batch(
            self.batch_size, num_workers=self.num_workers,
            n_queries=self.n_queries, query_mode=self.query_mode,
        )

        if self.normalize:
            batch = normalize_batch(batch, target_key=self.target_key)

        # Move to device
        if self.device != "cpu":
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        return batch
