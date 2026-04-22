"""Continuous-time regime-switching SCM.

The discrete-time :class:`CausalTimePrior` mixes in a 15% slice of
regime-switching SCMs: multiple structural regimes alternate according
to a sticky Markov chain so the generated trajectories contain
structural breaks.  Phase 8 ports that idea to continuous time on top
of :class:`ContinuousSCM`.

Design
------
A :class:`ContinuousRegimeSwitchingSCM` holds ``R`` independent
:class:`ContinuousSCM` regimes that share the same variable count
(so the observation tensor layout is consistent across regimes) plus
a row-stochastic :math:`R \\times R` transition matrix with
``P[i, i] ~ 0.9`` for "sticky" behaviour (expected regime duration
~10 observation steps).

Simulation uses a pre-sampled regime trajectory and a single shared
standard-normal noise table (shape ``(T - 1, n_vars)``).  At step
``i -> i+1`` the active regime ``r = regime_trajectory[i]`` decides
which regime's mechanisms drive the step; the noise table is
regime-agnostic and each regime's ``sigma_v`` multiplies it inside
:meth:`OUMechanism.step`.  Sharing the noise and regime trajectory
between obs and interventional runs keeps
:meth:`sample_counterfactual_pair`'s "identical before intervention"
property intact even across regime switches.

Intervention handling is unchanged from :class:`ContinuousSCM`: the
active regime's mechanisms produce the drift, then the intervention
spec (hard / soft / time-varying) modifies the variable as usual.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .continuous_scm import (
    ContinuousIntervention,
    ContinuousSCM,
)


def sample_sticky_transition_matrix(
    n_regimes: int,
    sticky_alpha: float = 9.0,
    other_alpha: float = 0.5,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw a sticky Markov transition matrix from a Dirichlet prior.

    Each row of the matrix is sampled as a Dirichlet with concentration
    ``sticky_alpha`` on the diagonal entry and ``other_alpha`` on the
    off-diagonal entries.  With the defaults ``(9.0, 0.5)`` the
    diagonal is typically around 0.9 -- i.e. an expected regime
    duration of 10 observation steps.

    Parameters
    ----------
    n_regimes : int
        Number of regimes.  Must be at least 2.
    sticky_alpha, other_alpha : float
        Dirichlet concentration on the diagonal / off-diagonal entries
        respectively.  Both must be positive.
    generator : torch.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    torch.Tensor
        Shape ``(n_regimes, n_regimes)``, row-stochastic.
    """
    if n_regimes < 2:
        raise ValueError(f"need >= 2 regimes, got {n_regimes}")
    if sticky_alpha <= 0 or other_alpha <= 0:
        raise ValueError(
            f"Dirichlet concentrations must be positive, got "
            f"sticky_alpha={sticky_alpha}, other_alpha={other_alpha}"
        )

    # Implementation note: :class:`torch.distributions.Dirichlet` is the
    # obvious choice, but its ``sample()`` method does not accept a
    # :class:`torch.Generator`.  The transition matrix is only sampled
    # once per regime-switching SCM, so we fall back to the global RNG
    # for that call; callers that need bit-exact reproducibility should
    # wrap their code in ``torch.manual_seed(...)`` and any per-call
    # generator only gates the rest of the pipeline.
    rows = []
    for r in range(n_regimes):
        alphas = torch.full((n_regimes,), float(other_alpha), device=device, dtype=dtype)
        alphas[r] = float(sticky_alpha)
        if generator is not None:
            # Seeded path: sample Gamma rows via the Wilson-Hilferty
            # transform of a seeded Gaussian.  Works well for alpha >= 1;
            # for small alpha (the default 0.5 off-diagonal) it is a
            # biased approximation but still produces valid sticky rows.
            z = torch.empty(n_regimes, device=device, dtype=dtype)
            z.normal_(generator=generator)
            g = alphas * (1.0 - 1.0 / (9.0 * alphas)
                          + z / torch.sqrt(9.0 * alphas)) ** 3
            g = g.clamp(min=1e-6)
        else:
            g = torch.distributions.Dirichlet(alphas).sample()
            g = g.clamp(min=1e-6)
        rows.append(g / g.sum())
    return torch.stack(rows, dim=0)


class ContinuousRegimeSwitchingSCM:
    """Continuous-time SCM that alternates between ``R`` regimes.

    Parameters
    ----------
    regimes : sequence of :class:`ContinuousSCM`
        Each regime is a complete SCM with its own OU mechanisms.  All
        regimes must agree on ``n_vars``, ``device``, and ``dtype``.
    transition_matrix : torch.Tensor
        Row-stochastic ``(R, R)`` matrix; ``P[i, j]`` is the
        probability of transitioning from regime ``i`` to regime ``j``
        at each observation step.
    initial_distribution : torch.Tensor, optional
        Shape ``(R,)`` probability vector over the first regime.
        Defaults to uniform.
    """

    def __init__(
        self,
        regimes: List[ContinuousSCM],
        transition_matrix: torch.Tensor,
        initial_distribution: Optional[torch.Tensor] = None,
    ) -> None:
        if len(regimes) < 2:
            raise ValueError(f"need >= 2 regimes, got {len(regimes)}")
        n_vars = regimes[0].n_vars
        for i, scm in enumerate(regimes):
            if scm.n_vars != n_vars:
                raise ValueError(
                    f"all regimes must share n_vars; regime 0 has {n_vars}, "
                    f"regime {i} has {scm.n_vars}"
                )
        if transition_matrix.shape != (len(regimes), len(regimes)):
            raise ValueError(
                f"transition_matrix must have shape ({len(regimes)}, {len(regimes)}), "
                f"got {tuple(transition_matrix.shape)}"
            )
        row_sums = transition_matrix.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            raise ValueError(
                f"transition_matrix rows must sum to 1, got row_sums={row_sums.tolist()}"
            )
        if (transition_matrix < 0).any():
            raise ValueError("transition_matrix must be nonnegative")

        self.regimes = list(regimes)
        self.transition_matrix = transition_matrix
        if initial_distribution is None:
            initial_distribution = torch.full(
                (len(regimes),), 1.0 / len(regimes),
                dtype=transition_matrix.dtype,
            )
        if initial_distribution.shape != (len(regimes),):
            raise ValueError(
                f"initial_distribution must have shape ({len(regimes)},), "
                f"got {tuple(initial_distribution.shape)}"
            )
        self.initial_distribution = initial_distribution

        # Convenience views
        self.n_vars = n_vars
        self.n_regimes = len(regimes)
        self.device = regimes[0].device
        self.dtype = regimes[0].dtype

    # ------------------------------------------------------------------ sampling

    @classmethod
    def sample_random(
        cls,
        n_vars: int,
        n_regimes: int = 2,
        edge_prob: float = 0.3,
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.6),
        weight_scale: float = 0.5,
        sticky_alpha: float = 9.0,
        other_alpha: float = 0.5,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ContinuousRegimeSwitchingSCM":
        """Sample ``n_regimes`` independent random-graph regimes and a sticky
        transition matrix."""
        regimes = [
            ContinuousSCM.sample_random(
                n_vars=n_vars,
                edge_prob=edge_prob,
                theta_range=theta_range,
                sigma_range=sigma_range,
                weight_scale=weight_scale,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            for _ in range(n_regimes)
        ]
        P = sample_sticky_transition_matrix(
            n_regimes=n_regimes,
            sticky_alpha=sticky_alpha,
            other_alpha=other_alpha,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return cls(regimes, P)

    # ------------------------------------------------------------------ helpers

    def _draw_regime_trajectory(
        self,
        T: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample ``T`` regime indices via the sticky Markov chain."""
        regime_traj = torch.empty(T, dtype=torch.long, device=self.device)
        u = torch.empty(T, device=self.device, dtype=self.dtype)
        u.uniform_(0.0, 1.0, generator=generator)

        init_cdf = torch.cumsum(self.initial_distribution, dim=0)
        regime_traj[0] = int((u[0] < init_cdf).long().argmax().item())

        cdfs = torch.cumsum(self.transition_matrix, dim=1)  # (R, R)
        for t in range(1, T):
            r_prev = int(regime_traj[t - 1].item())
            regime_traj[t] = int((u[t] < cdfs[r_prev]).long().argmax().item())
        return regime_traj

    def _draw_noise(
        self,
        num_steps: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Standard-normal noise table ``(num_steps, n_vars)``, shared across regimes."""
        noise = torch.empty(num_steps, self.n_vars, device=self.device, dtype=self.dtype)
        noise.normal_(mean=0.0, std=1.0, generator=generator)
        return noise

    # ------------------------------------------------------------------ simulate

    def simulate(
        self,
        times: torch.Tensor,
        dts: torch.Tensor,
        intervention: Optional[ContinuousIntervention] = None,
        x0: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        regime_trajectory: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the regime-switching SCM forward on the given schedule.

        Returns ``(times, trajectory)`` to match the
        :meth:`ContinuousSCM.simulate` signature so this class is a
        drop-in replacement for :class:`ContinuousSCM` in the extended
        prior's dispatch.  The last regime trajectory used is stashed
        on ``self.last_regime_trajectory`` for callers that want to
        inspect it.

        Passing in pre-sampled ``noise`` and ``regime_trajectory`` lets
        paired simulations share both, which is how counterfactual
        pairs stay identical before the intervention window even in the
        presence of regime switches.
        """
        if times.dim() != 1 or dts.dim() != 1 or dts.numel() != times.numel() - 1:
            raise ValueError(
                f"shape mismatch: times={tuple(times.shape)}, dts={tuple(dts.shape)}"
            )
        T = times.numel()
        if noise is None:
            noise = self._draw_noise(T - 1, generator=generator)
        elif noise.shape != (T - 1, self.n_vars):
            raise ValueError(
                f"noise has shape {tuple(noise.shape)}, expected ({T - 1}, {self.n_vars})"
            )
        if regime_trajectory is None:
            regime_trajectory = self._draw_regime_trajectory(T, generator=generator)
        elif regime_trajectory.shape != (T,):
            raise ValueError(
                f"regime_trajectory has shape {tuple(regime_trajectory.shape)}, "
                f"expected ({T},)"
            )

        if x0 is None:
            x = torch.zeros(self.n_vars, device=self.device, dtype=self.dtype)
        else:
            x = x0.to(device=self.device, dtype=self.dtype).clone()

        trajectory = torch.empty(T, self.n_vars, device=self.device, dtype=self.dtype)
        trajectory[0] = x
        for i in range(T - 1):
            # Active regime governing the step i -> i+1 is the regime at
            # time i (Euler-Maruyama is explicit, so start-of-step drift
            # / diffusion parameters are what we use).
            r = int(regime_trajectory[i].item())
            x = self.regimes[r]._step(
                x,
                dts[i],
                noise[i],
                intervention=intervention,
                t_next=float(times[i + 1].item()),
            )
            trajectory[i + 1] = x

        self.last_regime_trajectory = regime_trajectory
        return times, trajectory

    def sample_counterfactual_pair(
        self,
        times: torch.Tensor,
        dts: torch.Tensor,
        intervention: ContinuousIntervention,
        x0: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return matched ``(times, X_obs, X_cf)`` with shared noise + regimes.

        Matches :meth:`ContinuousSCM.sample_counterfactual_pair` for
        drop-in compatibility with the extended prior's dispatch.  Both
        the Brownian-noise table and the regime trajectory are drawn
        once and reused for the obs / cf runs, keeping the
        counterfactual-semantics guarantee (identical trajectories
        before ``intervention.t_start``) intact across regime switches.
        """
        T = times.numel()
        noise = self._draw_noise(T - 1, generator=generator)
        regime_trajectory = self._draw_regime_trajectory(T, generator=generator)
        _, x_obs = self.simulate(
            times, dts, intervention=None,
            x0=x0, noise=noise, regime_trajectory=regime_trajectory,
        )
        _, x_cf = self.simulate(
            times, dts, intervention=intervention,
            x0=x0, noise=noise, regime_trajectory=regime_trajectory,
        )
        return times, x_obs, x_cf

    def sample_interventional_pair(
        self,
        times: torch.Tensor,
        dts: torch.Tensor,
        intervention: ContinuousIntervention,
        x0: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return matched ``(times, X_obs, X_int)`` with independent noise /
        regime draws for obs vs int runs."""
        _, x_obs = self.simulate(
            times, dts, intervention=None, x0=x0, generator=generator,
        )
        _, x_int = self.simulate(
            times, dts, intervention=intervention, x0=x0, generator=generator,
        )
        return times, x_obs, x_int
