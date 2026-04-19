"""Baselines for prior-based identifiability evaluation.

All baselines implement SinglePointTimeSeriesBaseline, which takes a normalized
batch (as produced by prior_eval.py) and returns (pred: float, output).

Batch keys
----------
X_obs_norm          (1, T, n_max)  normalized observational trajectory
variable_mask       (1, n_max)     1 = observed, 0 = hidden
intervention_target (1,)           index of the intervened variable (A)
intervention_type   (1,)
intervention_value  (1,)           normalized intervention value
intervention_time_start (1,)       fractional time of intervention
intervention_time_end   (1,)
query_target        (1,)           index of the outcome variable (Y)
query_time          (1,)           fractional time of query
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import pandas as pd
import torch

from dotime.model.do_over_time_pfn import DoOverTimePFN
from pfns.model.bar_distribution import FullSupportBarDistribution
from tabpfn import TabPFNRegressor


class SinglePointTimeSeriesBaseline(ABC):

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]):
        """Return (pred: float, output) for a single query."""

    @property
    def checkpoint_path(self) -> str:
        return ""


class TrainedBaseline(SinglePointTimeSeriesBaseline):
    """Loads a DoOverTimePFN checkpoint and runs inference."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self.ckpt = ckpt
        self.model = DoOverTimePFN(**ckpt['config'])
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if ckpt.get('head_type', 'bar') == 'bar' and ckpt.get('borders') is not None:
            bar_dist = FullSupportBarDistribution(ckpt['borders'])
            self.model.bar_head.set_bar_distribution(bar_dist, ckpt['borders'])
        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, batch: Dict[str, torch.Tensor]):
        with torch.no_grad():
            output = self.model(batch)
            pred = self.model.head.predict_mean(output)
        return pred.cpu().item(), output.cpu()


class BackDoorObsPFNCausalEffect(TrainedBaseline):

    @property
    def checkpoint_path(self) -> str:
        return "/work/dlclarge1/robertsj-dotpfn/do-over-time-pfn/checkpoints/s4_bd_/s4_bd_nolag_full_obs/do_over_time_pfn_best.pt"


class BackDoorDoTPFNCausalEffect(TrainedBaseline):

    @property
    def checkpoint_path(self) -> str:
        return "/work/dlclarge1/robertsj-dotpfn/do-over-time-pfn/checkpoints/s4_bd_/s4_bd_nolag_full_causal/do_over_time_pfn_best.pt"


class AR1Baseline(SinglePointTimeSeriesBaseline):
    """Returns the last observed value of the query variable (AR(1) proxy)."""

    def forward(self, batch: Dict[str, torch.Tensor]):
        return batch['X_obs_norm'][:, -1, batch['query_target']].cpu().item(), None


class ZeroBaseline(SinglePointTimeSeriesBaseline):
    """Always predicts zero (normalized mean)."""

    def forward(self, batch: Dict[str, torch.Tensor]):
        return 0, None


class Chronos2Observational(SinglePointTimeSeriesBaseline):
    """Chronos-2 forecast treating the intervention as a known future covariate."""

    def __init__(self, device: str = "cpu"):
        from chronos import BaseChronosPipeline
        self.device = device
        self.pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)

    def forward(self, batch: Dict[str, torch.Tensor]):
        context_df = pd.DataFrame(batch["X_obs_norm"].squeeze(0).cpu().numpy())

        future_df = context_df.iloc[[-1]].copy()
        future_df.iloc[:, batch["variable_mask"].squeeze(0).cpu().numpy() == 1] = np.nan
        future_df.iloc[:, batch["intervention_target"].cpu().numpy().item()] = (
            batch["intervention_value"].cpu().numpy().item()
        )

        context_df["item_id"] = 0
        future_df["item_id"] = 0
        context_df["timestamp"] = np.arange(len(context_df))
        future_df["timestamp"] = len(context_df)
        future_df = future_df.drop(columns=[batch["query_target"].cpu().numpy().item()])

        pred_df = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=1,
            quantile_levels=[0.1, 0.5, 0.9],
            target=batch["query_target"].cpu().numpy().item(),
        )

        return pred_df["predictions"], torch.tensor([pred_df["0.1"], pred_df["0.5"], pred_df["0.9"]])


# ---------------------------------------------------------------------------
# Back-door adjustment (TabPFN)
# ---------------------------------------------------------------------------

class _BackDoorTabPFNBase(SinglePointTimeSeriesBaseline):
    """Back-door adjustment using two TabPFN regressors fit in-context.

    Models fit on the observational trajectory up to the intervention time:
      model_x: p(X_t | X_{t-1})
      model_y: p(Y_t | A_t, X_t, Y_{t-1})

    Assumes exactly one observed confounder X (neither A nor Y). offset=0 only.
    """

    def _setup(self, batch: Dict[str, torch.Tensor]):
        """Return (model_y, x_t_samples, y_prev, int_value_norm, a_obs)."""
        X_obs_norm = batch['X_obs_norm'][0]       # (T, n_max)
        variable_mask = batch['variable_mask'][0]  # (n_max,)
        a_idx = int(batch['intervention_target'].item())
        y_idx = int(batch['query_target'].item())
        int_value_norm = float(batch['intervention_value'].item())

        T = X_obs_norm.shape[0]
        t = int(round(batch['intervention_time_start'].item() * T))

        observed = [i for i in range(variable_mask.shape[0]) if variable_mask[i] > 0.5]
        x_vars = [i for i in observed if i != a_idx and i != y_idx]
        if len(x_vars) != 1:
            raise ValueError(
                f"BackDoor adjustment requires exactly one confounder X, found {len(x_vars)}: {x_vars}"
            )
        if t < 2:
            raise ValueError(f"BackDoor adjustment requires t >= 2, got t={t}")

        x_idx = x_vars[0]
        x_series = X_obs_norm[:t, x_idx].cpu().numpy()
        a_series = X_obs_norm[:t, a_idx].cpu().numpy()
        y_series = X_obs_norm[:t, y_idx].cpu().numpy()

        model_x = TabPFNRegressor()
        model_x.fit(x_series[:-1].reshape(-1, 1), x_series[1:])

        model_y = TabPFNRegressor()
        model_y.fit(
            np.column_stack([a_series[1:], x_series[1:], y_series[:-1]]),
            y_series[1:],
        )

        # Approximate E[Y|a,X_t,y_prev] by integrating over p(X_t|x_{t-1}) via quantiles
        q_levels = np.linspace(1 / (self.n_mc + 1), 1 - 1 / (self.n_mc + 1), self.n_mc).tolist()
        x_t_samples = np.array([
            float(q[0]) for q in model_x.predict(
                x_series[-1:].reshape(1, 1),
                output_type="quantiles",
                quantiles=q_levels,
            )
        ])

        return model_y, x_t_samples, float(y_series[-1]), int_value_norm, float(a_series[-1])

    def _mc_predict(self, model_y, a_val: float, x_t_samples, y_prev: float) -> float:
        """E[Y_t | a_val, X_t, y_prev] averaged over x_t_samples."""
        X_q = np.column_stack([
            np.full(self.n_mc, a_val),
            x_t_samples,
            np.full(self.n_mc, y_prev),
        ])
        return float(np.mean(model_y.predict(X_q, output_type="mean")))


class BackDoorTabPFNInterventional(_BackDoorTabPFNBase):
    """E[Y_t | do(A_t = a), D_{<t}] via back-door adjustment."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]):
        model_y, x_t_samples, y_prev, int_value_norm, _ = self._setup(batch)
        return self._mc_predict(model_y, int_value_norm, x_t_samples, y_prev), None


class BackDoorTabPFNObservational(_BackDoorTabPFNBase):
    """E[Y_t | A_t = a_obs, D_{<t}] via back-door adjustment (observational A)."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]):
        model_y, x_t_samples, y_prev, _, a_obs = self._setup(batch)
        return self._mc_predict(model_y, a_obs, x_t_samples, y_prev), None


class BackDoorTabPFNCausalEffect(_BackDoorTabPFNBase):
    """E[Y|do(a)] - E[Y|a_obs] via back-door adjustment."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]):
        model_y, x_t_samples, y_prev, int_value_norm, a_obs = self._setup(batch)
        pred_int = self._mc_predict(model_y, int_value_norm, x_t_samples, y_prev)
        pred_obs = self._mc_predict(model_y, a_obs, x_t_samples, y_prev)
        return pred_int - pred_obs, None


# ---------------------------------------------------------------------------
# Front-door adjustment (TabPFN)
# ---------------------------------------------------------------------------

class _FrontDoorTabPFNBase(SinglePointTimeSeriesBaseline):
    """Front-door adjustment using three TabPFN regressors fit in-context.

    Models fit on the observational trajectory up to the intervention time:
      model_m: P(M_t | A_t, M_{t-1})
      model_y: P(Y_t | M_t, A_t, Y_{t-1})
      model_a: P(A_t | A_{t-1})   — natural A marginal for the inner integral

    Front-door formula:
      E[Y_t | do(A_t=a), D_{<t}]
        = sum_m P(M_t=m | A_t=a, m_{t-1})
          * sum_{a'} E[Y_t | M_t=m, A_t=a', y_{t-1}] * P(A_t=a' | a_{t-1})

    Assumes exactly one mediator M (the observed variable that is neither A nor Y).
    offset=0 only.
    """

    def _setup(self, batch: Dict[str, torch.Tensor]):
        """Return (model_m, model_y, model_a, m_prev, y_prev, a_obs, int_value_norm, q_levels)."""
        X_obs_norm = batch['X_obs_norm'][0]        # (T, n_max)
        variable_mask = batch['variable_mask'][0]   # (n_max,)
        a_idx = int(batch['intervention_target'].item())
        y_idx = int(batch['query_target'].item())
        int_value_norm = float(batch['intervention_value'].item())

        T = X_obs_norm.shape[0]
        t = int(round(batch['intervention_time_start'].item() * T))

        observed = [i for i in range(variable_mask.shape[0]) if variable_mask[i] > 0.5]
        m_vars = [i for i in observed if i != a_idx and i != y_idx]
        if len(m_vars) != 1:
            raise ValueError(
                f"FrontDoor adjustment requires exactly one mediator M, found {len(m_vars)}: {m_vars}"
            )
        if t < 2:
            raise ValueError(f"FrontDoor adjustment requires t >= 2, got t={t}")

        m_idx = m_vars[0]
        a_series = X_obs_norm[:t, a_idx].cpu().numpy()
        m_series = X_obs_norm[:t, m_idx].cpu().numpy()
        y_series = X_obs_norm[:t, y_idx].cpu().numpy()

        model_m = TabPFNRegressor()
        model_m.fit(np.column_stack([a_series[1:], m_series[:-1]]), m_series[1:])

        model_y = TabPFNRegressor()
        model_y.fit(np.column_stack([m_series[1:], a_series[1:], y_series[:-1]]), y_series[1:])

        model_a = TabPFNRegressor()
        model_a.fit(a_series[:-1].reshape(-1, 1), a_series[1:])

        q_levels = np.linspace(1 / (self.n_mc + 1), 1 - 1 / (self.n_mc + 1), self.n_mc).tolist()
        return (
            model_m, model_y, model_a,
            float(m_series[-1]), float(y_series[-1]), float(a_series[-1]),
            int_value_norm, q_levels,
        )

    def _mc_predict(self, model_m, model_y, model_a, a_val, m_prev, y_prev, a_obs, q_levels) -> float:
        """Double MC integral over (m, a') pairs implementing the front-door formula."""
        m_samples = np.array([
            float(q[0]) for q in model_m.predict(
                np.array([[a_val, m_prev]]),
                output_type="quantiles",
                quantiles=q_levels,
            )
        ])
        a_samples = np.array([
            float(q[0]) for q in model_a.predict(
                np.array([[a_obs]]),
                output_type="quantiles",
                quantiles=q_levels,
            )
        ])
        m_grid, a_grid = np.meshgrid(m_samples, a_samples)
        X_q = np.column_stack([m_grid.ravel(), a_grid.ravel(), np.full(self.n_mc ** 2, y_prev)])
        return float(np.mean(model_y.predict(X_q, output_type="mean")))


class FrontDoorTabPFNInterventional(_FrontDoorTabPFNBase):
    """E[Y_t | do(A_t = a), D_{<t}] via front-door adjustment."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]):
        model_m, model_y, model_a, m_prev, y_prev, a_obs, int_value_norm, q_levels = self._setup(batch)
        return self._mc_predict(model_m, model_y, model_a, int_value_norm, m_prev, y_prev, a_obs, q_levels), None


class FrontDoorTabPFNObservational(_FrontDoorTabPFNBase):
    """E[Y_t | A_t = a_obs, D_{<t}] via front-door adjustment (observational A)."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]):
        model_m, model_y, model_a, m_prev, y_prev, a_obs, _, q_levels = self._setup(batch)
        return self._mc_predict(model_m, model_y, model_a, a_obs, m_prev, y_prev, a_obs, q_levels), None


class FrontDoorTabPFNCausalEffect(_FrontDoorTabPFNBase):
    """E[Y|do(a)] - E[Y|a_obs] via front-door adjustment."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]):
        model_m, model_y, model_a, m_prev, y_prev, a_obs, int_value_norm, q_levels = self._setup(batch)
        pred_int = self._mc_predict(model_m, model_y, model_a, int_value_norm, m_prev, y_prev, a_obs, q_levels)
        pred_obs = self._mc_predict(model_m, model_y, model_a, a_obs, m_prev, y_prev, a_obs, q_levels)
        return pred_int - pred_obs, None
