from abc import ABC, abstractmethod
from typing import Dict, List
import torch
from dotime.model.do_over_time_pfn import DoOverTimePFN
from pfns.model.bar_distribution import FullSupportBarDistribution
import pandas as pd
import numpy as np
from tabpfn import TabPFNRegressor


class SinglePointTimeSeriesBaseline(ABC):

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        batch = {
            'X_obs_norm': X_norm.unsqueeze(0).to(device),
            'variable_mask': variable_mask.unsqueeze(0).to(device),
            'intervention_target': torch.tensor([int_target], dtype=torch.long, device=device),
            'intervention_type': torch.tensor([0], dtype=torch.long, device=device),
            'intervention_value': torch.tensor([int_value_norm], dtype=torch.float32, device=device),
            'intervention_time_start': torch.tensor([time_start], dtype=torch.float32, device=device),
            'intervention_time_end': torch.tensor([time_end], dtype=torch.float32, device=device),
            'query_target': torch.tensor([q_idx], dtype=torch.long, device=device),
            'query_time': torch.tensor([query_time_idx / T], dtype=torch.float32, device=device),
        }
        """
        pass

    @property
    def checkpoint_path(self) -> str:
        return ""


class TrainedBaseline(SinglePointTimeSeriesBaseline):

    def __init__(self, device: str = "cpu"):
        # Load a trained model from the given path
        self.ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self.model = DoOverTimePFN(**self.ckpt['config'])
        self.model.load_state_dict(self.ckpt['model_state_dict'], strict=False)
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Restore bar distribution if applicable
        head_type = self.ckpt.get('head_type', 'bar')
        if head_type == 'bar' and self.ckpt.get('borders') is not None:
            
            borders = self.ckpt['borders']
            bar_dist = FullSupportBarDistribution(borders)
            self.model.bar_head.set_bar_distribution(bar_dist, borders)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Run the model on the input batch and return predictions
        with torch.no_grad():
            output =self.model(batch)
            pred = self.model.head.predict_mean(output)
        return pred.cpu().item(), output.cpu()

class ExampleTrainedBaseline(TrainedBaseline):

    @property
    def checkpoint_path(self) -> str:
        return "/work/dlclarge1/robertsj-dotpfn/do-over-time-pfn/checkpoints/causal_effect/do_over_time_pfn_best.pt"
    
class AR1Baseline(SinglePointTimeSeriesBaseline):

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch['X_obs_norm'][:, -1, batch['query_target']].cpu().item(), None
    

class BackDoorObsPFNCausalEffect(TrainedBaseline):

    @property
    def checkpoint_path(self) -> str:
        return "/work/dlclarge1/robertsj-dotpfn/do-over-time-pfn/checkpoints/sanity2_/sanity2_bd_obs_only/do_over_time_pfn_best.pt"
    
class BackDoorDoTPFNCausalEffect(TrainedBaseline):

    @property
    def checkpoint_path(self) -> str:
        return "/work/dlclarge1/robertsj-dotpfn/do-over-time-pfn/checkpoints/sanity2_/sanity2_bd_causal/do_over_time_pfn_best.pt"
    
class Chronos2Observational(SinglePointTimeSeriesBaseline):

    def __init__(self, device: str = "cpu"):
        from chronos import BaseChronosPipeline
        self.device = device
        self.pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        context_df = pd.DataFrame(batch["X_obs_norm"].squeeze(0).cpu().numpy())

        batch["X_obs_norm"].squeeze(0)[-1, :]

        future_df = context_df.iloc[[-1]]
        future_df.iloc[:, batch["variable_mask"].squeeze(0).cpu().numpy() == 1] = np.nan
        future_df.iloc[:, batch["intervention_target"].cpu().numpy().item()] = batch["intervention_value"].cpu().numpy().item()

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
        

class ZeroBaseline(SinglePointTimeSeriesBaseline):

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return 0, None
    

class _BackDoorTabPFNBase(SinglePointTimeSeriesBaseline):
    """Shared back-door adjustment logic using two TabPFN regressors.

    Fits in-context from the observational trajectory:
      - model_x: p(X_t | X_{t-1})
      - model_y: p(Y_t | A_t, X_t, Y_{t-1})

    Subclasses decide what to return from the two MC predictions:
      pred_int = E[Y_t | do(A_t = a)]    (back-door adjusted, intervention value)
      pred_obs = E[Y_t | A_t = a_obs]    (back-door adjusted, last observed A as
                                          stand-in for natural A_t)

    Assumes exactly one observed confounder X (neither A nor Y). offset=0 only.
    """

    def _init_(self, n_mc: int = 100):
        self.n_mc = n_mc

    def _setup(self, batch: Dict[str, torch.Tensor]):
        """Fit both TabPFN models and return (model_y, x_t_samples, y_prev,
        int_value_norm, a_obs), or None if the structure is not as expected."""
        X_obs_norm = batch['X_obs_norm'][0]       # (T, n_max)
        variable_mask = batch['variable_mask'][0]  # (n_max,)
        a_idx = int(batch['intervention_target'].item())
        y_idx = int(batch['query_target'].item())
        int_value_norm = float(batch['intervention_value'].item())

        T = X_obs_norm.shape[0]
        t = int(round(batch['intervention_time_start'].item() * T))
        t_prev = t - 1

        # Identify X: the single observed variable that is neither A nor Y
        observed = [i for i in range(variable_mask.shape[0]) if variable_mask[i] > 0.5]
        x_vars = [i for i in observed if i != a_idx and i != y_idx]
        if len(x_vars) != 1:
            raise ValueError(
                f"BackDoor adjustment requires exactly one confounder X, found {len(x_vars)}: {x_vars}"
            )
        if t_prev < 1:
            raise ValueError(
                f"BackDoor adjustment requires at least one lag (t_prev >= 1), got t_prev={t_prev}"
            )

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

        # Sample x_t via inverse-CDF from p(x_t | x_{t-1})
        q_levels = np.linspace(1 / (self.n_mc + 1), 1 - 1 / (self.n_mc + 1), self.n_mc).tolist()
        x_t_samples = np.array([
            float(q[0]) for q in model_x.predict(
                x_series[-1:].reshape(1, 1),
                output_type="quantiles",
                quantiles=q_levels,
            )
        ])

        y_prev = float(y_series[-1])
        a_obs = float(a_series[-1])  # last observed A as stand-in for natural A_t
        return model_y, x_t_samples, y_prev, int_value_norm, a_obs

    def _mc_predict(self, model_y, a_val: float, x_t_samples, y_prev: float) -> float:
        """One MC integral: E[Y_t | a_val, X_t, y_prev] averaged over x_t_samples."""
        X_q = np.column_stack([
            np.full(self.n_mc, a_val),
            x_t_samples,
            np.full(self.n_mc, y_prev),
        ])
        return float(np.mean(model_y.predict(X_q, output_type="mean")))


class BackDoorTabPFNInterventional(_BackDoorTabPFNBase):
    """Predicts E[Y_t | do(A_t = a), D_{<t}] via back-door adjustment."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_y, x_t_samples, y_prev, int_value_norm, _ = self._setup(batch)
        return self._mc_predict(model_y, int_value_norm, x_t_samples, y_prev), None


class BackDoorTabPFNObservational(_BackDoorTabPFNBase):
    """Predicts E[Y_t | A_t = a_obs, D_{<t}] via back-door adjustment (no intervention)."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_y, x_t_samples, y_prev, _, a_obs = self._setup(batch)
        return self._mc_predict(model_y, a_obs, x_t_samples, y_prev), None


class BackDoorTabPFNCausalEffect(_BackDoorTabPFNBase):
    """Predicts the causal effect E[Y_t | do(A_t = a), D_{<t}] - E[Y_t | A_t = a_obs, D_{<t}]."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_y, x_t_samples, y_prev, int_value_norm, a_obs = self._setup(batch)
        pred_int = self._mc_predict(model_y, int_value_norm, x_t_samples, y_prev)
        pred_obs = self._mc_predict(model_y, a_obs, x_t_samples, y_prev)
        return pred_int - pred_obs, None


class _FrontDoorTabPFNBase(SinglePointTimeSeriesBaseline):
    """Shared front-door adjustment logic using three TabPFN regressors.

    Fits in-context from the observational trajectory:
      - model_m: P(M_t | A_t, M_{t-1})
      - model_y: P(Y_t | M_t, A_t, Y_{t-1})
      - model_a: P(A_t | A_{t-1})   (marginal for integrating out A' in inner sum)

    Front-door adjustment formula:
      E[Y_t | do(A_t=a), D_{<t}]
        = sum_m P(M_t=m | A_t=a, m_{t-1})
          * sum_{a'} E[Y_t | M_t=m, A_t=a', y_{t-1}] * P(A_t=a' | a_{t-1})

    The outer integral over m de-confounds A->M (unconfounded by structure).
    The inner integral over a' de-confounds M->Y by blocking M<-A<-U->Y.

    Assumes exactly one mediator M (the observed variable that is neither A nor Y).
    offset=0 only.
    """

    def _setup(self, batch: Dict[str, torch.Tensor]):
        """Fit all three TabPFN models and return
        (model_m, model_y, model_a, m_prev, y_prev, a_obs, int_value_norm, q_levels).
        """
        X_obs_norm = batch['X_obs_norm'][0]        # (T, n_max)
        variable_mask = batch['variable_mask'][0]   # (n_max,)
        a_idx = int(batch['intervention_target'].item())
        y_idx = int(batch['query_target'].item())
        int_value_norm = float(batch['intervention_value'].item())

        T = X_obs_norm.shape[0]
        t = int(round(batch['intervention_time_start'].item() * T))
        t_prev = t - 1

        # Identify M: the single observed variable that is neither A nor Y
        observed = [i for i in range(variable_mask.shape[0]) if variable_mask[i] > 0.5]
        m_vars = [i for i in observed if i != a_idx and i != y_idx]
        if len(m_vars) != 1:
            raise ValueError(
                f"FrontDoor adjustment requires exactly one mediator M, found {len(m_vars)}: {m_vars}"
            )
        if t_prev < 1:
            raise ValueError(
                f"FrontDoor adjustment requires at least one lag (t_prev >= 1), got t_prev={t_prev}"
            )

        m_idx = m_vars[0]
        a_series = X_obs_norm[:t, a_idx].cpu().numpy()
        m_series = X_obs_norm[:t, m_idx].cpu().numpy()
        y_series = X_obs_norm[:t, y_idx].cpu().numpy()

        # model_m: P(M_t | A_t, M_{t-1})
        model_m = TabPFNRegressor()
        model_m.fit(
            np.column_stack([a_series[1:], m_series[:-1]]),
            m_series[1:],
        )

        # model_y: P(Y_t | M_t, A_t, Y_{t-1})
        model_y = TabPFNRegressor()
        model_y.fit(
            np.column_stack([m_series[1:], a_series[1:], y_series[:-1]]),
            y_series[1:],
        )

        # model_a: P(A_t | A_{t-1})  — natural marginal for inner sum over a'
        model_a = TabPFNRegressor()
        model_a.fit(a_series[:-1].reshape(-1, 1), a_series[1:])

        q_levels = np.linspace(1 / (self.n_mc + 1), 1 - 1 / (self.n_mc + 1), self.n_mc).tolist()
        m_prev = float(m_series[-1])
        y_prev = float(y_series[-1])
        a_obs = float(a_series[-1])
        return model_m, model_y, model_a, m_prev, y_prev, a_obs, int_value_norm, q_levels

    def _mc_predict(
        self,
        model_m,
        model_y,
        model_a,
        a_val: float,
        m_prev: float,
        y_prev: float,
        a_obs: float,
        q_levels,
    ) -> float:
        """Double MC integral implementing the front-door formula.

        Outer loop: sample m from P(M_t | A_t=a_val, m_prev).
        Inner loop: sample a' from P(A_t | a_obs) and average E[Y | m, a', y_prev].
        Result is the mean over all n_mc^2 (m, a') combinations.
        """
        # Sample m_t quantiles from P(M_t | A_t=a_val, M_{t-1}=m_prev)
        m_samples = np.array([
            float(q[0]) for q in model_m.predict(
                np.array([[a_val, m_prev]]),
                output_type="quantiles",
                quantiles=q_levels,
            )
        ])

        # Sample a' quantiles from P(A_t | A_{t-1}=a_obs)
        a_samples = np.array([
            float(q[0]) for q in model_a.predict(
                np.array([[a_obs]]),
                output_type="quantiles",
                quantiles=q_levels,
            )
        ])

        # Evaluate model_y on the full (m, a') Cartesian product — shape (n_mc^2, 3)
        m_grid, a_grid = np.meshgrid(m_samples, a_samples)
        X_q = np.column_stack([
            m_grid.ravel(),
            a_grid.ravel(),
            np.full(self.n_mc ** 2, y_prev),
        ])
        return float(np.mean(model_y.predict(X_q, output_type="mean")))


class FrontDoorTabPFNInterventional(_FrontDoorTabPFNBase):
    """Predicts E[Y_t | do(A_t = a), D_{<t}] via front-door adjustment."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_m, model_y, model_a, m_prev, y_prev, a_obs, int_value_norm, q_levels = self._setup(batch)
        return self._mc_predict(model_m, model_y, model_a, int_value_norm, m_prev, y_prev, a_obs, q_levels), None


class FrontDoorTabPFNObservational(_FrontDoorTabPFNBase):
    """Predicts E[Y_t | A_t = a_obs, D_{<t}] via front-door adjustment (no intervention)."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_m, model_y, model_a, m_prev, y_prev, a_obs, _, q_levels = self._setup(batch)
        return self._mc_predict(model_m, model_y, model_a, a_obs, m_prev, y_prev, a_obs, q_levels), None


class FrontDoorTabPFNCausalEffect(_FrontDoorTabPFNBase):
    """Predicts the causal effect E[Y_t | do(A_t = a), D_{<t}] - E[Y_t | A_t = a_obs, D_{<t}]."""

    def __init__(self, device: str = "cpu", n_mc: int = 100):
        self.n_mc = n_mc
        self.device = device

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_m, model_y, model_a, m_prev, y_prev, a_obs, int_value_norm, q_levels = self._setup(batch)
        pred_int = self._mc_predict(model_m, model_y, model_a, int_value_norm, m_prev, y_prev, a_obs, q_levels)
        pred_obs = self._mc_predict(model_m, model_y, model_a, a_obs,          m_prev, y_prev, a_obs, q_levels)
        return pred_int - pred_obs, None

