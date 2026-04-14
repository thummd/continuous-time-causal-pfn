from abc import ABC, abstractmethod
from typing import Dict, List
import torch
from dotime.model.do_over_time_pfn import DoOverTimePFN
from pfns.model.bar_distribution import FullSupportBarDistribution
from chronos import BaseChronosPipeline
import pandas as pd
import numpy as np


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
    

class ObsPFNBD(TrainedBaseline):

    @property
    def checkpoint_path(self) -> str:
        return "/work/dlclarge1/robertsj-dotpfn/do-over-time-pfn/checkpoints/sanity2_/sanity2_bd_obs_only/do_over_time_pfn_best.pt"
    
class DoTPFNBD(TrainedBaseline):

    @property
    def checkpoint_path(self) -> str:
        return "/work/dlclarge1/robertsj-dotpfn/do-over-time-pfn/checkpoints/sanity2_/sanity2_bd_causal/do_over_time_pfn_best.pt"
    
class Chronos2Baseline(SinglePointTimeSeriesBaseline):

    def __init__(self, device: str = "cpu"):
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

    

