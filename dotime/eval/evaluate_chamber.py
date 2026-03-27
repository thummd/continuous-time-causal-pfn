"""Zero-shot evaluation on CausalChamber."""

import torch
from typing import Dict, List, Optional, Tuple, Union

from dotime.data.causal_chamber import CausalChamberLoader, LT_SENSORS
from dotime.eval.metrics import (
    compute_rmse, compute_mae,
    compute_quantile_calibration, compute_pinball_metric,
)


def _prepare_chamber_eval(
    dataset_name: str,
    experiment_name: str,
    subgraph_vars: Optional[List[str]],
    obs_window: int,
    max_episodes: int,
    query_vars: Optional[List[str]],
) -> Tuple[Optional[CausalChamberLoader], Optional[List], Optional[List[str]]]:
    """Load episodes and resolve query_vars.

    Returns (loader, episodes, query_vars), or (None, None, None) if no episodes found.
    """
    loader = CausalChamberLoader(dataset_name=dataset_name, subgraph_vars=subgraph_vars)
    df = loader.load_experiment(experiment_name)
    episodes = loader.extract_episodes(df, obs_window=obs_window)

    if not episodes:
        print("No valid episodes found.")
        return None, None, None

    episodes = episodes[:max_episodes]
    print(f"Evaluating on {len(episodes)} episodes from {experiment_name}")

    if query_vars is None:
        query_vars = [v for v in loader.subgraph_vars if v in LT_SENSORS]

    return loader, episodes, query_vars


@torch.no_grad()
def evaluate_on_chamber(
    model,
    dataset_name: str = "lt_walks_v1",
    experiment_name: str = "actuators_white",
    subgraph_vars: Optional[List[str]] = None,
    query_vars: Optional[List[str]] = None,
    obs_window: int = 50,
    query_time_offset: int = 5,
    max_episodes: int = 100,
    device: str = "cpu",
    tau_levels: Optional[Union[torch.Tensor, List[float]]] = None,
) -> Dict:
    """Evaluate a trained model zero-shot on CausalChamber.

    Works with both bar distribution and quantile head models.

    Parameters
    ----------
    model : DoOverTimePFN (bar or quantile head)
    dataset_name : CausalChamber dataset
    experiment_name : which experiment to evaluate on
    subgraph_vars : variables to include
    query_vars : variables to query (must be in subgraph_vars, should be sensors)
    obs_window : observation window length
    query_time_offset : steps after intervention to query
    max_episodes : maximum number of episodes to evaluate
    device : torch device
    tau_levels : quantile levels for calibration metrics (bar head only)

    Returns
    -------
    dict with overall and per-variable metrics
    """
    model.eval()
    head = model.head

    loader, episodes, query_vars = _prepare_chamber_eval(
        dataset_name, experiment_name, subgraph_vars, obs_window, max_episodes, query_vars,
    )
    if loader is None:
        return {}

    # Bar head borders for calibration metrics (if applicable)
    has_bar = hasattr(model, 'bar_head') and model.bar_head is not None
    borders = model.bar_head.borders.cpu() if has_bar else None

    results = {"per_var": {}, "overall": {}}
    all_preds = []
    all_targets = []
    all_outputs = []

    for qvar in query_vars:
        var_preds = []
        var_targets = []
        var_outputs = []

        for ep in episodes:
            if qvar not in ep['var_names']:
                continue

            batch = loader.episode_to_model_input(
                ep, query_var=qvar, query_time_offset=query_time_offset,
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch)
            pred = head.predict_mean(output)

            var_preds.append(pred.cpu())
            var_targets.append(batch['Y_true_norm'].cpu())
            var_outputs.append(output.cpu())

        if var_preds:
            preds_t = torch.cat(var_preds)
            targets_t = torch.cat(var_targets)
            var_results = {
                "rmse": compute_rmse(preds_t, targets_t),
                "mae": compute_mae(preds_t, targets_t),
                "n_episodes": len(var_preds),
            }
            if tau_levels is not None and has_bar:
                outputs_t = torch.cat(var_outputs)
                var_results["pinball_loss"] = compute_pinball_metric(
                    outputs_t, borders, targets_t, tau_levels)
                var_results.update(compute_quantile_calibration(
                    outputs_t, borders, targets_t, tau_levels))
                all_outputs.append(outputs_t)
            results["per_var"][qvar] = var_results
            all_preds.append(preds_t)
            all_targets.append(targets_t)

    if all_preds:
        all_preds_t = torch.cat(all_preds)
        all_targets_t = torch.cat(all_targets)
        results["overall"] = {
            "rmse": compute_rmse(all_preds_t, all_targets_t),
            "mae": compute_mae(all_preds_t, all_targets_t),
            "n_episodes": len(all_preds_t),
        }
        if tau_levels is not None and has_bar and all_outputs:
            all_outputs_t = torch.cat(all_outputs)
            results["overall"]["pinball_loss"] = compute_pinball_metric(
                all_outputs_t, borders, all_targets_t, tau_levels)
            results["overall"].update(compute_quantile_calibration(
                all_outputs_t, borders, all_targets_t, tau_levels))

    return results
