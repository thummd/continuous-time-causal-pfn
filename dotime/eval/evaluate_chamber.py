"""Zero-shot evaluation on CausalChamber."""

import torch
from typing import Dict, List, Optional, Union

from dotime.data.causal_chamber import CausalChamberLoader
from dotime.eval.metrics import (
    compute_rmse, compute_mae,
    compute_quantile_calibration, compute_pinball_metric,
)


@torch.no_grad()
def evaluate_on_chamber(
    model,
    dataset_name: str = "lt_walks_v1",
    experiment_name: str = "smooth_polarizers",
    subgraph_vars: Optional[List[str]] = None,
    query_vars: Optional[List[str]] = None,
    obs_window: int = 50,
    query_time_offset: int = 5,
    max_episodes: int = 100,
    device: str = "cpu",
    tau_levels: Optional[Union[torch.Tensor, List[float]]] = None,
) -> Dict:
    """Evaluate a trained model zero-shot on CausalChamber.

    Parameters
    ----------
    model : DoOverTimePFN (with bar distribution calibrated)
    dataset_name : CausalChamber dataset
    experiment_name : which experiment to evaluate on
    subgraph_vars : variables to include
    query_vars : variables to query (must be in subgraph_vars, should be sensors)
    obs_window : observation window length
    query_time_offset : steps after intervention to query
    max_episodes : maximum number of episodes to evaluate
    device : torch device

    Returns
    -------
    dict with overall and per-variable metrics
    """
    model.eval()

    loader = CausalChamberLoader(
        dataset_name=dataset_name,
        subgraph_vars=subgraph_vars,
    )

    df = loader.load_experiment(experiment_name)
    episodes = loader.extract_episodes(df, obs_window=obs_window)

    if not episodes:
        print("No valid episodes found.")
        return {}

    episodes = episodes[:max_episodes]
    print(f"Evaluating on {len(episodes)} episodes from {experiment_name}")

    # Default query vars: all sensor variables in the subgraph
    if query_vars is None:
        from dotime.data.causal_chamber import LT_SENSORS
        query_vars = [v for v in loader.subgraph_vars if v in LT_SENSORS]

    results = {"per_var": {}, "overall": {}}
    all_preds = []
    all_targets = []
    all_logits = []
    borders = model.bar_head.borders.cpu()

    for qvar in query_vars:
        var_preds = []
        var_targets = []
        var_logits = []

        for ep in episodes:
            if qvar not in ep['var_names']:
                continue

            batch = loader.episode_to_model_input(ep, query_var=qvar,
                                                   query_time_offset=query_time_offset)
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(batch)
            pred = model.bar_head.predict_mean(logits)

            var_preds.append(pred.cpu())
            var_targets.append(batch['Y_true_norm'].cpu())
            if tau_levels is not None:
                var_logits.append(logits.cpu())

        if var_preds:
            preds_t = torch.cat(var_preds)
            targets_t = torch.cat(var_targets)
            var_results = {
                "rmse": compute_rmse(preds_t, targets_t),
                "mae": compute_mae(preds_t, targets_t),
                "n_episodes": len(var_preds),
            }
            if tau_levels is not None and var_logits:
                logits_t = torch.cat(var_logits)
                var_results["pinball_loss"] = compute_pinball_metric(
                    logits_t, borders, targets_t, tau_levels)
                var_results.update(compute_quantile_calibration(
                    logits_t, borders, targets_t, tau_levels))
                all_logits.append(logits_t)
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
        if tau_levels is not None and all_logits:
            all_logits_t = torch.cat(all_logits)
            results["overall"]["pinball_loss"] = compute_pinball_metric(
                all_logits_t, borders, all_targets_t, tau_levels)
            results["overall"].update(compute_quantile_calibration(
                all_logits_t, borders, all_targets_t, tau_levels))

    return results
