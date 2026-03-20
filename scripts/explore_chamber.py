#!/usr/bin/env python
"""Explore CausalChamber datasets to identify subgraphs for evaluation."""

from dotime.data.causal_chamber import (
    CausalChamberLoader, LT_ACTUATORS, LT_SENSORS, LT_SUBGRAPH_5VAR,
)


def main():
    print("=" * 60)
    print("CausalChamber Exploration")
    print("=" * 60)

    loader = CausalChamberLoader(dataset_name="lt_walks_v1")

    # List experiments
    exps = loader.list_experiments()
    print(f"\nAvailable experiments: {exps}")

    for exp_name in exps:
        print(f"\n--- {exp_name} ---")
        df = loader.load_experiment(exp_name)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        # Show actuator ranges
        for act in LT_ACTUATORS:
            if act in df.columns:
                col = df[act]
                print(f"  Actuator {act}: range=[{col.min():.1f}, {col.max():.1f}], "
                      f"std={col.std():.2f}")

        # Show sensor ranges
        for sens in LT_SENSORS[:6]:
            if sens in df.columns:
                col = df[sens]
                print(f"  Sensor {sens}: range=[{col.min():.1f}, {col.max():.1f}], "
                      f"std={col.std():.2f}")

        # Extract episodes with default subgraph
        episodes = loader.extract_episodes(df)
        print(f"\n  Episodes (5-var subgraph {LT_SUBGRAPH_5VAR}): {len(episodes)}")
        if episodes:
            ep = episodes[0]
            print(f"  First episode: int_var={ep['intervention_var']}, "
                  f"value={ep['intervention_value']:.2f}, "
                  f"X_obs shape={ep['X_obs'].shape}")


if __name__ == "__main__":
    main()
