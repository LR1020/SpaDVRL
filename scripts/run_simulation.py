"""Reproduce the seed-2024 SpaDVRL simulation clustering and UMAP plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpaDVRL.datasets import prepare_simulation
from SpaDVRL.experiment import run_seed2024_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Simulate_data_RNA_Protein"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/simulation_seed2024")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_seed2024_experiment(
        dataset_name="Simulation",
        loader=prepare_simulation,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        datatype="SPOTS",
        epochs=600,
        n_clusters=5,
        palette_mode="simulation",
        spatial_point_size=15,
        spatial_equal_aspect=True,
        umap_point_size=4,
        umap_seed=2024,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
