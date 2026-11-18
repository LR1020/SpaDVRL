"""Reproduce the seed-2024 SpaDVRL E15.5 clustering and UMAP plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpaDVRL.datasets import prepare_e155
from SpaDVRL.experiment import run_seed2024_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Mouse_Brain_E15_S1"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/e155_seed2024")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_seed2024_experiment(
        dataset_name="E15.5",
        loader=prepare_e155,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        datatype="MISAR-seq",
        epochs=600,
        n_clusters=11,
        palette_mode="greedy",
        spatial_point_size=15,
        spatial_equal_aspect=True,
        umap_point_size=4,
        umap_seed=0,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
