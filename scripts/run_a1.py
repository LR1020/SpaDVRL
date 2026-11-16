"""Reproduce the seed-2024 SpaMCSRL A1 clustering and UMAP plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpaMCSRL.datasets import prepare_a1
from SpaMCSRL.experiment import run_seed2024_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/10x_human_lymph_node_A1"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/a1_seed2024")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_seed2024_experiment(
        dataset_name="A1",
        loader=prepare_a1,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        datatype="10x",
        epochs=350,
        n_clusters=10,
        palette_mode="hungarian",
        spatial_point_size=10,
        spatial_equal_aspect=False,
        umap_point_size=2,
        umap_seed=0,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
