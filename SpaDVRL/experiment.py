"""Single-seed experiment pipeline for public SpaDVRL examples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment

from .datasets import set_seed
from .evaluation import clustering_metrics, mclust_eee
from .trainer import TrainSpaDVRL


SEED = 2024
CLUSTER_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4a90e2",
    "#f5a623",
    "#a7a8aa",
    "#d62728",
)
SIMULATION_COLORS = {
    "1": "#9467bd",
    "2": "#1f77b4",
    "3": "#ff7f0e",
    "4": "#2ca02c",
    "5": "#d62728",
}


def _natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(value))
    ]


def _hungarian_mapping(prediction: np.ndarray, truth: np.ndarray) -> dict[str, str]:
    predicted = np.asarray(prediction).astype(str)
    reference = np.asarray(truth).astype(str)
    predicted_levels = sorted(np.unique(predicted), key=_natural_key)
    reference_levels = list(pd.unique(reference))
    table = pd.crosstab(
        pd.Categorical(predicted, categories=predicted_levels),
        pd.Categorical(reference, categories=reference_levels),
        dropna=False,
    ).to_numpy()
    rows, columns = linear_sum_assignment(-table)
    return {
        predicted_levels[row]: reference_levels[column]
        for row, column in zip(rows, columns)
    }


def _matched_palette(
    prediction: np.ndarray,
    truth: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, str], list[tuple[str, str]]]:
    predicted = np.asarray(prediction).astype(str)
    reference = np.asarray(truth).astype(str)
    predicted_levels = sorted(np.unique(predicted), key=_natural_key)
    reference_levels = list(pd.unique(reference))
    truth_palette = {
        label: CLUSTER_COLORS[index]
        for index, label in enumerate(reference_levels)
    }

    if mode == "hungarian":
        mapping = _hungarian_mapping(predicted, reference)
        palette = {
            label: truth_palette[target] for label, target in mapping.items()
        }
    elif mode == "greedy":
        counts = pd.DataFrame(
            {"predicted": predicted, "reference": reference}
        ).value_counts()
        palette = {}
        for (label, target), _ in counts.items():
            color = truth_palette[target]
            if label not in palette and color not in palette.values():
                palette[label] = color
    else:
        raise ValueError(f"Unknown palette matching mode: {mode}")

    unused_colors = [
        color for color in truth_palette.values() if color not in palette.values()
    ]
    for label in predicted_levels:
        if label not in palette:
            palette[label] = unused_colors.pop(0)
    legend = [(label, label) for label in predicted_levels]
    return predicted, palette, legend


def _simulation_palette(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, dict[str, str], list[tuple[str, str]]]:
    reference = np.asarray(truth, dtype=int) + 1
    mapping = _hungarian_mapping(prediction, reference)
    display = np.asarray([mapping[str(label)] for label in prediction], dtype=str)
    legend = [
        ("2", "Factor 1"),
        ("3", "Factor 2"),
        ("4", "Factor 3"),
        ("5", "Factor 4"),
        ("1", "Background"),
    ]
    return display, SIMULATION_COLORS, legend


def _style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=10, fontweight="normal", pad=4)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def _add_legend(
    axis: plt.Axes,
    legend: list[tuple[str, str]],
    palette: dict[str, str],
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[value],
            markeredgecolor="none",
            markersize=6,
            label=label,
        )
        for value, label in legend
    ]
    axis.legend(
        handles=handles,
        title="Clusters",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _plot_embedding(
    coordinates: np.ndarray,
    labels: np.ndarray,
    palette: dict[str, str],
    legend: list[tuple[str, str]],
    output_dir: Path,
    stem: str,
    point_size: float,
    equal_aspect: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(4.2, 4.0), dpi=200)
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=point_size,
        c=[palette[str(label)] for label in labels],
        linewidths=0,
        rasterized=True,
    )
    axis.set_box_aspect(1)
    if equal_aspect:
        axis.set_aspect("equal")
    _style_axis(axis, "SpaDVRL")
    _add_legend(axis, legend, palette)
    figure.subplots_adjust(left=0.05, right=0.78, bottom=0.05, top=0.92)
    _save_figure(figure, output_dir, stem)


def run_seed2024_experiment(
    *,
    dataset_name: str,
    loader: Callable,
    data_dir: Path,
    output_dir: Path,
    datatype: str,
    epochs: int,
    n_clusters: int,
    palette_mode: str,
    spatial_point_size: float,
    spatial_equal_aspect: bool,
    umap_point_size: float,
    umap_seed: int,
    device_name: str,
) -> None:
    """Train SpaDVRL once with seed 2024 and save clustering and UMAP plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    data, truth = loader(data_dir, SEED)
    device = torch.device(
        device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu"
    )

    trainer = TrainSpaDVRL(
        data=data,
        datatype=datatype,
        device=device,
        random_seed=SEED,
        learning_rate=1e-4,
        weight_decay=0.0,
        epochs=epochs,
        dim_output=64,
        n_clusters=n_clusters,
        gnn_encoder="gcn",
        decoder_hidden_mult_omics1=2.0,
        decoder_hidden_mult_omics2=1.5,
        warmup_fraction=0.2,
        common_fusion_mode="bidirectional_gate",
        private_fusion_mode="attention",
        feature_fusion_mode="equal",
    )
    result = trainer.train()
    embedding = np.asarray(result["SpaDVRL"], dtype=np.float32)
    labels = mclust_eee(embedding, n_clusters, SEED)
    metrics = clustering_metrics(truth, labels)

    if palette_mode == "simulation":
        display_labels, palette, legend = _simulation_palette(labels, truth)
    else:
        display_labels, palette, legend = _matched_palette(
            labels, truth, palette_mode
        )

    adata = data["adata_omics1"]
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    holder = sc.AnnData(X=np.asarray(embedding, dtype=np.float32))
    sc.pp.neighbors(holder, n_neighbors=10, use_rep="X", random_state=umap_seed)
    sc.tl.umap(holder, random_state=umap_seed)
    umap = np.asarray(holder.obsm["X_umap"], dtype=float)

    stem = dataset_name.replace(".", "p")
    _plot_embedding(
        spatial,
        display_labels,
        palette,
        legend,
        output_dir,
        f"{stem}_seed2024_spatial_clusters",
        spatial_point_size,
        spatial_equal_aspect,
    )
    _plot_embedding(
        umap,
        display_labels,
        palette,
        legend,
        output_dir,
        f"{stem}_seed2024_umap",
        umap_point_size,
        False,
    )

    observation_names = np.asarray(adata.obs_names, dtype=str)
    pd.DataFrame(
        {
            "spot": observation_names,
            "cluster": labels,
            "display_cluster": display_labels,
            "reference": np.asarray(truth).astype(str),
        }
    ).to_csv(output_dir / f"{stem}_seed2024_cluster_labels.csv", index=False)
    pd.DataFrame(
        {
            "spot": observation_names,
            "UMAP1": umap[:, 0],
            "UMAP2": umap[:, 1],
            "cluster": labels,
        }
    ).to_csv(output_dir / f"{stem}_seed2024_umap_coordinates.csv", index=False)

    configuration = {
        "dataset": dataset_name,
        "seed": SEED,
        "umap_seed": umap_seed,
        "epochs": epochs,
        "n_clusters": n_clusters,
        "model_version": result["model_version"],
        "common_fusion_mode": result["common_fusion_mode"],
        "specific_fusion_mode": result["private_fusion_mode"],
        "feature_fusion_mode": result["feature_fusion_mode"],
        **{name: float(value) for name, value in metrics.items()},
    }
    with (output_dir / f"{stem}_seed2024_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(configuration, handle, indent=2)
    np.savez_compressed(
        output_dir / f"{stem}_SpaDVRL_seed2024.npz",
        embedding=embedding,
        labels=labels,
        truth=np.asarray(truth),
        obs_names=observation_names,
    )
    print(json.dumps(configuration, indent=2), flush=True)
