"""Dataset loaders matching the final SpaDVRL experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA

from .preprocess import (
    clr_normalize_each_cell,
    construct_neighbor_graph,
    lsi,
    pca,
)


DEFAULT_SEEDS = (2022, 2023, 2024, 2025, 2026)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def prepare_a1(data_dir: str | Path, seed: int):
    set_seed(seed)
    data_dir = Path(data_dir)
    rna = sc.read_h5ad(data_dir / "adata_RNA.h5ad")
    adt = sc.read_h5ad(data_dir / "adata_ADT.h5ad")
    rna.var_names_make_unique()
    adt.var_names_make_unique()

    annotation = pd.read_csv(data_dir / "annotation.csv").set_index("Barcode")
    truth = annotation.loc[rna.obs_names.astype(str), "manual-anno"].to_numpy()

    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    rna.obsm["feat"] = pca(
        rna[:, rna.var["highly_variable"]], n_comps=adt.n_vars - 1
    )

    adt = clr_normalize_each_cell(adt)
    sc.pp.scale(adt)
    adt.obsm["feat"] = pca(adt, n_comps=adt.n_vars - 1)
    return construct_neighbor_graph(rna, adt, datatype="10x"), truth


def prepare_e155(data_dir: str | Path, seed: int):
    set_seed(seed)
    data_dir = Path(data_dir)
    rna = sc.read_h5ad(data_dir / "adata_RNA.h5ad")
    atac = sc.read_h5ad(data_dir / "adata_ATAC.h5ad")
    rna.var_names_make_unique()
    atac.var_names_make_unique()

    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    rna.obsm["feat"] = pca(
        rna[:, rna.var["highly_variable"]], n_comps=50
    )

    atac = atac[rna.obs_names].copy()
    lsi(atac, use_highly_variable=False, n_components=51)
    atac.obsm["feat"] = atac.obsm["X_lsi"].copy()
    annotation = pd.read_csv(data_dir / "anno.csv").set_index("barcode")
    truth = annotation.loc[rna.obs_names.astype(str), "cluster"].to_numpy()
    data = construct_neighbor_graph(rna, atac, datatype="MISAR-seq")
    return data, truth


def prepare_e13(data_dir: str | Path, seed: int):
    set_seed(seed)
    data_dir = Path(data_dir)
    rna = sc.read_h5ad(data_dir / "adata_RNA.h5ad")
    atac = sc.read_h5ad(data_dir / "adata_ATAC.h5ad")
    truth_table = pd.read_csv(data_dir / "ground_truth.csv", dtype=str).set_index("spot")
    truth = truth_table.loc[rna.obs_names.astype(str), "truth"].to_numpy()
    rna.obsm["feat"] = PCA(
        50, svd_solver="randomized", random_state=seed
    ).fit_transform(_dense(rna.X))
    atac.obsm["feat"] = PCA(
        50, svd_solver="randomized", random_state=seed
    ).fit_transform(_dense(atac.X))
    data = construct_neighbor_graph(
        rna, atac, datatype="Spatial-epigenome-transcriptome"
    )
    return data, truth


def prepare_simulation(data_dir: str | Path, seed: int):
    set_seed(seed)
    data_dir = Path(data_dir)
    rna = sc.read_h5ad(data_dir / "adata_RNA.h5ad")
    adt_path = data_dir / "adata_ADT.h5ad"
    if not adt_path.exists():
        adt_path = data_dir / "adata_Protein.h5ad"
    adt = sc.read_h5ad(adt_path)
    rna.var_names_make_unique()
    adt.var_names_make_unique()

    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    rna.obsm["feat"] = pca(
        rna[:, rna.var["highly_variable"]], n_comps=adt.n_vars - 1
    )
    adt = clr_normalize_each_cell(adt)
    sc.pp.scale(adt)
    adt.obsm["feat"] = pca(adt, n_comps=adt.n_vars - 1)

    spfac = np.asarray(rna.obsm["spfac"])
    active = spfac.sum(axis=1) > 0
    truth = np.zeros(rna.n_obs, dtype=int)
    truth[active] = np.argmax(spfac[active], axis=1) + 1
    return construct_neighbor_graph(rna, adt, datatype="SPOTS"), truth


def prepare_thymus(data_dir: str | Path, seed: int):
    set_seed(seed)
    data_dir = Path(data_dir)
    rna = sc.read_h5ad(data_dir / "adata_RNA.h5ad")
    adt = sc.read_h5ad(data_dir / "adata_ADT.h5ad")
    rna.var_names_make_unique()
    adt.var_names_make_unique()

    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.filter_cells(rna, min_genes=80)
    sc.pp.filter_genes(adt, min_cells=50)
    adt = adt[rna.obs_names].copy()
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    rna.raw = rna

    rna_high = rna[:, rna.var["highly_variable"]].copy()
    sc.pp.scale(rna_high, max_value=10)
    adt = clr_normalize_each_cell(adt)
    adt.layers["clr"] = np.asarray(adt.X, dtype=np.float32).copy()
    sc.pp.scale(adt, max_value=10)
    n_components = min(
        48, adt.n_vars - 1, rna_high.n_obs - 1, rna_high.n_vars - 1
    )
    rna.obsm["feat"] = pca(rna_high, n_comps=n_components).astype(np.float32)
    adt.obsm["feat"] = pca(adt, n_comps=n_components).astype(np.float32)
    return construct_neighbor_graph(rna, adt, datatype="Stereo-CITE-seq")


DATASETS = {
    "A1": (prepare_a1, "10x", 10, 350),
    "E15.5": (prepare_e155, "MISAR-seq", 11, 600),
    "E13": (prepare_e13, "Spatial-epigenome-transcriptome", 14, 800),
    "Simulation": (prepare_simulation, "SPOTS", 5, 600),
}
