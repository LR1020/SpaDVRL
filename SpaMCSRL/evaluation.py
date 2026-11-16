"""Clustering and evaluation utilities used by the final experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors


def mclust_eee(embedding, n_clusters: int, seed: int, n_components: int = 20):
    """Cluster an embedding using the same PCA + Mclust EEE protocol."""
    conda_r_home = Path(sys.prefix) / "lib" / "R"
    if conda_r_home.is_dir():
        os.environ.setdefault("R_HOME", str(conda_r_home))
    import rpy2.robjects as ro

    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("Embedding must be a finite two-dimensional matrix")
    components = min(n_components, embedding.shape[0] - 1, embedding.shape[1])
    reduced = PCA(n_components=components, random_state=seed).fit_transform(embedding)
    ro.r.library("mclust")
    ro.r["set.seed"](seed)
    vector = ro.FloatVector(reduced.ravel(order="F"))
    matrix = ro.r.matrix(vector, nrow=reduced.shape[0], ncol=reduced.shape[1])
    matrix = ro.r["colnames<-"](
        matrix, ro.StrVector([f"V{index + 1}" for index in range(reduced.shape[1])])
    )
    matrix = ro.r["rownames<-"](
        matrix, ro.StrVector([f"cell{index + 1}" for index in range(reduced.shape[0])])
    )
    matrix = ro.r["as.matrix"](matrix)
    result = ro.r["Mclust"](matrix, G=n_clusters, modelNames="EEE")
    return np.asarray(result.rx2("classification"), dtype=int)


def clustering_metrics(truth, prediction):
    truth = np.asarray(truth).astype(str)
    prediction = np.asarray(prediction).astype(str)
    return {
        "ARI": adjusted_rand_score(truth, prediction),
        "NMI": normalized_mutual_info_score(truth, prediction),
        "AMI": adjusted_mutual_info_score(truth, prediction),
        "Homogeneity": homogeneity_score(truth, prediction),
        "V_measure": v_measure_score(truth, prediction),
    }


def spatial_metrics(coords, labels, k: int = 6):
    import networkx as nx

    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels).astype(str)
    n = len(labels)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(
        return_distance=False
    )[:, 1:]
    nla = float(np.mean(labels[neighbors] == labels[:, None]))

    rows = np.repeat(np.arange(n), k)
    cols = neighbors.ravel()
    adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float64)
    graph = nx.from_scipy_sparse_array(adjacency)
    connected_total = 0
    for cluster in np.unique(labels):
        indices = np.flatnonzero(labels == cluster)
        subgraph = graph.subgraph(indices)
        connected_total += max(
            (len(component) for component in nx.connected_components(subgraph)),
            default=0,
        )

    weight_sum = float(adjacency.sum())
    moran_values, geary_values, cluster_weights = [], [], []
    row_idx, col_idx = adjacency.nonzero()
    for cluster in np.unique(labels):
        x = (labels == cluster).astype(float)
        centered = x - x.mean()
        denominator = float(np.dot(centered, centered))
        if denominator == 0:
            continue
        moran_values.append(
            (n / weight_sum) * float(centered @ adjacency @ centered) / denominator
        )
        geary_num = float(np.sum((x[row_idx] - x[col_idx]) ** 2))
        geary_values.append(
            ((n - 1) / (2 * weight_sum)) * geary_num / denominator
        )
        cluster_weights.append(float(x.sum()))

    scaled = (coords - coords.min(axis=0)) / np.maximum(np.ptp(coords, axis=0), 1e-12)
    compact_sum = 0.0
    for cluster in np.unique(labels):
        points = scaled[labels == cluster]
        centroid = points.mean(axis=0, keepdims=True)
        mean_distance = float(cdist(points, centroid).mean())
        compact_sum += len(points) * (1.0 - min(mean_distance / np.sqrt(2.0), 1.0))
    return {
        "Neighborhood_label_agreement": nla,
        "Domain_compactness": compact_sum / n,
        "Largest_connected_component_ratio": connected_total / n,
        "Morans_I": float(np.average(moran_values, weights=cluster_weights)),
        "Gearys_C": float(np.average(geary_values, weights=cluster_weights)),
        "spatial_knn": k,
    }
