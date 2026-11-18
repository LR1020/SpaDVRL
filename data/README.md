# SpaDVRL data

This directory documents the datasets used to reproduce the SpaDVRL
experiments.

## Simulated dataset

The fixed simulated RNA--protein dataset used for the paper benchmark is
included in this repository:

```text
data/Simulate_data_RNA_Protein/
  adata_RNA.h5ad
  adata_Protein.h5ad
```

It contains 1,296 matched spatial locations and was generated with the
`ggblocks` model described in the manuscript and supplementary material. Keep
these files unchanged when reproducing the reported benchmark.

## Real datasets

The real datasets are publicly available from the following sources:

| Dataset | Source |
| --- | --- |
| Human lymph node A1 | [SpatialGlue data repository](https://doi.org/10.5281/zenodo.10362607) |
| E15.5 MISAR-seq mouse brain | [National Genomics Data Center, OEP003285](https://www.biosino.org/node/project/detail/OEP003285) |
| E13.0 mouse embryo spatial RNA--ATAC | [UCSC Cell Browser](https://cells.ucsc.edu/?ds=brain-spatial-omics) and [GEO, GSE205055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055) |
| Stereo-CITE-seq mouse thymus | Originally obtained from BGI; the processed input is available from the [SpatialGlue data repository](https://doi.org/10.5281/zenodo.10362607) |

These larger files are intentionally excluded from Git. After downloading the
processed data, arrange the inputs using the following directory and file
names:

```text
data/
  10x_human_lymph_node_A1/
    adata_RNA.h5ad
    adata_ADT.h5ad
    annotation.csv
  Mouse_Brain_E15_S1/
    adata_RNA.h5ad
    adata_ATAC.h5ad
    anno.csv
  Mouse_Embryo_E13_CoMo/
    adata_RNA.h5ad
    adata_ATAC.h5ad
    ground_truth.csv
  Stereo-CITE-seq_Mouse_Thymus/
    adata_RNA.h5ad
    adata_ADT.h5ad
```

