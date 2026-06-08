"""
umap_celltypes.py
-----------------
Genera una figura con todos los datasets en un solo panel,
cada uno coloreado por tipo celular, usando las coordenadas
X_umap guardadas por annotate_datasets_v2.py.

Uso:
    python scripts/umap_celltypes.py \\
        --data_dir data \\
        --figures_dir figures/umap_celltypes
"""

import os
import glob
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
warnings.filterwarnings("ignore")


CELLTYPES_ORDER = ["Fibroblast", "Endothelial", "Hepatocyte", "Myeloid", "T/NK", "B"]

CELLTYPE_COLORS = {
    "Fibroblast":  "#FFD54F",
    "Endothelial": "#81C784",
    "Hepatocyte":  "#FF6B6B",
    "Myeloid":     "#4FC3F7",
    "T/NK":        "#DA9EC4",
    "B":           "#FF8A65",
    "Unknown":     "#CCCCCC",
}

GENE_SYMBOL_CANDIDATES = ["feature_name", "gene_name", "gene_symbol", "Symbol", "symbol"]


def normalizar_var_names(adata):
    if not adata.var_names[0].startswith("ENSG"):
        return adata
    for col in GENE_SYMBOL_CANDIDATES:
        if col in adata.var.columns:
            adata.var_names = adata.var[col].astype(str).values
            adata.var_names_make_unique()
            adata.raw = None
            return adata
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--figures_dir", default="figures/umap_celltypes")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # Cargar datasets
    h5ad_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5ad")))
    print(f"Datasets encontrados: {len(h5ad_files)}")

    datasets_loaded = {}
    for h5ad_path in h5ad_files:
        ds_id = (os.path.basename(h5ad_path)
                 .replace(".h5ad", "")
                 .replace("adata_", "")
                 .replace("_raw", ""))
        adata = sc.read_h5ad(h5ad_path)
        adata = normalizar_var_names(adata)
        if "X_umap" not in adata.obsm:
            print(f"  [SKIP] {ds_id} — sin X_umap")
            continue
        if "celltype" not in adata.obs.columns:
            print(f"  [SKIP] {ds_id} — sin celltype")
            continue
        datasets_loaded[ds_id] = adata
        print(f"  Cargado {ds_id}: {adata.shape}")

    if not datasets_loaded:
        print("[ERROR] Sin datasets válidos.")
        return

    n_ds  = len(datasets_loaded)
    ncols = min(n_ds, 3)
    nrows = int(np.ceil(n_ds / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows),
                             squeeze=False)

    fig.suptitle("UMAP por tipo celular — todos los datasets",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, (ds_id, adata) in enumerate(datasets_loaded.items()):
        row = idx // ncols
        col = idx  % ncols
        ax  = axes[row][col]

        umap      = adata.obsm["X_umap"]
        celltypes = adata.obs["celltype"].values

        # Células de tipos no en CELLTYPES_ORDER en gris primero
        mask_otros = np.array([ct not in CELLTYPES_ORDER for ct in celltypes])
        if mask_otros.sum() > 0:
            ax.scatter(umap[mask_otros, 0], umap[mask_otros, 1],
                       c=CELLTYPE_COLORS["Unknown"], s=1.5,
                       rasterized=True, linewidths=0, zorder=1)

        # Un scatter por tipo celular para leyenda correcta
        for ct in CELLTYPES_ORDER:
            mask = celltypes == ct
            if mask.sum() == 0:
                continue
            ax.scatter(umap[mask, 0], umap[mask, 1],
                       c=CELLTYPE_COLORS[ct], s=1.5,
                       rasterized=True, linewidths=0,
                       label=ct, zorder=2)

        ax.set_title(ds_id, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Apagar ejes vacíos si n_ds no es múltiplo de ncols
    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # Leyenda compartida fuera del panel
    handles = [
        mpatches.Patch(color=CELLTYPE_COLORS[ct], label=ct)
        for ct in CELLTYPES_ORDER
    ]
    fig.legend(handles=handles,
               title="Tipo celular",
               title_fontsize=9,
               fontsize=8,
               loc="lower center",
               ncol=len(CELLTYPES_ORDER),
               bbox_to_anchor=(0.5, -0.02),
               frameon=True,
               edgecolor="gray")

    plt.tight_layout()
    out = os.path.join(args.figures_dir, "umap_celltypes_todos.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"\n[OK] Figura guardada: {out}")


if __name__ == "__main__":
    main()
