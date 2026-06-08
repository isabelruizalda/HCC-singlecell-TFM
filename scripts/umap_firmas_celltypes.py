"""
umap_firmas_celltypes.py
------------------------
Para cada grupo funcional genera una figura con 2 filas x 5 columnas:
  - Fila 1: UMAP de cada dataset coloreado por score de la firma
  - Fila 2: UMAP de cada dataset coloreado por tipo celular

Usa las coordenadas X_umap guardadas por annotate_datasets_v2.py.
NO renormaliza — usa adata.X tal cual está guardado por annotate.

Uso:
    python scripts/umap_firmas_celltypes.py \\
        --data_dir data \\
        --genes_dir 170_genes \\
        --genes_file genes_filtrados_manual.xlsx \\
        --figures_dir figures/umap_firmas_celltypes \\
        --percentil 75
"""

import os
import re
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────
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
CMAP_SCORE = "RdBu_r"


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
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


def safe_name(s):
    s = s.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return re.sub(r"_+", "_", s).strip("_")


def calcular_score(adata, genes, score_key):
    adata_pp = adata.copy()
    genes_presentes = [g for g in genes if g in adata_pp.var_names]
    if not genes_presentes:
        return np.zeros(adata.n_obs)
    sc.tl.score_genes(adata_pp, gene_list=genes_presentes, score_name=score_key)
    return adata_pp.obs[score_key].values


# ──────────────────────────────────────────────────────────────
# FIGURA PRINCIPAL POR FIRMA
# ──────────────────────────────────────────────────────────────
def figura_firma(grupo, genes, datasets_loaded, figures_dir, percentil, num=None):
    """
    2 filas x n_ds columnas + 1 columna colorbar:
      Fila 0: score de la firma por dataset
      Fila 1: tipo celular por dataset
    """
    ds_ids = list(datasets_loaded.keys())
    n_ds   = len(ds_ids)
    score_key = f"score_{safe_name(grupo)[:20]}"

    # Calcular scores para todos los datasets
    scores_por_ds = {}
    for ds_id, adata in datasets_loaded.items():
        scores_por_ds[ds_id] = calcular_score(adata, genes, score_key)

    # Escala de color común para scores
    all_scores = np.concatenate(list(scores_por_ds.values()))
    vmin  = float(np.percentile(all_scores, 2))
    vmax  = float(np.percentile(all_scores, 98))
    if vmax <= vmin:
        vmax = vmin + 0.01
    norm  = Normalize(vmin=vmin, vmax=vmax)
    umbral = np.percentile(all_scores, percentil)

    # Layout: 2 filas x (n_ds cols + 1 colorbar estrecha)
    panel_size = 4
    fig = plt.figure(figsize=(panel_size * n_ds + 0.8, panel_size * 2 + 0.8))
    gs  = fig.add_gridspec(2, n_ds + 1,
                           width_ratios=[panel_size] * n_ds + [0.25],
                           hspace=0.08, wspace=0.05)

    titulo = f"{num}. {grupo}" if num else grupo
    fig.suptitle(f"{titulo}\n({len(genes)} genes — umbral percentil {percentil})",
                 fontsize=11, fontweight="bold", y=1.02)

    # ── Fila 0: score de la firma ──
    for col, ds_id in enumerate(ds_ids):
        ax   = fig.add_subplot(gs[0, col])
        adata = datasets_loaded[ds_id]

        if "X_umap" not in adata.obsm:
            ax.axis("off")
            continue

        umap  = adata.obsm["X_umap"]
        score = scores_por_ds[ds_id]
        mask_bajo = score < umbral

        # Células bajo umbral en gris
        ax.scatter(umap[mask_bajo, 0], umap[mask_bajo, 1],
                   c="#EEEEEE", s=1.5, rasterized=True, linewidths=0)
        # Células sobre umbral coloreadas por score
        if (~mask_bajo).sum() > 0:
            ax.scatter(umap[~mask_bajo, 0], umap[~mask_bajo, 1],
                       c=score[~mask_bajo], cmap=CMAP_SCORE,
                       norm=norm, s=2, rasterized=True, linewidths=0)

        if col == 0:
            ax.set_ylabel("Score firma", fontsize=8, labelpad=4)
        ax.set_title(ds_id, fontsize=8, fontweight="bold")
        ax.axis("off")

    # Colorbar para fila 0 — ocupa solo la fila 0
    ax_cb = fig.add_subplot(gs[0, n_ds])
    sm = ScalarMappable(cmap=CMAP_SCORE, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Score firma", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    # ── Fila 1: tipo celular ──
    for col, ds_id in enumerate(ds_ids):
        ax    = fig.add_subplot(gs[1, col])
        adata = datasets_loaded[ds_id]

        if "X_umap" not in adata.obsm:
            ax.axis("off")
            continue

        umap      = adata.obsm["X_umap"]
        celltypes = adata.obs["celltype"].values

        # Tipos no en CELLTYPES_ORDER en gris
        mask_otros = np.array([ct not in CELLTYPES_ORDER for ct in celltypes])
        if mask_otros.sum() > 0:
            ax.scatter(umap[mask_otros, 0], umap[mask_otros, 1],
                       c=CELLTYPE_COLORS["Unknown"], s=1.5,
                       rasterized=True, linewidths=0, zorder=1)

        for ct in CELLTYPES_ORDER:
            mask = celltypes == ct
            if mask.sum() == 0:
                continue
            ax.scatter(umap[mask, 0], umap[mask, 1],
                       c=CELLTYPE_COLORS[ct], s=1.5,
                       rasterized=True, linewidths=0, zorder=2)

        if col == 0:
            ax.set_ylabel("Tipo celular", fontsize=8, labelpad=4)
        ax.axis("off")

    # Celda vacía en fila 1 columna colorbar
    fig.add_subplot(gs[1, n_ds]).axis("off")

    # Leyenda de tipos celulares abajo centrada
    handles = [mpatches.Patch(color=CELLTYPE_COLORS[ct], label=ct)
               for ct in CELLTYPES_ORDER]
    fig.legend(handles=handles,
               title="Tipo celular", title_fontsize=8, fontsize=8,
               loc="lower center", ncol=len(CELLTYPES_ORDER),
               bbox_to_anchor=(0.5, -0.03),
               frameon=True, edgecolor="gray")

    plt.tight_layout()
    grupo_safe = safe_name(grupo)
    prefix_dir = f"{num:02d}_" if num else ""
    figures_dir_grupo = os.path.join(figures_dir, f"{prefix_dir}{grupo_safe}")
    os.makedirs(figures_dir_grupo, exist_ok=True)
    prefix = f"{num:02d}_" if num else ""
    out = os.path.join(figures_dir_grupo, f"{prefix}umap_score_celltypes_{grupo_safe}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"    [OK] Figura guardada: {out}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--genes_dir",   default="170_genes")
    parser.add_argument("--genes_file",  default="genes_filtrados_manual.xlsx")
    parser.add_argument("--figures_dir", default="figures/umap_firmas_celltypes")
    parser.add_argument("--percentil",   type=int, default=75)
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # Cargar Excel
    genes_path = os.path.join(args.genes_dir, args.genes_file)
    genes_df   = pd.read_excel(genes_path)
    gene_col   = "Gene"
    grupo_col  = "Clasificacion_final"
    grupos     = genes_df[grupo_col].dropna().unique()
    print(f"Grupos funcionales: {len(grupos)}")

    # Cargar datasets
    h5ad_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5ad")))
    print(f"Datasets encontrados: {len(h5ad_files)}")

    datasets_loaded = {}
    for h5ad_path in h5ad_files:
        ds_id = (os.path.basename(h5ad_path)
                 .replace(".h5ad", "").replace("adata_", "").replace("_raw", ""))
        adata = sc.read_h5ad(h5ad_path)
        adata = normalizar_var_names(adata)
        if "X_umap" not in adata.obsm:
            print(f"  [SKIP] {ds_id} — sin X_umap")
            continue
        if "celltype" not in adata.obs.columns:
            print(f"  [SKIP] {ds_id} — sin celltype")
            continue
        datasets_loaded[ds_id] = adata
        print(f"  Cargado {ds_id}: {adata.shape}, X_umap OK")

    if not datasets_loaded:
        print("[ERROR] Sin datasets válidos.")
        return

    for num, grupo in enumerate(grupos, 1):
        genes = genes_df[genes_df[grupo_col] == grupo][gene_col].dropna().tolist()
        print(f"\n{'='*55}")
        print(f"  Grupo {num}: {grupo} ({len(genes)} genes)")
        print(f"{'='*55}")
        figura_firma(grupo, genes, datasets_loaded, args.figures_dir, args.percentil, num)

    print(f"\n{'='*55}")
    print(f"  Completado. Figuras en: {args.figures_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
