"""
umap_firmas.py
--------------
Para cada grupo funcional, genera figuras con UMAPs coloreados por
expresión génica, usando las coordenadas X_umap guardadas en cada h5ad
por annotate_datasets_v2.py.

- NO renormaliza los datos — usa adata.X tal cual está guardado por annotate
- Colorbar siempre a la derecha del último panel, nunca encima de un UMAP

Uso:
    python scripts/umap_firmas.py \\
        --data_dir data \\
        --genes_dir 170_genes \\
        --genes_file genes_filtrados_manual.xlsx \\
        --figures_dir figures/umap_firmas \\
        --percentil 75
"""

import os
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
warnings.filterwarnings("ignore")

GENE_SYMBOL_CANDIDATES = ["feature_name", "gene_name", "gene_symbol", "Symbol", "symbol"]
CMAP_EXPR  = "Reds"
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


def get_expr(adata, gen):
    """Lee expresión directamente de adata.X sin tocar nada."""
    if gen not in adata.var_names:
        return np.zeros(adata.n_obs)
    x = adata[:, gen].X
    if scipy.sparse.issparse(x):
        x = x.toarray()
    return x.flatten()


def calcular_score(adata, genes, score_key):
    """
    Calcula gene score sobre adata.X sin renormalizar.
    annotate ya dejó los datos en log-norm.
    """
    adata_pp = adata.copy()
    genes_presentes = [g for g in genes if g in adata_pp.var_names]
    if not genes_presentes:
        return np.zeros(adata.n_obs)
    sc.tl.score_genes(adata_pp, gene_list=genes_presentes, score_name=score_key)
    return adata_pp.obs[score_key].values


def safe_name(s):
    return s.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


# ──────────────────────────────────────────────────────────────
# FIGURA POR GEN
# ──────────────────────────────────────────────────────────────
def figura_gen(gen, grupo, datasets_loaded, figures_dir_grupo):
    ds_names  = list(datasets_loaded.keys())
    n_ds      = len(ds_names)
    n_panels  = n_ds + 1  # +1 para el conjunto

    # Escala común entre todos los datasets
    all_expr = [get_expr(adata, gen) for adata in datasets_loaded.values()]
    all_concat = np.concatenate(all_expr)
    vmin = float(np.percentile(all_concat, 2))
    vmax = float(np.percentile(all_concat, 98))
    if vmax <= vmin:
        vmax = vmin + 0.01
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Layout: n_panels UMAPs + 1 columna estrecha para colorbar
    fig = plt.figure(figsize=(4 * n_panels + 0.8, 4))
    gs  = fig.add_gridspec(1, n_panels + 1,
                           width_ratios=[4] * n_panels + [0.25],
                           wspace=0.05)

    fig.suptitle(f"{grupo}\nGen: {gen}", fontsize=11, fontweight="bold", y=1.02)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    ax_cb = fig.add_subplot(gs[0, n_panels])

    # Panel por dataset
    for ax, (ds_id, adata) in zip(axes[:n_ds], datasets_loaded.items()):
        if "X_umap" not in adata.obsm:
            ax.text(0.5, 0.5, f"Sin UMAP\n{ds_id}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            continue

        umap = adata.obsm["X_umap"]
        expr = get_expr(adata, gen)
        mask_cero = expr <= vmin

        ax.scatter(umap[mask_cero, 0], umap[mask_cero, 1],
                   c="#EEEEEE", s=1.5, rasterized=True, linewidths=0)
        if (~mask_cero).sum() > 0:
            ax.scatter(umap[~mask_cero, 0], umap[~mask_cero, 1],
                       c=expr[~mask_cero], cmap=CMAP_EXPR,
                       norm=norm, s=2, rasterized=True, linewidths=0)
        ax.set_title(ds_id, fontsize=8, fontweight="bold")
        ax.axis("off")

    # Panel conjunto
    ax_conj = axes[n_ds]
    umap_all, expr_all = [], []
    for adata in datasets_loaded.values():
        if "X_umap" in adata.obsm:
            umap_all.append(adata.obsm["X_umap"])
            expr_all.append(get_expr(adata, gen))

    if umap_all:
        umap_cat = np.concatenate(umap_all)
        expr_cat = np.concatenate(expr_all)
        mask_cero = expr_cat <= vmin
        ax_conj.scatter(umap_cat[mask_cero, 0], umap_cat[mask_cero, 1],
                        c="#EEEEEE", s=0.8, rasterized=True, linewidths=0)
        if (~mask_cero).sum() > 0:
            ax_conj.scatter(umap_cat[~mask_cero, 0], umap_cat[~mask_cero, 1],
                            c=expr_cat[~mask_cero], cmap=CMAP_EXPR,
                            norm=norm, s=1, rasterized=True, linewidths=0)
    ax_conj.set_title("Todos los datasets", fontsize=8, fontweight="bold")
    ax_conj.axis("off")

    # Colorbar en columna dedicada — nunca tapa ningún UMAP
    sm = ScalarMappable(cmap=CMAP_EXPR, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Expresión (log-norm)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    out = os.path.join(figures_dir_grupo, f"umap_{safe_name(gen)}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")


# ──────────────────────────────────────────────────────────────
# FIGURA SCORE CONJUNTO
# ──────────────────────────────────────────────────────────────
def figura_score_conjunto(grupo, genes, datasets_loaded, figures_dir_grupo, percentil):
    ds_names = list(datasets_loaded.keys())
    n_ds     = len(ds_names)
    n_panels = n_ds + 1
    score_key = f"score_{safe_name(grupo)[:20]}"

    # Calcular scores — sin renormalizar
    scores_por_ds = {}
    for ds_id, adata in datasets_loaded.items():
        scores_por_ds[ds_id] = calcular_score(adata, genes, score_key)

    all_scores = np.concatenate(list(scores_por_ds.values()))
    vmin = float(np.percentile(all_scores, 2))
    vmax = float(np.percentile(all_scores, 98))
    if vmax <= vmin:
        vmax = vmin + 0.01
    norm    = Normalize(vmin=vmin, vmax=vmax)
    umbral  = np.percentile(all_scores, percentil)

    fig = plt.figure(figsize=(4 * n_panels + 0.8, 4))
    gs  = fig.add_gridspec(1, n_panels + 1,
                           width_ratios=[4] * n_panels + [0.25],
                           wspace=0.05)
    fig.suptitle(f"{grupo}\nScore conjunto ({len(genes)} genes) — umbral percentil {percentil}",
                 fontsize=10, fontweight="bold", y=1.02)

    axes  = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    ax_cb = fig.add_subplot(gs[0, n_panels])

    # Panel por dataset
    for ax, (ds_id, adata) in zip(axes[:n_ds], datasets_loaded.items()):
        if "X_umap" not in adata.obsm:
            ax.axis("off")
            continue
        umap  = adata.obsm["X_umap"]
        score = scores_por_ds[ds_id]
        mask_bajo = score < umbral

        ax.scatter(umap[mask_bajo, 0], umap[mask_bajo, 1],
                   c="#EEEEEE", s=1.5, rasterized=True, linewidths=0)
        if (~mask_bajo).sum() > 0:
            ax.scatter(umap[~mask_bajo, 0], umap[~mask_bajo, 1],
                       c=score[~mask_bajo], cmap=CMAP_SCORE,
                       norm=norm, s=2, rasterized=True, linewidths=0)
        ax.set_title(ds_id, fontsize=8, fontweight="bold")
        ax.axis("off")

    # Panel conjunto
    ax_conj = axes[n_ds]
    umap_all, score_all = [], []
    for ds_id, adata in datasets_loaded.items():
        if "X_umap" in adata.obsm:
            umap_all.append(adata.obsm["X_umap"])
            score_all.append(scores_por_ds[ds_id])

    if umap_all:
        umap_cat  = np.concatenate(umap_all)
        score_cat = np.concatenate(score_all)
        mask_bajo = score_cat < umbral
        ax_conj.scatter(umap_cat[mask_bajo, 0], umap_cat[mask_bajo, 1],
                        c="#EEEEEE", s=0.8, rasterized=True, linewidths=0)
        if (~mask_bajo).sum() > 0:
            ax_conj.scatter(umap_cat[~mask_bajo, 0], umap_cat[~mask_bajo, 1],
                            c=score_cat[~mask_bajo], cmap=CMAP_SCORE,
                            norm=norm, s=1, rasterized=True, linewidths=0)
    ax_conj.set_title("Todos los datasets", fontsize=8, fontweight="bold")
    ax_conj.axis("off")

    # Colorbar en columna dedicada
    sm = ScalarMappable(cmap=CMAP_SCORE, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label(f"Score firma", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    out = os.path.join(figures_dir_grupo, "umap_score_conjunto.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"    [OK] Score conjunto guardado")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--genes_dir",   default="170_genes")
    parser.add_argument("--genes_file",  default="genes_filtrados_manual.xlsx")
    parser.add_argument("--figures_dir", default="figures/umap_firmas")
    parser.add_argument("--percentil",   type=int, default=75)
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # Cargar Excel de genes
    genes_path = os.path.join(args.genes_dir, args.genes_file)
    genes_df   = pd.read_excel(genes_path)
    gene_col   = "Gene"
    grupo_col  = "Clasificacion_final"
    grupos     = genes_df[grupo_col].dropna().unique()
    print(f"Grupos funcionales: {len(grupos)}")

    # Cargar datasets — NO renormalizar, usar X_umap guardado por annotate
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
            print(f"  [SKIP] {ds_id} — sin X_umap, lanza annotate primero")
            continue
        datasets_loaded[ds_id] = adata
        print(f"  Cargado {ds_id}: {adata.shape}, X_umap OK")

    if not datasets_loaded:
        print("[ERROR] Ningún dataset tiene X_umap.")
        return

    # Procesar cada grupo
    for grupo in grupos:
        genes      = genes_df[genes_df[grupo_col] == grupo][gene_col].dropna().tolist()
        grupo_safe = safe_name(grupo)

        print(f"\n{'='*55}")
        print(f"  Grupo: {grupo} ({len(genes)} genes)")
        print(f"{'='*55}")

        figures_dir_grupo = os.path.join(args.figures_dir, grupo_safe)
        os.makedirs(figures_dir_grupo, exist_ok=True)

        for gen in genes:
            print(f"    Gen: {gen}")
            figura_gen(gen, grupo, datasets_loaded, figures_dir_grupo)

        print(f"    Score conjunto...")
        figura_score_conjunto(grupo, genes, datasets_loaded,
                              figures_dir_grupo, args.percentil)

    print(f"\n{'='*55}")
    print(f"  Completado. Figuras en: {args.figures_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
