"""
dotplot_firmas.py
-----------------
Para cada grupo funcional genera un panel con un dotplot por dataset +
uno de media global, todos con la misma escala de color para comparar.

  - Eje X: genes de la firma
  - Eje Y: tipos celulares
  - Tamaño del punto: % de células que expresan el gen (expresión > 0)
  - Color del punto: expresión media (log-norm)

Uso:
    python scripts/dotplot_firmas.py \\
        --data_dir data \\
        --genes_dir 170_genes \\
        --genes_file genes_filtrados_manual.xlsx \\
        --figures_dir figures/dotplots
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


CELLTYPES_ORDER = ["Fibroblast", "Endothelial", "Hepatocyte", "Myeloid", "T/NK", "B"]

CELLTYPE_COLORS = {
    "Fibroblast":  "#FFD54F",
    "Endothelial": "#81C784",
    "Hepatocyte":  "#FF6B6B",
    "Myeloid":     "#4FC3F7",
    "T/NK":        "#DA9EC4",
    "B":           "#FF8A65",
}

GENE_SYMBOL_CANDIDATES = ["feature_name", "gene_name", "gene_symbol", "Symbol", "symbol"]
CMAP_DOTPLOT = "RdBu_r"
MAX_SIZE     = 250


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


def get_expr_gen(adata, gen):
    if gen not in adata.var_names:
        return None
    x = adata[:, gen].X
    if scipy.sparse.issparse(x):
        x = x.toarray()
    return x.flatten()


def calcular_stats_dataset(adata, genes, celltypes_order):
    """
    Para un dataset devuelve:
      expr_df: DataFrame (genes x celltypes) con expresión media
      pct_df:  DataFrame (genes x celltypes) con % células expresando
    """
    expr_data = {ct: {} for ct in celltypes_order}
    pct_data  = {ct: {} for ct in celltypes_order}
    celltypes = adata.obs["celltype"].values

    for gen in genes:
        expr = get_expr_gen(adata, gen)
        for ct in celltypes_order:
            mask = celltypes == ct
            n_ct = mask.sum()
            if n_ct == 0 or expr is None:
                expr_data[ct][gen] = 0.0
                pct_data[ct][gen]  = 0.0
            else:
                vals = expr[mask]
                expr_data[ct][gen] = float(np.mean(vals))
                pct_data[ct][gen]  = float((vals > 0).sum() / n_ct * 100)

    expr_df = pd.DataFrame(expr_data).T  # celltypes x genes → transponer
    pct_df  = pd.DataFrame(pct_data).T
    # Ahora son celltypes x genes, transponer a genes x celltypes
    expr_df = expr_df.T
    pct_df  = pct_df.T
    return expr_df, pct_df


def calcular_stats_media(datasets_stats, genes, celltypes_order):
    """Promedia expr y pct entre todos los datasets."""
    expr_acum = {gen: {ct: [] for ct in celltypes_order} for gen in genes}
    pct_acum  = {gen: {ct: [] for ct in celltypes_order} for gen in genes}

    for ds_id, (expr_df, pct_df) in datasets_stats.items():
        for gen in genes:
            if gen not in expr_df.index:
                continue
            for ct in celltypes_order:
                if ct not in expr_df.columns:
                    continue
                expr_acum[gen][ct].append(expr_df.loc[gen, ct])
                pct_acum[gen][ct].append(pct_df.loc[gen, ct])

    expr_data, pct_data = {}, {}
    for gen in genes:
        expr_data[gen] = {ct: float(np.mean(v)) if v else 0.0
                          for ct, v in expr_acum[gen].items()}
        pct_data[gen]  = {ct: float(np.mean(v)) if v else 0.0
                          for ct, v in pct_acum[gen].items()}

    expr_df = pd.DataFrame(expr_data, index=celltypes_order).T
    pct_df  = pd.DataFrame(pct_data,  index=celltypes_order).T
    return expr_df, pct_df


# ──────────────────────────────────────────────────────────────
# DIBUJAR UN PANEL DE DOTPLOT
# ──────────────────────────────────────────────────────────────
def dibujar_panel(ax, genes_ok, cts_ok, expr_mat, pct_mat, norm, cmap, titulo):
    for i, ct in enumerate(cts_ok):
        for j, gen in enumerate(genes_ok):
            expr_val = expr_mat[j, i]
            pct_val  = pct_mat[j, i]
            color    = cmap(norm(expr_val))
            size     = (pct_val / 100) * MAX_SIZE
            ax.scatter(j, i, s=max(size, 1), color=color,
                       edgecolors="gray", linewidths=0.3, zorder=3)

    ax.set_xticks(range(len(genes_ok)))
    ax.set_xticklabels(genes_ok, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cts_ok)))
    ax.set_yticklabels(cts_ok, fontsize=8)
    for tick, ct in zip(ax.get_yticklabels(), cts_ok):
        tick.set_color(CELLTYPE_COLORS.get(ct, "black"))
        tick.set_fontweight("bold")
    ax.set_xlim(-0.5, len(genes_ok) - 0.5)
    ax.set_ylim(-0.5, len(cts_ok)   - 0.5)
    ax.grid(True, linewidth=0.3, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(titulo, fontsize=8, fontweight="bold", pad=6)


# ──────────────────────────────────────────────────────────────
# FIGURA COMPLETA POR GRUPO
# ──────────────────────────────────────────────────────────────
def generar_figura_grupo(grupo, genes, datasets_stats, media_stats, figures_dir):
    genes_ok = genes
    cts_ok   = CELLTYPES_ORDER

    # Recoger todos los valores para escala de color común
    all_expr = []
    for ds_id, (expr_df, _) in datasets_stats.items():
        for gen in genes_ok:
            if gen in expr_df.index:
                all_expr.extend(expr_df.loc[gen, :].values.tolist())
    expr_df_media, _ = media_stats
    for gen in genes_ok:
        if gen in expr_df_media.index:
            all_expr.extend(expr_df_media.loc[gen, :].values.tolist())

    all_expr = np.array(all_expr)
    vmin = float(np.percentile(all_expr, 2))
    vmax = float(np.percentile(all_expr, 98))
    if vmax <= vmin:
        vmax = vmin + 0.01
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP_DOTPLOT)

    # 3 arriba + 3 abajo (5 datasets + 1 media = 6 paneles)
    # Fila 0: ds[0], ds[1], ds[2]
    # Fila 1: ds[3], ds[4], Media
    NCOLS    = 3
    ds_list  = list(datasets_stats.keys())  # 5 datasets
    panel_w  = max(3, len(genes_ok) * 0.65 + 1.5)
    panel_h  = max(3, len(cts_ok) * 0.55 + 1.8)
    fig_w    = panel_w * NCOLS + 0.8
    fig_h    = panel_h * 2 + 0.5

    # gridspec: 2 filas x (NCOLS paneles + 1 colorbar)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(2, NCOLS + 1,
                           width_ratios=[panel_w] * NCOLS + [0.3],
                           hspace=0.35, wspace=0.08)

    # Crear ejes de dotplot: fila 0 cols 0-2, fila 1 cols 0-2
    axes = []
    for row in range(2):
        for col in range(NCOLS):
            axes.append(fig.add_subplot(gs[row, col]))

    # Colorbar ocupa toda la columna derecha (ambas filas)
    ax_cb = fig.add_subplot(gs[:, NCOLS])

    fig.suptitle(grupo, fontsize=11, fontweight="bold", y=1.02)

    # Asignar datasets a paneles: ds[0..4] + Media en posición 5
    all_panels = [(ds_id, datasets_stats[ds_id]) for ds_id in ds_list]
    all_panels.append(("Media datasets", media_stats))

    for ax, (titulo_panel, (expr_df, pct_df)) in zip(axes, all_panels):
        expr_mat = np.array([[expr_df.loc[g, ct] if g in expr_df.index and ct in expr_df.columns else 0.0
                              for ct in cts_ok] for g in genes_ok])
        pct_mat  = np.array([[pct_df.loc[g, ct]  if g in pct_df.index  and ct in pct_df.columns  else 0.0
                              for ct in cts_ok] for g in genes_ok])
        dibujar_panel(ax, genes_ok, cts_ok, expr_mat, pct_mat, norm, cmap, titulo_panel)

    # Colorbar en columna dedicada
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Expresión media (log-norm)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Leyenda de tamaño
    legend_pcts = [25, 50, 75, 100]
    handles = [plt.scatter([], [], s=(p/100)*MAX_SIZE, color="gray",
                           edgecolors="gray", linewidths=0.3, label=f"{p}%")
               for p in legend_pcts]
    fig.legend(handles=handles, title="% células\nexpresando",
               title_fontsize=7, fontsize=7,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03),
               frameon=True, edgecolor="gray")

    plt.tight_layout()
    grupo_safe = grupo.replace("/","_").replace(" ","_").replace("(","").replace(")","")
    out = os.path.join(figures_dir, f"dotplot_{grupo_safe}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"    [OK] Dotplot guardado: {out}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--genes_dir",   default="170_genes")
    parser.add_argument("--genes_file",  default="genes_filtrados_manual.xlsx")
    parser.add_argument("--figures_dir", default="figures/dotplots")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    genes_path = os.path.join(args.genes_dir, args.genes_file)
    genes_df   = pd.read_excel(genes_path)
    gene_col   = "Gene"
    grupo_col  = "Clasificacion_final"
    grupos     = genes_df[grupo_col].dropna().unique()
    print(f"Grupos funcionales: {len(grupos)}")

    h5ad_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5ad")))
    print(f"Datasets encontrados: {len(h5ad_files)}")

    datasets_loaded = {}
    for h5ad_path in h5ad_files:
        ds_id = (os.path.basename(h5ad_path)
                 .replace(".h5ad","").replace("adata_","").replace("_raw",""))
        adata = sc.read_h5ad(h5ad_path)
        adata = normalizar_var_names(adata)
        if "celltype" not in adata.obs.columns:
            print(f"  [SKIP] {ds_id} — sin celltype")
            continue
        datasets_loaded[ds_id] = adata
        print(f"  Cargado {ds_id}: {adata.shape}")

    if not datasets_loaded:
        print("[ERROR] Sin datasets válidos.")
        return

    for grupo in grupos:
        genes = genes_df[genes_df[grupo_col]==grupo][gene_col].dropna().tolist()
        print(f"\n{'='*55}")
        print(f"  Grupo: {grupo} ({len(genes)} genes)")
        print(f"{'='*55}")

        # Calcular stats por dataset
        datasets_stats = {}
        for ds_id, adata in datasets_loaded.items():
            expr_df, pct_df = calcular_stats_dataset(adata, genes, CELLTYPES_ORDER)
            datasets_stats[ds_id] = (expr_df, pct_df)

        # Calcular media
        media_stats = calcular_stats_media(datasets_stats, genes, CELLTYPES_ORDER)

        generar_figura_grupo(grupo, genes, datasets_stats, media_stats, args.figures_dir)

    print(f"\n{'='*55}")
    print(f"  Completado. Dotplots en: {args.figures_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
