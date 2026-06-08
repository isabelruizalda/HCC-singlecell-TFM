"""
heatmap_comparativo.py
----------------------
Genera un heatmap comparativo de expresión media (z-score) por tipo celular
y dataset para cada grupo funcional de genes.

Por cada grupo funcional produce UNA figura con:
  - Filas:    genes del grupo presentes en al menos un dataset
  - Columnas: tipo celular × dataset (en el orden definido en DATASETS_ORDER)
  - Color:    expresión media z-score (rojo = alta, azul = baja)

Uso:
    python scripts/heatmap_comparativo.py \
        --data_dir data \
        --genes_dir 170_genes \
        --figures_dir figures \
        --genes_file 170genes_clasificacion_FINAL.xlsx
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────

# Datasets: se descubren automáticamente desde data_dir (como en annotate/analyse)
# No hay que cambiar nada al añadir un dataset nuevo.
DATASETS_ORDER   = []   # se rellena en main() con glob
DATASETS_LABELS  = {}   # se rellena en main()
DATASET_COLORS   = {}   # se rellena en main()

# Paleta de colores para asignar a datasets automáticamente
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

# Orden fijo de tipos celulares
CELLTYPES_ORDER = [
    "Fibroblast",
    "Endothelial",
    "Hepatocyte",
    "Myeloid",
    "T/NK",
    "B",
]

# Reagrupación funcional (igual que en analyze_datasets.py)
REAGRUPACION_GENERAL = {
    "ECM / Fibrosis / Colágeno":                 "ECM / Estroma / Remodelacion",
    "Proteolisis / remodelacion proteica":        "ECM / Estroma / Remodelacion",
    "Adhesion / ECM / migracion":                 "ECM / Estroma / Remodelacion",
    "Glicosilacion / proteoglicanos / secrecion": "ECM / Estroma / Remodelacion",

    "Chemotaxis":                                 "Inmunidad / Inflamacion / Chemotaxis",
    "Inmunidad / inflamacion":                    "Inmunidad / Inflamacion / Chemotaxis",
    "Complemento":                                "Inmunidad / Inflamacion / Chemotaxis",

    "Angiogénesis / Vascular":                    "Angiogenesis / Vascular / Hipoxia",
    "Hipoxia / estres / apoptosis":               "Angiogenesis / Vascular / Hipoxia",

    "Señalizacion celular":                       "Señalizacion / Regulacion celular",
    "Desarrollo / regulacion":                    "Señalizacion / Regulacion celular",
    "Transporte / canales / trafico":             "Señalizacion / Regulacion celular",

    "Neuronal / sinapsis":                        "Otros / poco clara",
    "Otros / funcion poco clara":                 "Otros / poco clara",
}




# ──────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────
def safe_name(text: str) -> str:
    replacements = {
        " ": "_", "/": "_",
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def dataset_id_from_path(path: str) -> str:
    basename = os.path.basename(path)
    name = basename.replace(".h5ad", "").replace("adata_", "").replace("_raw", "")
    return name


def normalizar_var_names(adata, dataset_name: str):
    """Convierte Ensembl IDs a símbolos de gen si es necesario."""
    GENE_SYMBOL_CANDIDATES = ["feature_name", "gene_name", "gene_symbol", "Symbol", "symbol"]
    if not adata.var_names[0].startswith("ENSG"):
        return adata
    for col in GENE_SYMBOL_CANDIDATES:
        if col in adata.var.columns:
            adata.var_names = adata.var[col].astype(str).values
            adata.var_names_make_unique()
            adata.raw = None
            print(f"    [{dataset_name}] var_names reindexados con '{col}'")
            return adata
    print(f"    [AVISO] No se encontró columna de símbolos para {dataset_name}")
    return adata


# ──────────────────────────────────────────────────────────────
# CALCULAR EXPRESIÓN MEDIA POR TIPO CELULAR
# ──────────────────────────────────────────────────────────────
def calcular_media_por_celltype(adata, genes, celltype_col):
    """
    Devuelve un DataFrame (genes × celltypes) con la expresión media
    de cada gen en cada tipo celular.
    """
    genes_presentes = [g for g in genes if g in adata.var_names]
    if not genes_presentes:
        return pd.DataFrame()

    adata_sub = adata[:, genes_presentes]
    X = adata_sub.X
    if scipy.sparse.issparse(X):
        X = X.toarray()

    df = pd.DataFrame(X, index=adata.obs[celltype_col].values, columns=genes_presentes)
    mean_df = df.groupby(level=0).mean()  # celltypes × genes
    return mean_df.T  # genes × celltypes


# ──────────────────────────────────────────────────────────────
# CONSTRUIR MATRIZ COMPLETA PARA EL HEATMAP
# ──────────────────────────────────────────────────────────────
def construir_matriz_heatmap(datasets_data, genes_grupo):
    """
    Construye la matriz final genes × (celltype_dataset) con z-score por gen.

    datasets_data: dict {dataset_id: (adata, celltype_col)}
    genes_grupo:   lista de genes del grupo funcional

    Devuelve:
        matrix_df:   DataFrame con z-scores
        col_info:    lista de (celltype, dataset_id) para cada columna
    """
    # Recoger expresión media de cada dataset
    todas_medias = {}
    for ds_id in DATASETS_ORDER:
        if ds_id not in datasets_data:
            continue
        adata, celltype_col = datasets_data[ds_id]
        media = calcular_media_por_celltype(adata, genes_grupo, celltype_col)
        if media.empty:
            continue
        todas_medias[ds_id] = media

    if not todas_medias:
        return pd.DataFrame(), []

    # Genes presentes en al menos un dataset
    todos_genes = sorted(
        set(g for m in todas_medias.values() for g in m.index),
        key=lambda g: genes_grupo.index(g) if g in genes_grupo else 9999
    )

    # Construir columnas: celltype × dataset
    col_info = []   # lista de (celltype, dataset_id)
    columnas = {}   # clave → serie de valores por gen

    for ct in CELLTYPES_ORDER:
        for ds_id in DATASETS_ORDER:
            if ds_id not in todas_medias:
                continue
            media = todas_medias[ds_id]
            col_key = f"{ct}|{ds_id}"
            if ct in media.columns:
                columnas[col_key] = media[ct].reindex(todos_genes)
            else:
                columnas[col_key] = pd.Series(np.nan, index=todos_genes)
            col_info.append((ct, ds_id))

    matrix_df = pd.DataFrame(columnas, index=todos_genes)

    # Z-score por gen (normalizar cada fila)
    def zscore_row(row):
        std = row.std()
        if std == 0 or np.isnan(std):
            return row - row.mean()
        return (row - row.mean()) / std

    matrix_zscored = matrix_df.apply(zscore_row, axis=1)
    return matrix_zscored, col_info


# ──────────────────────────────────────────────────────────────
# PALETAS DE COLOR (estética de referencia)
# ──────────────────────────────────────────────────────────────
DATASET_COLORS_VIVID = {
    "GSE149614": "#2c5fa8",
    "GSE166635": "#c45c00",
    "GSE125449": "#1e7a30",
    "ICB_HCC":   "#b01010",
}

# Uno por grupo funcional (se reciclan si hay más de 4)
GRUPO_TITLE_COLORS = [
    "#1a6fa8",   # azul
    "#c0392b",   # rojo
    "#27ae60",   # verde
    "#8e44ad",   # morado
]

# Texto de los recuadros de pie de página (compartido entre figuras)
_LEGEND_TEXT = (
    "Leyenda:\n"
    "  • Cada columna corresponde a un dataset dentro de cada tipo celular.\n"
    "  • El color representa la expresión media (z-score) en ese tipo celular y dataset.\n"
    "  • Rojo: mayor expresión;  Azul: menor expresión."
)
_DATASETS_TEXT = (
    "Datasets:\n"
    "  • GSE149614: con anotación de tipos celulares (referencia)\n"
    "  • GSE125449, GSE166635: sin anotación (anotadas por label transfer)\n"
    "  • ICB_HCC: dataset propio de HCC (anotado)"
)
_BBOX_STYLE = dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                   edgecolor="#aaaaaa", linewidth=0.8)


# ──────────────────────────────────────────────────────────────
# NÚCLEO: pintar un subheatmap en ejes ya creados
# ──────────────────────────────────────────────────────────────
def _pintar_subheatmap(fig, ax, cax, matrix_df, col_info,
                       titulo, num_grupo,
                       fs_genes=8, fs_ds=8, fs_ct=9, fs_cbar=8):
    """
    Pinta el heatmap en `ax` y la colorbar en `cax`.
    fs_* controlan los tamaños de fuente para poder escalar entre
    figura individual (más grande) y multipanel (más compacto).
    """
    n_genes    = len(matrix_df)
    n_cols     = len(col_info)
    n_datasets = len({ds for _, ds in col_info})

    # ── Colormap centrado en 0 ──
    vmax = np.nanpercentile(np.abs(matrix_df.values), 95)
    vmax = max(vmax, 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        matrix_df.values,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )

    # ── Eje Y: genes ──
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(matrix_df.index, fontsize=fs_genes)
    ax.set_ylabel("Genes", fontsize=fs_genes + 0.5)
    ax.tick_params(axis="y", length=0)

    # ── Eje X: etiquetas de dataset en color ──
    labels_x = [DATASETS_LABELS.get(ds, ds) for _, ds in col_info]
    colors_x = [DATASET_COLORS_VIVID.get(ds, "black") for _, ds in col_info]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels_x, rotation=90, ha="center", va="top",
                       fontsize=fs_ds, fontweight="bold")
    for tick, color in zip(ax.get_xticklabels(), colors_x):
        tick.set_color(color)
    ax.tick_params(axis="x", length=0)

    # ── Líneas blancas entre tipos celulares ──
    celltypes_en_datos = [ct for ct in CELLTYPES_ORDER
                          if any(ct == c for c, _ in col_info)]
    for i in range(1, len(celltypes_en_datos)):
        ax.axvline(x=i * n_datasets - 0.5, color="white", linewidth=2)

    # ── Eje superior: nombres de tipo celular + etiqueta "Tipo celular" ──
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ct_centers = [i * n_datasets + (n_datasets - 1) / 2
                  for i in range(len(celltypes_en_datos))]
    ax_top.set_xticks(ct_centers)
    ax_top.set_xticklabels(celltypes_en_datos, fontsize=fs_ct, fontweight="bold")
    ax_top.tick_params(length=0)
    ax_top.set_xlabel("Tipo celular", fontsize=fs_ct - 1, labelpad=5)

    # ── Título numerado con color ──
    color_t = GRUPO_TITLE_COLORS[(num_grupo - 1) % len(GRUPO_TITLE_COLORS)]
    ax.set_title(
        f"{num_grupo}. {titulo}",
        fontsize=fs_ct, fontweight="bold", color=color_t,
        loc="left", pad=30,
    )

    # ── Colorbar ──
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Expresión media\n(z-score)", fontsize=fs_cbar)
    cbar.ax.tick_params(labelsize=fs_cbar - 1)

    return im


# ──────────────────────────────────────────────────────────────
# FIGURA INDIVIDUAL POR GRUPO
# ──────────────────────────────────────────────────────────────
def dibujar_heatmap_individual(matrix_df, col_info, grupo, num_grupo, figures_dir):
    """Una figura por grupo funcional, con recuadros de leyenda al pie."""
    n_genes = len(matrix_df)
    n_cols  = len(col_info)

    if n_genes == 0 or n_cols == 0:
        print(f"    [SKIP] Sin datos para '{grupo}'")
        return

    # Dimensiones adaptativas
    fig_w = max(14, n_cols * 0.42 + 4.5)
    fig_h = max(7,  n_genes * 0.33 + 5.5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # GridSpec: heatmap | spacer | colorbar
    # Espacio generoso a la derecha para que la colorbar no solape nunca el heatmap
    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[n_cols * 0.42, 0.35, 0.30],
        wspace=0.08,
        left=0.08, right=0.92,
        top=0.88, bottom=0.22,
    )
    ax  = fig.add_subplot(gs[0])
    # gs[1] actúa como separador vacío; la colorbar va en gs[2]
    cax = fig.add_subplot(gs[2])

    _pintar_subheatmap(fig, ax, cax, matrix_df, col_info,
                       titulo=grupo, num_grupo=num_grupo,
                       fs_genes=8.5, fs_ds=8.5, fs_ct=10, fs_cbar=9)

    # ── Título principal de la figura ──
    fig.suptitle(
        "Expresión media por tipo celular y dataset",
        fontsize=12, fontweight="bold", y=0.97,
    )

    # ── Recuadros de pie de página ──
    fig.text(0.04, 0.01, _LEGEND_TEXT,
             fontsize=7.5, va="bottom", ha="left", bbox=_BBOX_STYLE)
    # Colorbar está ahora más a la derecha; desplazar segunda caja de texto
    fig.text(0.52, 0.01, _DATASETS_TEXT,
             fontsize=7.5, va="bottom", ha="left", bbox=_BBOX_STYLE)

    # Guardar
    os.makedirs(figures_dir, exist_ok=True)
    fname = f"heatmap_{num_grupo:02d}_{safe_name(grupo)}.png"
    out_path = os.path.join(figures_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"    [OK] Individual: {out_path}")


# ──────────────────────────────────────────────────────────────
# FIGURA MULTIPANEL (todos los grupos en rejilla 2×N)
# ──────────────────────────────────────────────────────────────
def dibujar_figura_multipanel(grupos_data, figures_dir):
    """
    grupos_data: lista de (grupo_str, matrix_df, col_info).
    Genera UNA figura con todos los grupos en layout 2 columnas.
    """
    import math

    n_grupos = len(grupos_data)
    if n_grupos == 0:
        print("[SKIP] Sin grupos con datos.")
        return

    n_cols_fig = 2
    n_rows_fig = math.ceil(n_grupos / n_cols_fig)

    # Dimensiones basadas en el grupo más grande
    max_genes = max(len(m) for _, m, _ in grupos_data)
    max_xcols = max(len(ci) for _, _, ci in grupos_data)

    sub_w  = max(6.5, max_xcols * 0.36 + 1.2)
    sub_h  = max(3.5, max_genes * 0.28 + 2.2)
    cbar_w = 0.32   # ligeramente más ancha para legibilidad
    gap_w  = 0.20   # separador explícito entre heatmap y colorbar

    fig_w = n_cols_fig * (sub_w + gap_w + cbar_w + 0.7) + 0.6
    fig_h = n_rows_fig * (sub_h + 1.4) + 2.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Expresión media de grupos funcionales de genes por tipo celular y dataset",
        fontsize=11, fontweight="bold", y=0.995,
    )

    # GridSpec: cada panel ocupa TRES columnas (heatmap | gap | colorbar)
    panel_w = []
    for _ in range(n_cols_fig):
        panel_w += [sub_w, gap_w, cbar_w]

    outer_gs = gridspec.GridSpec(
        n_rows_fig, n_cols_fig * 3,
        figure=fig,
        width_ratios=panel_w,
        wspace=0.06,
        hspace=0.60,
        left=0.06, right=0.97,
        top=0.93, bottom=0.15,
    )

    for idx, (grupo, matrix_df, col_info) in enumerate(grupos_data):
        row = idx // n_cols_fig
        col = idx %  n_cols_fig
        ax  = fig.add_subplot(outer_gs[row, col * 3])
        # col*3+1 es el gap vacío; colorbar en col*3+2
        cax = fig.add_subplot(outer_gs[row, col * 3 + 2])
        _pintar_subheatmap(fig, ax, cax, matrix_df, col_info,
                           titulo=grupo, num_grupo=idx + 1,
                           fs_genes=7, fs_ds=7, fs_ct=8, fs_cbar=7.5)

    # ── Recuadros de pie de página ──
    fig.text(0.03, 0.005, _LEGEND_TEXT,
             fontsize=7, va="bottom", ha="left", bbox=_BBOX_STYLE)
    fig.text(0.52, 0.005, _DATASETS_TEXT,
             fontsize=7, va="bottom", ha="left", bbox=_BBOX_STYLE)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "heatmap_comparativo_multipanel.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"    [OK] Multipanel: {out_path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Heatmaps comparativos multi-dataset por grupo funcional."
    )
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--genes_dir",   default="170_genes")
    parser.add_argument("--figures_dir", default="figures/heatmaps_comparativos")
    parser.add_argument("--genes_file",  default="170genes_clasificacion_FINAL.xlsx")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Cargar Excel de genes ──
    genes_path = os.path.join(args.genes_dir, args.genes_file)
    print(f"Cargando Excel de genes: {genes_path}")
    genes_df = pd.read_excel(genes_path)
    gene_col = genes_df.columns[0]
    func_col = genes_df.columns[1]
    # Usar Clasificacion_final si existe (Excel nuevo con puntuacion ponderada)
    if "Clasificacion_final" in genes_df.columns:
        genes_df["grupo_general"] = genes_df["Clasificacion_final"]
        print("  Usando columna 'Clasificacion_final' (GO+Reactome+MSigDB+KEGG)")
    else:
        genes_df["grupo_general"] = genes_df[func_col].map(REAGRUPACION_GENERAL)
        print("  Usando reagrupacion manual como fallback")

    grupos = genes_df["grupo_general"].dropna().unique()
    print(f"Grupos funcionales: {list(grupos)}")

    # ── Autodescubrimiento de datasets (igual que annotate/analyse) ──
    import glob as _glob
    h5ad_files = sorted(_glob.glob(os.path.join(args.data_dir, "*.h5ad")))
    if not h5ad_files:
        print(f"[ERROR] No se encontraron ficheros .h5ad en '{args.data_dir}'.")
        return

    print(f"\nDatasets encontrados: {len(h5ad_files)}")
    for f in h5ad_files:
        print(f"  - {os.path.basename(f)}")

    # Rellenar DATASETS_ORDER, LABELS y COLORS dinámicamente
    for i, f in enumerate(h5ad_files):
        ds_id = os.path.basename(f).replace(".h5ad","").replace("adata_","").replace("_raw","")
        DATASETS_ORDER.append(ds_id)
        DATASETS_LABELS[ds_id] = ds_id
        DATASET_COLORS[ds_id]  = _PALETTE[i % len(_PALETTE)]

    # ── Cargar datasets ──
    print("\nCargando datasets...")
    datasets_data = {}

    for ds_id, h5ad_path in zip(DATASETS_ORDER, h5ad_files):
        print(f"  Cargando {ds_id} desde {h5ad_path}...")
        adata = sc.read_h5ad(h5ad_path)
        adata = normalizar_var_names(adata, ds_id)

        if "celltype" not in adata.obs.columns:
            print(f"  [SKIP] Sin columna 'celltype' en {ds_id}")
            continue

        print(f"  {ds_id}: {adata.shape}, {adata.obs['celltype'].nunique()} tipos celulares")
        datasets_data[ds_id] = (adata, "celltype")

    if not datasets_data:
        print("[ERROR] No se cargó ningún dataset con columna 'celltype'.")
        return

    # ── Generar heatmaps: uno por grupo + uno multipanel ──
    print(f"\nGenerando heatmaps en: {args.figures_dir}")
    grupos_data = []   # acumular para el multipanel

    for num_grupo, grupo in enumerate(grupos, start=1):
        print(f"\n  Grupo {num_grupo}: {grupo}")

        genes_grupo = (
            genes_df.loc[genes_df["grupo_general"] == grupo, gene_col]
            .dropna().astype(str).tolist()
        )
        print(f"  Genes en el grupo: {len(genes_grupo)}")

        matrix_df, col_info = construir_matriz_heatmap(datasets_data, genes_grupo)

        if matrix_df.empty:
            print(f"  [SKIP] Sin datos para '{grupo}'")
            continue

        print(f"  Matriz antes de filtrar: {matrix_df.shape[0]} genes × {matrix_df.shape[1]} columnas")

        # ── FILTRO z-score MODERADO ─────────────────────────────
        # Mantener genes con z-score >= 0.5 en al menos la MITAD
        # de los datasets presentes.
        datasets_presentes = list(dict.fromkeys(ds for _, ds in col_info))
        n_datasets = len(datasets_presentes)
        min_datasets = max(1, n_datasets // 2)  # al menos la mitad
        genes_a_mantener = []
        for gen in matrix_df.index:
            datasets_que_pasan = 0
            for ds in datasets_presentes:
                cols_ds = [f"{ct}|{ds}" for ct, ds2 in col_info if ds2 == ds]
                cols_ds = [c for c in cols_ds if c in matrix_df.columns]
                if not cols_ds:
                    continue
                max_zscore = matrix_df.loc[gen, cols_ds].max()
                if max_zscore >= 0.5:
                    datasets_que_pasan += 1
            if datasets_que_pasan >= min_datasets:
                genes_a_mantener.append(gen)

        n_antes = len(matrix_df)
        matrix_df = matrix_df.loc[genes_a_mantener]
        print(f"  Genes tras filtro z-score >= 0.5 en al menos {min_datasets}/{n_datasets} datasets: {len(matrix_df)}/{n_antes}")

        if matrix_df.empty:
            print(f"  [SKIP] Sin genes tras el filtro para '{grupo}'")
            continue
        # ────────────────────────────────────────────────────────

        print(f"  Matriz final: {matrix_df.shape[0]} genes × {matrix_df.shape[1]} columnas")

        # Figura individual
        dibujar_heatmap_individual(matrix_df, col_info, grupo, num_grupo, args.figures_dir)

        # Acumular para el multipanel
        grupos_data.append((grupo, matrix_df, col_info))

    # Figura multipanel con todos los grupos
    dibujar_figura_multipanel(grupos_data, args.figures_dir)

    print(f"\n{'='*60}")
    print("  Heatmaps comparativos completados.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
