"""
score_genes.py
--------------
Calcula el gene score de cada grupo funcional en cada dataset
y genera UMAPs individuales + UMAP integrado (Harmony) por grupo.

Por cada grupo funcional produce UNA figura con:
  - Fila superior: 4 UMAPs individuales (uno por dataset)
  - Panel inferior: UMAP integrado con Harmony de todos los datasets
  - Color: gene score (sc.tl.score_genes)

Uso:
    python scripts/score_genes.py \
        --data_dir data \
        --genes_dir 170_genes \
        --figures_dir figures/score \
        --genes_file 170genes_clasificacion_FINAL.xlsx
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────
# Datasets: se descubren automáticamente desde data_dir
DATASETS_ORDER  = []
DATASETS_LABELS = {}
DATASET_COLORS  = {}

_PALETTE = [
    "#2c5fa8", "#c45c00", "#1e7a30", "#b01010",
    "#7a1e7a", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

# Colores para el título de cada grupo funcional
GRUPO_COLORS = [
    "#1a6fa8", "#b05a00", "#7a1e7a", "#1e7a30", "#7a1e1e"
]


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


def normalizar_var_names(adata, dataset_name: str):
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
# PREPROCESAMIENTO Y UMAP INDIVIDUAL
# ──────────────────────────────────────────────────────────────
def preparar_umap(adata, genes_score, score_key):
    """
    Normaliza, calcula HVGs, PCA, vecinos, UMAP y gene score.
    Trabaja sobre una copia para no modificar el original.
    """
    import scipy.sparse

    adata_pp = adata.copy()

    # Normalización (solo si no está ya log-transformado)
    sc.pp.normalize_total(adata_pp, target_sum=1e4)
    sc.pp.log1p(adata_pp)

    # HVGs y PCA
    sc.pp.highly_variable_genes(adata_pp, n_top_genes=2000)
    sc.tl.pca(adata_pp, use_highly_variable=True)

    # Reutilizar UMAP del annotate si ya existe en el h5ad original
    if "X_umap" in adata.obsm:
        print(f"    Reutilizando UMAP calculado por annotate_datasets.py")
        adata_pp.obsm["X_umap"] = adata.obsm["X_umap"].copy()
    else:
        print(f"    UMAP no encontrado en h5ad, calculando desde cero...")
        sc.pp.neighbors(adata_pp, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata_pp)

    # Gene score
    genes_presentes = [g for g in genes_score if g in adata_pp.var_names]
    if len(genes_presentes) == 0:
        print(f"    [AVISO] Sin genes del grupo en este dataset")
        adata_pp.obs[score_key] = 0.0
    else:
        sc.tl.score_genes(adata_pp, gene_list=genes_presentes, score_name=score_key)

    return adata_pp, len(genes_presentes)


# ──────────────────────────────────────────────────────────────
# INTEGRACIÓN CON HARMONY
# ──────────────────────────────────────────────────────────────
def integrar_harmony(datasets_raw, genes_score, score_key):
    """
    Combina todos los datasets con Harmony y calcula el gene score integrado.
    """
    import scipy.sparse

    adatas_pp = []
    for ds_id, adata in datasets_raw.items():
        print(f"    Preprocesando {ds_id} para integración...")
        adata_pp = adata.copy()
        sc.pp.normalize_total(adata_pp, target_sum=1e4)
        sc.pp.log1p(adata_pp)
        adata_pp.obs["dataset"] = ds_id
        adatas_pp.append(adata_pp)

    # Concatenar
    print("    Concatenando datasets...")
    adata_concat = adatas_pp[0].concatenate(
        adatas_pp[1:],
        batch_key="dataset_batch",
        batch_categories=list(datasets_raw.keys()),
    )

    # HVGs comunes
    sc.pp.highly_variable_genes(
        adata_concat,
        n_top_genes=2000,
        batch_key="dataset_batch",
    )

    # PCA
    sc.tl.pca(adata_concat, use_highly_variable=True)

    # Harmony
    print("    Ejecutando Harmony...")
    try:
        import harmonypy as hm
        pca_matrix = adata_concat.obsm["X_pca"]
        meta = adata_concat.obs[["dataset_batch"]]
        ho = hm.run_harmony(pca_matrix, meta, "dataset_batch", max_iter_harmony=20)
        # harmonypy 2.x devuelve Z_corr como (n_cells, n_pcs) directamente
        Z_corr = np.array(ho.Z_corr)
        if Z_corr.shape[0] == pca_matrix.shape[1]:  # si viene transpuesto (n_pcs, n_cells)
            Z_corr = Z_corr.T
        adata_concat.obsm["X_pca_harmony"] = Z_corr  # shape (n_cells, n_pcs)
        use_rep = "X_pca_harmony"
        print("    Harmony completado correctamente.")
    except Exception as e:
        print(f"    [AVISO] Harmony falló ({e}), usando PCA sin integrar")
        use_rep = "X_pca"

    # Vecinos y UMAP sobre la representación integrada
    sc.pp.neighbors(adata_concat, use_rep=use_rep, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_concat)

    # Gene score
    genes_presentes = [g for g in genes_score if g in adata_concat.var_names]
    if genes_presentes:
        sc.tl.score_genes(adata_concat, gene_list=genes_presentes, score_name=score_key)
    else:
        adata_concat.obs[score_key] = 0.0

    print(f"    Integración completada. Shape: {adata_concat.shape}")
    return adata_concat


# ──────────────────────────────────────────────────────────────
# DIBUJAR FIGURA POR GRUPO
# ──────────────────────────────────────────────────────────────
def dibujar_figura_grupo(adatas_umap, adata_integrado, grupo, num_grupo,
                          score_key, figures_dir):
    """
    Genera la figura multipanel para un grupo funcional:
      - Fila 1: 4 UMAPs individuales coloreados por gene score
      - Fila 2: UMAP integrado coloreado por gene score (grande)
                + UMAP integrado coloreado por dataset (orientación)
    """
    color_titulo = GRUPO_COLORS[(num_grupo - 1) % len(GRUPO_COLORS)]

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        f"Gene score: {grupo}",
        fontsize=16, fontweight="bold", color=color_titulo, y=0.98
    )

    # GridSpec: 2 filas
    # Fila 0: 4 paneles individuales
    # Fila 1: UMAP integrado score (grande) + UMAP integrado por dataset
    n_ds = len(adatas_umap)
    gs = gridspec.GridSpec(
        2, max(n_ds, 2),
        figure=fig,
        hspace=0.35, wspace=0.3,
        top=0.92, bottom=0.06,
        left=0.05, right=0.97
    )

    vmin_score = None
    vmax_score = None

    # Calcular rango de color común para todos los UMAPs de score
    all_scores = []
    for ds_id, adata_pp in adatas_umap.items():
        if score_key in adata_pp.obs.columns:
            all_scores.extend(adata_pp.obs[score_key].values)
    if adata_integrado is not None and score_key in adata_integrado.obs.columns:
        all_scores.extend(adata_integrado.obs[score_key].values)
    if all_scores:
        vmin_score = np.percentile(all_scores, 2)
        vmax_score = np.percentile(all_scores, 98)

    # ── Fila superior: UMAPs individuales ──
    for i, (ds_id, adata_pp) in enumerate(adatas_umap.items()):
        ax = fig.add_subplot(gs[0, i])

        if "X_umap" not in adata_pp.obsm:
            ax.set_visible(False)
            continue

        umap_coords = adata_pp.obsm["X_umap"]
        scores = adata_pp.obs.get(score_key, pd.Series(np.zeros(len(adata_pp))))

        sc_plot = ax.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=scores,
            cmap="RdBu_r",
            vmin=vmin_score, vmax=vmax_score,
            s=2, alpha=0.6, rasterized=True,
        )

        ax.set_title(DATASETS_LABELS.get(ds_id, ds_id), fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)

        plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04,
                     label="Gene score")

    # ── Fila inferior izquierda (span 3 cols): UMAP integrado score ──
    ax_int_score = fig.add_subplot(gs[1, :3])
    if adata_integrado is not None and "X_umap" in adata_integrado.obsm:
        umap_int = adata_integrado.obsm["X_umap"]
        scores_int = adata_integrado.obs.get(
            score_key, pd.Series(np.zeros(len(adata_integrado)))
        )
        sc_int = ax_int_score.scatter(
            umap_int[:, 0], umap_int[:, 1],
            c=scores_int,
            cmap="RdBu_r",
            vmin=vmin_score, vmax=vmax_score,
            s=1.5, alpha=0.5, rasterized=True,
        )
        ax_int_score.set_title("UMAP integrado — Gene score", fontsize=12, fontweight="bold")
        ax_int_score.set_xlabel("UMAP 1", fontsize=9)
        ax_int_score.set_ylabel("UMAP 2", fontsize=9)
        ax_int_score.tick_params(labelsize=8)
        plt.colorbar(sc_int, ax=ax_int_score, fraction=0.03, pad=0.02,
                     label="Gene score")

    # ── Fila inferior derecha (1 col): UMAP integrado coloreado por dataset ──
    ax_int_ds = fig.add_subplot(gs[1, 3])
    if adata_integrado is not None and "X_umap" in adata_integrado.obsm:
        umap_int = adata_integrado.obsm["X_umap"]
        datasets_obs = adata_integrado.obs["dataset"].values

        for ds_id, color in DATASET_COLORS.items():
            mask = datasets_obs == ds_id
            if mask.sum() == 0:
                continue
            ax_int_ds.scatter(
                umap_int[mask, 0], umap_int[mask, 1],
                c=color, s=1, alpha=0.5,
                label=DATASETS_LABELS.get(ds_id, ds_id),
                rasterized=True,
            )

        ax_int_ds.set_title("Integrado — Dataset", fontsize=11, fontweight="bold")
        ax_int_ds.set_xlabel("UMAP 1", fontsize=8)
        ax_int_ds.set_ylabel("UMAP 2", fontsize=8)
        ax_int_ds.tick_params(labelsize=7)
        legend = ax_int_ds.legend(
            fontsize=7, markerscale=4,
            loc="upper right", framealpha=0.8
        )

    # Guardar
    os.makedirs(figures_dir, exist_ok=True)
    fname = f"score_{num_grupo:02d}_{safe_name(grupo)}.png"
    out_path = os.path.join(figures_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"    [OK] Figura guardada: {out_path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Gene score UMAPs por grupo funcional."
    )
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--genes_dir",   default="170_genes")
    parser.add_argument("--figures_dir", default="figures/score")
    parser.add_argument("--genes_file",  default="170genes_clasificacion_FINAL.xlsx")
    parser.add_argument("--skip_integration", action="store_true",
                        help="Saltar la integración Harmony (más rápido)")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Cargar Excel de genes ──
    genes_path = os.path.join(args.genes_dir, args.genes_file)
    print(f"Cargando Excel de genes: {genes_path}")
    genes_df = pd.read_excel(genes_path)
    gene_col = genes_df.columns[0]

    if "Clasificacion_final" in genes_df.columns:
        genes_df["grupo_general"] = genes_df["Clasificacion_final"]
        print("  Usando columna 'Clasificacion_final'")
    else:
        print("  [ERROR] No se encontró columna 'Clasificacion_final'")
        return

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

    # ── Cargar datasets raw ──
    print("\nCargando datasets...")
    datasets_raw = {}

    for ds_id, h5ad_path in zip(DATASETS_ORDER, h5ad_files):
        print(f"  Cargando {ds_id}...")
        adata = sc.read_h5ad(h5ad_path)
        adata = normalizar_var_names(adata, ds_id)

        if "celltype" not in adata.obs.columns:
            print(f"  [SKIP] Sin columna 'celltype' en {ds_id}")
            continue

        datasets_raw[ds_id] = adata
        print(f"  {ds_id}: {adata.shape}")

    if not datasets_raw:
        print("[ERROR] No se cargó ningún dataset con columna 'celltype'.")
        return

    # ── Integración Harmony (una sola vez para todos los grupos) ──
    adata_integrado = None
    if not args.skip_integration:
        print("\nIntegrando datasets con Harmony (esto puede tardar varios minutos)...")
        try:
            adata_integrado = integrar_harmony(datasets_raw, [], "dummy_score")
        except Exception as e:
            print(f"  [AVISO] Integración falló: {e}")
            print("  Continuando sin UMAP integrado...")

    # ── Procesar cada grupo funcional ──
    for num_grupo, grupo in enumerate(grupos, start=1):
        print(f"\n{'='*60}")
        print(f"  Grupo {num_grupo}: {grupo}")
        print(f"{'='*60}")

        genes_grupo = (
            genes_df.loc[genes_df["grupo_general"] == grupo, gene_col]
            .dropna().astype(str).tolist()
        )
        print(f"  Genes en el grupo: {len(genes_grupo)}")

        score_key = f"score_{safe_name(grupo)}"

        # UMAPs individuales
        adatas_umap = {}
        for ds_id, adata in datasets_raw.items():
            print(f"  Calculando UMAP y score para {ds_id}...")
            try:
                adata_pp, n_genes = preparar_umap(adata, genes_grupo, score_key)
                adatas_umap[ds_id] = adata_pp
                print(f"    {n_genes}/{len(genes_grupo)} genes presentes")
            except Exception as e:
                print(f"    [ERROR] {ds_id}: {e}")

        # Score en el integrado
        if adata_integrado is not None:
            print(f"  Calculando score en dataset integrado...")
            genes_presentes = [g for g in genes_grupo if g in adata_integrado.var_names]
            try:
                sc.tl.score_genes(
                    adata_integrado,
                    gene_list=genes_presentes,
                    score_name=score_key
                )
                print(f"    {len(genes_presentes)}/{len(genes_grupo)} genes presentes en integrado")
            except Exception as e:
                print(f"    [ERROR] Score integrado: {e}")
                adata_integrado.obs[score_key] = 0.0

        # Dibujar figura
        print(f"  Generando figura...")
        try:
            dibujar_figura_grupo(
                adatas_umap, adata_integrado,
                grupo, num_grupo, score_key,
                args.figures_dir
            )
        except Exception as e:
            print(f"  [ERROR] Figura {grupo}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Gene scores completados.")
    print(f"  Figuras en: {args.figures_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
