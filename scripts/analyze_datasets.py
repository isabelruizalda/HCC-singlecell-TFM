"""
analyze_datasets.py
-------------------
Automatiza el análisis scRNA-seq de TODOS los datasets encontrados en DATA_DIR.
Por cada dataset genera:
  - Matrixplot y Dotplot globales (todos los genes de interés)
  - Clustermap de expresión media por tipo celular
  - Dotplot y Matrixplot por cada grupo funcional

El script detecta automáticamente la columna de tipo celular de cada dataset.
Si un dataset no tiene ninguna columna reconocida, se salta con un aviso.

Uso:
    python scripts/analyze_datasets.py \
        --data_dir data \
        --genes_dir 170_genes \
        --figures_dir figures \
        --results_dir results \
        --genes_file 170genes_clasificacion_FINAL.xlsx
"""

import os
import argparse
import glob

import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib
matplotlib.use("Agg")   # backend sin pantalla, necesario en HPC
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE REAGRUPACIÓN FUNCIONAL
# ──────────────────────────────────────────────────────────────
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

# Columnas de tipo celular a buscar en cada dataset, en orden de preferencia.
# Se usa la primera que se encuentre en adata.obs.
# Si en el futuro tienes un dataset nuevo con otra columna, añádela aquí.
CELLTYPE_CANDIDATES = [
    "celltype",                # GSE149614
    "cell_type",               # formato estándar alternativo
    "Cell_type_broad",         # ICB_HCC
    "author_cell_type",        # ICB_HCC (alternativa)
    "author_cell_type_update", # ICB_HCC (alternativa 2)
    "louvain",                 # GSE156625 (clusters Louvain)
    "leiden",                  # clusters Leiden (alternativa común)
    "seurat_clusters",         # por si hay alguno de Seurat
]


# ──────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────
def safe_name(text: str) -> str:
    """Convierte un string en un nombre seguro para ficheros."""
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
    """Extrae el ID del dataset a partir del nombre del fichero h5ad."""
    basename = os.path.basename(path)      # adata_GSE149614_raw.h5ad
    name = basename.replace(".h5ad", "")   # adata_GSE149614_raw
    name = name.replace("adata_", "")      # GSE149614_raw
    name = name.replace("_raw", "")        # GSE149614
    return name


def detectar_columna_celltype(adata) -> str:
    """
    Devuelve la primera columna de tipo celular encontrada en adata.obs.
    Devuelve None si no encuentra ninguna columna conocida.
    """
    for col in CELLTYPE_CANDIDATES:
        if col in adata.obs.columns:
            return col
    return None


# ──────────────────────────────────────────────────────────────
# ANÁLISIS GLOBAL DE UN DATASET
# ──────────────────────────────────────────────────────────────
def analisis_global(adata_sub, genes_presentes, celltype_col, figures_dir):
    """Matrixplot, Dotplot y Clustermap con todos los genes de interés."""

    import scipy.sparse

    sc.settings.figdir = figures_dir

    sc.pl.matrixplot(
        adata_sub,
        var_names=genes_presentes,
        groupby=celltype_col,
        standard_scale="var",
        save="_matrixplot_all_genes.png",
        show=False,
    )

    sc.pl.dotplot(
        adata_sub,
        var_names=genes_presentes,
        groupby=celltype_col,
        standard_scale="var",
        save="_dotplot_all_genes.png",
        show=False,
    )

    # Clustermap de expresión media por tipo celular
    X = adata_sub.X
    if scipy.sparse.issparse(X):
        X = X.toarray()

    expr_df = pd.DataFrame(
        X,
        index=adata_sub.obs[celltype_col],
        columns=adata_sub.var_names,
    )
    mean_expr = expr_df.groupby(level=0, observed=True).mean()
    mean_expr_scaled = (
        (mean_expr - mean_expr.min()) / (mean_expr.max() - mean_expr.min())
    ).fillna(0)

    g = sns.clustermap(
        mean_expr_scaled.T,
        figsize=(14, 20),
        cmap="viridis",
        col_cluster=True,
        row_cluster=True,
    )
    g.savefig(
        os.path.join(figures_dir, "clustermap_mean_expression_all_genes.png"),
        dpi=300,
    )
    plt.close("all")
    print("    [OK] Análisis global completado.")


# ──────────────────────────────────────────────────────────────
# ANÁLISIS POR GRUPO FUNCIONAL
# ──────────────────────────────────────────────────────────────
def analisis_por_grupo(adata, genes_df, gene_col, celltype_col, figures_dir):
    """Dotplot y Matrixplot para cada grupo funcional general."""

    grupos = genes_df["grupo_general"].dropna().unique()

    for grupo in grupos:
        genes_grupo = (
            genes_df.loc[genes_df["grupo_general"] == grupo, gene_col]
            .dropna()
            .astype(str)
            .tolist()
        )
        genes_presentes_grupo = [g for g in genes_grupo if g in adata.var_names]
        grupo_safe = safe_name(grupo)

        print(f"    Grupo '{grupo}': {len(genes_presentes_grupo)}/{len(genes_grupo)} genes presentes")

        if len(genes_presentes_grupo) == 0:
            print(f"    [SKIP] Sin genes presentes para el grupo '{grupo}'.")
            continue

        sc.settings.figdir = figures_dir

        sc.pl.dotplot(
            adata,
            var_names=genes_presentes_grupo,
            groupby=celltype_col,
            standard_scale="var",
            title=grupo,
            save=f"_{grupo_safe}_dotplot.png",
            show=False,
        )

        sc.pl.matrixplot(
            adata,
            var_names=genes_presentes_grupo,
            groupby=celltype_col,
            standard_scale="var",
            title=grupo,
            save=f"_{grupo_safe}_matrixplot.png",
            show=False,
        )

    plt.close("all")
    print("    [OK] Análisis por grupos funcionales completado.")


# ──────────────────────────────────────────────────────────────
# TABLA DE TIPOS CELULARES
# ──────────────────────────────────────────────────────────────
def generar_tabla_celltypes(adata, celltype_col, dataset_id, results_dir):
    """
    Genera una tabla con N_células y porcentaje por tipo celular,
    añade una fila TOTAL y la guarda como Excel en results_dir.
    Equivalente a la celda 'Tabla resumida x tipo cel' del notebook de referencia.
    """
    os.makedirs(results_dir, exist_ok=True)

    tabla = adata.obs[celltype_col].value_counts().reset_index()
    tabla.columns = ["Tipo_celular", "N_celulas"]
    tabla["Porcentaje"] = (tabla["N_celulas"] / tabla["N_celulas"].sum()) * 100

    total_row = pd.DataFrame({
        "Tipo_celular": ["TOTAL"],
        "N_celulas":    [tabla["N_celulas"].sum()],
        "Porcentaje":   [100.0],
    })
    tabla = pd.concat([tabla, total_row], ignore_index=True)

    out_path = os.path.join(results_dir, f"tabla_celltypes_{dataset_id}.xlsx")
    tabla.to_excel(out_path, index=False)
    print(f"    [OK] Tabla de tipos celulares guardada en: {out_path}")


# ──────────────────────────────────────────────────────────────
# UMAP
# ──────────────────────────────────────────────────────────────
def generar_umap(adata, celltype_col, figures_dir):
    """
    Normaliza, selecciona HVGs, calcula PCA + vecinos + UMAP
    y guarda la figura coloreada por tipo celular.
    Equivalente a las celdas 'PCA y vecinos para UMAP' y 'Guardar UMAP'
    del notebook de referencia. Trabaja sobre una copia para no alterar
    la matriz de expresión usada en el resto del análisis.
    """
    import scipy.sparse

    adata_umap = adata.copy()

    # Normalización
    sc.pp.normalize_total(adata_umap, target_sum=1e4)
    sc.pp.log1p(adata_umap)

    # HVGs
    sc.pp.highly_variable_genes(adata_umap, n_top_genes=2000)
    adata_umap = adata_umap[:, adata_umap.var.highly_variable].copy()

    # Reducción dimensional
    sc.pp.scale(adata_umap, max_value=10)
    sc.tl.pca(adata_umap, svd_solver="arpack")
    sc.pp.neighbors(adata_umap, n_neighbors=10, n_pcs=20)
    sc.tl.umap(adata_umap)

    # Guardar figura
    sc.settings.figdir = figures_dir
    sc.pl.umap(adata_umap, color=[celltype_col], save="_umap_celltypes.png", show=False)
    plt.close("all")
    print(f"    [OK] UMAP guardado en: {figures_dir}")


# ──────────────────────────────────────────────────────────────
# PROCESO COMPLETO DE UN DATASET
# ──────────────────────────────────────────────────────────────
def procesar_dataset(h5ad_path, genes_df, gene_col, func_col, figures_base_dir, args_results_dir):
    """Carga el dataset y ejecuta todos los análisis."""

    dataset_id = dataset_id_from_path(h5ad_path)
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_id}")
    print(f"  Fichero: {h5ad_path}")
    print(f"{'='*60}")

    # Carpeta de figuras específica del dataset
    figures_dir = os.path.join(figures_base_dir, dataset_id)
    os.makedirs(figures_dir, exist_ok=True)

    # Cargar datos
    print("  Cargando datos...")
    adata = sc.read_h5ad(h5ad_path)

    # Si los var_names son Ensembl IDs, reindexar con los símbolos de gen
    if adata.var_names[0].startswith("ENSG") and "feature_name" in adata.var.columns:
        print("  Reindexando var_names con feature_name (Ensembl IDs detectados)...")
        adata.var_names = adata.var["feature_name"].astype(str).values
        adata.var_names_make_unique()
        adata.raw = None  # evitar que scanpy busque genes en raw (tiene Ensembl IDs)

    # Detectar columna de tipo celular automáticamente
    celltype_col = detectar_columna_celltype(adata)
    if celltype_col is None:
        print(f"  [SKIP] No se encontró columna de tipo celular en este dataset.")
        print(f"         Columnas disponibles: {adata.obs.columns.tolist()}")
        print(f"         Si quieres procesarlo, añade su columna a CELLTYPE_CANDIDATES en el script.")
        return

    print(f"  Columna de tipo celular detectada: '{celltype_col}'")
    print(f"  Shape: {adata.shape}  |  Grupos únicos: {adata.obs[celltype_col].nunique()}")

    # Cruzar genes de interés con el dataset
    genes_interes = genes_df[gene_col].dropna().astype(str).tolist()
    genes_presentes = [g for g in genes_interes if g in adata.var_names]
    print(f"  Genes de interés: {len(genes_interes)}  |  presentes: {len(genes_presentes)}")

    if len(genes_presentes) == 0:
        print("  [SKIP] No hay genes de interés presentes en este dataset.")
        return

    # Subconjunto con genes de interés
    adata_sub = adata[:, genes_presentes].copy()

    # Reagrupación funcional
    # grupo_general ya viene asignado desde main() con la clasificacion ponderada
    if "grupo_general" not in genes_df.columns:
        if "Clasificacion_final" in genes_df.columns:
            genes_df["grupo_general"] = genes_df["Clasificacion_final"]
        else:
            genes_df["grupo_general"] = genes_df[func_col].map(REAGRUPACION_GENERAL)

    # ── Tabla de tipos celulares ──
    print("  Generando tabla de tipos celulares...")
    generar_tabla_celltypes(adata, celltype_col, dataset_id, args_results_dir)

    # ── UMAP ──
    print("  Generando UMAP...")
    generar_umap(adata, celltype_col, figures_dir)

    # ── Análisis global ──
    print("  Generando figuras globales...")
    analisis_global(adata_sub, genes_presentes, celltype_col, figures_dir)

    # ── Análisis por grupo funcional ──
    print("  Generando figuras por grupo funcional...")
    analisis_por_grupo(adata, genes_df, gene_col, celltype_col, figures_dir)

    print(f"  Figuras guardadas en: {figures_dir}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Análisis scRNA-seq automatizado para múltiples datasets."
    )
    parser.add_argument("--data_dir",    default="data",        help="Carpeta con los ficheros .h5ad")
    parser.add_argument("--genes_dir",   default="170_genes",   help="Carpeta con el Excel de genes")
    parser.add_argument("--results_dir", default="results",     help="Carpeta de resultados")
    parser.add_argument("--figures_dir", default="figures",     help="Carpeta raíz para las figuras")
    parser.add_argument(
        "--genes_file",
        default="170genes_clasificacion_FINAL.xlsx",
        help="Nombre del fichero Excel de genes (dentro de genes_dir)",
    )
    args = parser.parse_args()

    # Crear directorios necesarios
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # Cargar Excel de genes
    genes_path = os.path.join(args.genes_dir, args.genes_file)
    print(f"Cargando Excel de genes: {genes_path}")
    genes_df = pd.read_excel(genes_path)

    gene_col = genes_df.columns[0]
    func_col = genes_df.columns[1]
    print(f"  Columna genes: '{gene_col}'  |  Columna función: '{func_col}'")

    # Usar Clasificacion_final si existe (Excel nuevo con puntuación ponderada)
    # Si no, usar reagrupación manual como fallback
    if "Clasificacion_final" in genes_df.columns:
        genes_df["grupo_general"] = genes_df["Clasificacion_final"]
        print("  Usando columna 'Clasificacion_final' (clasificacion ponderada GO+Reactome+MSigDB+KEGG)")
    else:
        genes_df["grupo_general"] = genes_df[func_col].map(REAGRUPACION_GENERAL)
        print("  Usando reagrupacion manual (REAGRUPACION_GENERAL) como fallback")
    print(f"  Distribución de grupos generales:\n{genes_df['grupo_general'].value_counts().to_string()}")

    # Buscar todos los datasets h5ad
    pattern = os.path.join(args.data_dir, "*.h5ad")
    datasets = sorted(glob.glob(pattern))

    if not datasets:
        print(f"\n[ERROR] No se encontraron ficheros .h5ad en '{args.data_dir}'.")
        return

    print(f"\nDatasets encontrados: {len(datasets)}")
    for d in datasets:
        print(f"  - {os.path.basename(d)}")

    # Procesar cada dataset
    for h5ad_path in datasets:
        try:
            procesar_dataset(h5ad_path, genes_df.copy(), gene_col, func_col, args.figures_dir, args.results_dir)
        except Exception as e:
            print(f"\n[ERROR] Fallo al procesar '{h5ad_path}': {e}")
            import traceback
            traceback.print_exc()
            print("  Continuando con el siguiente dataset...\n")

    print(f"\n{'='*60}")
    print("  Análisis completado para todos los datasets.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
