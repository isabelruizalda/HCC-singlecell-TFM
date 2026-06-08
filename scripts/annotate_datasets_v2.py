"""
annotate_datasets.py
--------------------
Transfiere las anotaciones de tipo celular desde el dataset de referencia
(Atlas_HCC) al resto de datasets usando KNN Label Transfer con scanpy.

El resultado es que cada dataset target tendrá una nueva columna 'celltype'
en adata.obs, sobreescribiendo el fichero original.

Por defecto, el script descubre automáticamente todos los .h5ad presentes
en data_dir y anota los que no son la referencia. Si se pasa --targets,
se usan solo esos.

Uso:
    # Autodescubrimiento (recomendado):
    python scripts/annotate_datasets.py \\
        --data_dir data \\
        --figures_dir figures

    # Targets manuales:
    python scripts/annotate_datasets.py \\
        --data_dir data \\
        --targets adata_HCC_iCCA_raw.h5ad adata_Tumor_vs_Normal_raw.h5ad
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────
# NOMBRE DEL FICHERO DE REFERENCIA
# Si cambias el dataset de referencia, solo tienes que cambiar esto.
# ──────────────────────────────────────────────────────────────
REFERENCE_FILE = "adata_Atlas_HCC_raw.h5ad"


# ──────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE VAR_NAMES (Ensembl IDs → símbolos de gen)
# ──────────────────────────────────────────────────────────────
# Columnas candidatas donde puede estar el símbolo de gen,
# en orden de preferencia. Añade aquí cualquier columna nueva
# que aparezca en futuros datasets.
GENE_SYMBOL_CANDIDATES = ["feature_name", "gene_name", "gene_symbol", "Symbol", "symbol"]

def normalizar_var_names(adata, dataset_name: str):
    """
    Si los var_names son Ensembl IDs (empiezan por ENSG), intenta
    reindexarlos con símbolos de gen buscando en GENE_SYMBOL_CANDIDATES.
    Si no encuentra ninguna columna válida, avisa pero continúa.
    Si los var_names ya son símbolos, no hace nada.
    """
    if not adata.var_names[0].startswith("ENSG"):
        return adata  # ya son símbolos, nada que hacer

    print(f"    [{dataset_name}] Ensembl IDs detectados, reindexando var_names...")

    col_encontrada = None
    for col in GENE_SYMBOL_CANDIDATES:
        if col in adata.var.columns:
            col_encontrada = col
            break

    if col_encontrada is None:
        print(f"    [AVISO] No se encontró columna de símbolos de gen en adata.var.")
        print(f"            Columnas disponibles: {adata.var.columns.tolist()}")
        print(f"            Añade la columna correcta a GENE_SYMBOL_CANDIDATES en el script.")
        return adata  # devuelve sin cambios; el error de 0 genes comunes será descriptivo

    adata.var_names = adata.var[col_encontrada].astype(str).values
    adata.var_names_make_unique()
    adata.raw = None  # evitar que scanpy busque genes en raw (tiene Ensembl IDs)
    print(f"    [{dataset_name}] var_names reindexados con '{col_encontrada}'. "
          f"Ejemplo: {adata.var_names[:5].tolist()}")
    return adata


# ──────────────────────────────────────────────────────────────
# AUTODESCUBRIMIENTO DE DATASETS
# ──────────────────────────────────────────────────────────────
def descubrir_targets(data_dir: str, reference_file: str) -> list:
    """
    Busca todos los .h5ad en data_dir y devuelve los que no son
    la referencia, ordenados alfabéticamente.
    """
    todos = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".h5ad") and f != reference_file
    ])
    print(f"\n  Datasets encontrados en {data_dir} (excluida referencia):")
    for f in todos:
        print(f"    - {f}")
    return todos


# ──────────────────────────────────────────────────────────────
# PREPROCESAMIENTO SIN SCALE (evita densificar la matriz)
# ──────────────────────────────────────────────────────────────
def preprocesar(adata, dataset_name: str):
    """
    Normalización y PCA básicos para poder comparar datasets.
    Se omite sc.pp.scale() deliberadamente para no densificar
    la matriz sparse y evitar problemas de memoria.
    """
    print(f"    Preprocesando {dataset_name}...")

    # Guardar counts raw
    adata.layers["counts"] = adata.X.copy()

    # Normalizar a 10.000 cuentas por célula y log-transformar
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Genes altamente variables para reducir dimensionalidad
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    # PCA solo con HVGs (sin escalar — el KNN con coseno no lo necesita)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

    print(f"    [OK] {dataset_name} preprocesado. Shape: {adata.shape}")
    return adata


# ──────────────────────────────────────────────────────────────
# KNN LABEL TRANSFER EN ESPACIO PCA
# ──────────────────────────────────────────────────────────────
def knn_label_transfer(adata_ref, adata_query, celltype_col: str, n_neighbors: int = 30):
    """
    Transfiere etiquetas de tipo celular desde adata_ref a adata_query
    usando KNN en el espacio PCA compartido.

    Usa distancia coseno, que funciona bien con datos log-normalizados
    sin necesidad de escalar.
    """
    from sklearn.neighbors import KNeighborsClassifier

    print(f"    Buscando genes compartidos entre referencia y query...")
    genes_comunes = adata_ref.var_names.intersection(adata_query.var_names)
    print(f"    Genes comunes: {len(genes_comunes)}")

    if len(genes_comunes) < 500:
        print(f"    [AVISO] Pocos genes en común ({len(genes_comunes)}). "
              f"La transferencia puede ser menos precisa.")

    # HVGs comunes entre los dos datasets
    hvg_ref    = set(adata_ref.var_names[adata_ref.var.highly_variable])
    hvg_query  = set(adata_query.var_names[adata_query.var.highly_variable])
    hvg_comunes = list(hvg_ref & hvg_query & set(genes_comunes))
    print(f"    HVGs comunes para el KNN: {len(hvg_comunes)}")

    if len(hvg_comunes) < 200:
        print(f"    [AVISO] Pocos HVGs comunes, usando todos los genes comunes.")
        hvg_comunes = list(genes_comunes)

    # Extraer matrices de expresión (sparse → array solo del subconjunto HVG)
    import scipy.sparse
    X_ref = adata_ref[:, hvg_comunes].X
    if scipy.sparse.issparse(X_ref):
        X_ref = X_ref.toarray()

    X_query = adata_query[:, hvg_comunes].X
    if scipy.sparse.issparse(X_query):
        X_query = X_query.toarray()

    labels_ref = adata_ref.obs[celltype_col].values

    # Entrenar KNN
    print(f"    Entrenando KNN con {n_neighbors} vecinos (distancia coseno)...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    knn.fit(X_ref, labels_ref)

    # Predecir
    print(f"    Prediciendo tipos celulares para {adata_query.shape[0]} células...")
    predicted_labels = knn.predict(X_query)
    predicted_proba  = knn.predict_proba(X_query).max(axis=1)

    return predicted_labels, predicted_proba


# ──────────────────────────────────────────────────────────────
# FIGURA DE VALIDACIÓN
# ──────────────────────────────────────────────────────────────
def guardar_figura_validacion(adata, dataset_name, figures_dir):
    """UMAP coloreado por tipo celular predicho y confianza del KNN."""

    os.makedirs(figures_dir, exist_ok=True)
    sc.settings.figdir = figures_dir

    print(f"    Calculando UMAP para validación visual...")
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(adata)

    sc.pl.umap(
        adata,
        color=["celltype", "knn_confidence"],
        save=f"_{dataset_name}_celltype_transfer.png",
        show=False,
    )
    plt.close("all")
    print(f"    [OK] UMAP guardado en {figures_dir}")


# ──────────────────────────────────────────────────────────────
# PROCESO COMPLETO PARA UN DATASET TARGET
# ──────────────────────────────────────────────────────────────
def anotar_dataset(adata_ref, h5ad_path, celltype_col, n_neighbors, figures_dir):
    """Carga un dataset sin anotar, transfiere etiquetas y lo guarda."""

    dataset_name = (
        os.path.basename(h5ad_path)
        .replace(".h5ad", "")
        .replace("adata_", "")
        .replace("_raw", "")
    )

    print(f"\n{'='*60}")
    print(f"  Anotando: {dataset_name}")
    print(f"{'='*60}")

    print("  Cargando datos...")
    adata_query = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata_query.shape}")
    adata_query = normalizar_var_names(adata_query, dataset_name)

    # Preprocesar
    adata_query = preprocesar(adata_query, dataset_name)

    # Transferir etiquetas
    print("  Transfiriendo etiquetas celulares...")
    predicted_labels, predicted_proba = knn_label_transfer(
        adata_ref, adata_query, celltype_col, n_neighbors
    )

    # Añadir columnas al objeto
    adata_query.obs["celltype"]       = pd.Categorical(predicted_labels)
    adata_query.obs["knn_confidence"] = predicted_proba

    # Resumen
    print(f"\n  Distribución de tipos celulares predichos:")
    print(adata_query.obs["celltype"].value_counts().to_string())
    print(f"\n  Confianza media del KNN: {predicted_proba.mean():.3f}")
    cells_low = (predicted_proba < 0.5).sum()
    print(f"  Células con confianza < 0.5: {cells_low} ({cells_low/len(predicted_proba)*100:.1f}%)")

    # UMAP de validación
    fig_dir = os.path.join(figures_dir, dataset_name)
    guardar_figura_validacion(adata_query, dataset_name, fig_dir)

    # Guardar (sobreescribe el fichero raw con la versión anotada)
    print(f"\n  Guardando dataset anotado en: {h5ad_path}")
    adata_query.write(h5ad_path)
    print(f"  [OK] Guardado correctamente.")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Transferencia de anotaciones celulares por KNN."
    )
    parser.add_argument("--data_dir",     default="data",
                        help="Carpeta donde están los .h5ad")
    parser.add_argument("--figures_dir",  default="figures",
                        help="Carpeta donde guardar las figuras")
    parser.add_argument("--reference",    default=REFERENCE_FILE,
                        help="Fichero .h5ad de referencia (ya anotado)")
    parser.add_argument("--targets",      nargs="+", default=None,
                        help="Ficheros .h5ad a anotar. Si no se indica, "
                             "se descubren automáticamente todos los .h5ad "
                             "en data_dir excepto la referencia.")
    parser.add_argument("--celltype_col", default="celltype",
                        help="Columna de tipo celular en la referencia")
    parser.add_argument("--n_neighbors",  type=int, default=30,
                        help="Número de vecinos para el KNN")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # Cargar y preprocesar referencia
    ref_path = os.path.join(args.data_dir, args.reference)
    print(f"\nCargando dataset de referencia: {ref_path}")
    adata_ref = sc.read_h5ad(ref_path)
    print(f"  Shape: {adata_ref.shape}")
    adata_ref = normalizar_var_names(adata_ref, "Atlas_HCC")
    print(f"  Tipos celulares:\n{adata_ref.obs[args.celltype_col].value_counts().to_string()}")
    adata_ref = preprocesar(adata_ref, "Atlas_HCC")

    # Calcular y guardar UMAP de la referencia si no existe
    if "X_umap" not in adata_ref.obsm:
        print("\n  Calculando UMAP de la referencia (Atlas_HCC)...")
        sc.pp.neighbors(adata_ref, n_neighbors=15, use_rep="X_pca")
        sc.tl.umap(adata_ref)
        print("  Guardando referencia con UMAP...")
        adata_ref.write(ref_path)
        print("  [OK] UMAP de referencia guardado.")
    else:
        print("\n  UMAP de referencia ya existe, reutilizando.")

    # Descubrir targets automáticamente si no se especifican
    if args.targets is None:
        target_files = descubrir_targets(args.data_dir, args.reference)
    else:
        target_files = args.targets

    if not target_files:
        print("\n[AVISO] No se encontraron datasets para anotar.")
        return

    print(f"\n  Se anotarán {len(target_files)} dataset(s).")

    # Anotar cada dataset target
    for target_file in target_files:
        target_path = os.path.join(args.data_dir, target_file)
        if not os.path.exists(target_path):
            print(f"\n[SKIP] No se encontró: {target_path}")
            continue
        try:
            anotar_dataset(
                adata_ref=adata_ref,
                h5ad_path=target_path,
                celltype_col=args.celltype_col,
                n_neighbors=args.n_neighbors,
                figures_dir=args.figures_dir,
            )
        except Exception as e:
            print(f"\n[ERROR] Fallo al anotar '{target_file}': {e}")
            import traceback
            traceback.print_exc()
            print("  Continuando con el siguiente...\n")

    print(f"\n{'='*60}")
    print("  Anotación completada.")
    print(f"  Ahora puedes ejecutar analyze_datasets.py normalmente.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
