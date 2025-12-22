"""
analyze_data.py

Etapa 4: análisis estadístico.
Entrada: data/processed/adult_clean.csv
Salidas:
- reports/summary.json
- reports/figures/*.png
- reports/tables/*.csv

Incluye:
- Análisis supervisado (Regresión logística).
- Análisis no supervisado (PCA + KMeans) con muestreo configurable.
- Contraste de hipótesis (Mann-Whitney U) entre grupos de income.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .utils import ensure_dirs, load_project_config
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass(frozen=True)
class SupervisedMetrics:
    """Métricas del modelo supervisado."""

    roc_auc: float
    accuracy: float
    precision_pos: float
    recall_pos: float
    f1_pos: float
    tn: int
    fp: int
    fn: int
    tp: int


@dataclass(frozen=True)
class HypothesisTest:
    """Resultado del contraste de hipótesis."""

    test: str
    variable: str
    group0: str
    group1: str
    mean0: float
    mean1: float
    median0: float
    median1: float
    p_value: float
    ci_mean_diff_lo: float
    ci_mean_diff_hi: float

@dataclass(frozen=True)
class ClusterResult:
    """Resultado del análisis no supervisado."""

    sample_n: int
    k: int
    silhouette: float


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["income"] = df["income"].astype("string").str.strip()
    return df


def _binary_target(df: pd.DataFrame) -> pd.Series:
    return (df["income"] == ">50K").astype(int)


def _class_distribution(y: pd.Series) -> Dict[str, Any]:
    counts = y.value_counts().to_dict()
    total = int(len(y))
    pct = {str(k): float(v / total) for k, v in counts.items()}
    return {"total": total, "counts": counts, "pct": pct}


def _supervised(
    df: pd.DataFrame, figures_dir: Path, tables_dir: Path, random_state: int
) -> SupervisedMetrics:
    y = _binary_target(df)
    X = df.drop(columns=["income"]).copy()

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_cols,
            ),
        ]
    )

    model = LogisticRegression(max_iter=2000, n_jobs=None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, proba))
    acc = float((pred == y_test).mean())

    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    pos = rep.get("1", {})
    precision_pos = float(pos.get("precision", 0.0))
    recall_pos = float(pos.get("recall", 0.0))
    f1_pos = float(pos.get("f1-score", 0.0))

    # Confusion matrix
    tn = int(((y_test == 0) & (pred == 0)).sum())
    fp = int(((y_test == 0) & (pred == 1)).sum())
    fn = int(((y_test == 1) & (pred == 0)).sum())
    tp = int(((y_test == 1) & (pred == 1)).sum())

    # Save classification report table
    rep_df = pd.DataFrame(rep).T
    rep_df.to_csv(tables_dir / "classification_report.csv")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay.from_predictions(y_test, pred)
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    return SupervisedMetrics(
        roc_auc=roc_auc,
        accuracy=acc,
        precision_pos=precision_pos,
        recall_pos=recall_pos,
        f1_pos=f1_pos,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )

def bootstrap_ci_mean_diff(
    g0: pd.Series,
    g1: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Calcula un intervalo de confianza bootstrap para la diferencia de medias:
    mean(g1) - mean(g0).

    Parameters
    ----------
    g0:
        Serie numérica del grupo 0.
    g1:
        Serie numérica del grupo 1.
    n_boot:
        Número de remuestreos bootstrap.
    alpha:
        Nivel de significancia (0.05 para IC 95%).
    random_state:
        Semilla para reproducibilidad.

    Returns
    -------
    tuple[float, float]
        (límite inferior, límite superior) del IC.
    """
    rng = np.random.default_rng(random_state)
    a0 = g0.dropna().to_numpy(dtype=float)
    a1 = g1.dropna().to_numpy(dtype=float)

    if len(a0) == 0 or len(a1) == 0:
        return float("nan"), float("nan")

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b0 = rng.choice(a0, size=len(a0), replace=True)
        b1 = rng.choice(a1, size=len(a1), replace=True)
        diffs[i] = float(b1.mean() - b0.mean())

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi

def _hypothesis_test(df: pd.DataFrame) -> HypothesisTest:
    y = _binary_target(df)
    x = pd.to_numeric(df["hours_per_week"], errors="coerce").fillna(0.0)

    g0 = x[y == 0]
    g1 = x[y == 1]

    stat = mannwhitneyu(g0, g1, alternative="two-sided")
    ci_lo, ci_hi = bootstrap_ci_mean_diff(
        g0, g1, n_boot=2000, alpha=0.05, random_state=42
    )
    return HypothesisTest(
        test="Mann–Whitney U",
        variable="hours_per_week",
        group0="<=50K",
        group1=">50K",
        mean0=float(g0.mean()),
        mean1=float(g1.mean()),
        median0=float(g0.median()),
        median1=float(g1.median()),
        p_value=float(stat.pvalue),
        ci_mean_diff_lo=float(ci_lo),
        ci_mean_diff_hi=float(ci_hi),
    )


def _clustering(
    df: pd.DataFrame, tables_dir: Path, random_state: int, sample_n: int
) -> ClusterResult:
    # Muestreo para escalabilidad
    if sample_n > len(df):
        sample_n = len(df)

    df_s = df.sample(n=sample_n, random_state=random_state).copy()
    y = _binary_target(df_s)
    X = df_s.drop(columns=["income"]).copy()

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_enc = pre.fit_transform(X)

    # PCA para reducir dimensionalidad
    pca = PCA(n_components=10, random_state=random_state)
    X_pca = pca.fit_transform(X_enc.toarray())

    # Probar k=2..6 y elegir silhouette máximo
    best_k = 2
    best_sil = -1.0
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_pca)
        sil = float(silhouette_score(X_pca, labels))
        if sil > best_sil:
            best_sil = sil
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_pca)

    prof = (
        pd.DataFrame({"cluster": labels, "income_pos": y})
        .groupby("cluster")
        .agg(n=("income_pos", "size"), pct_pos=("income_pos", "mean"))
        .reset_index()
    )
    prof.to_csv(tables_dir / "cluster_profile.csv", index=False)

    return ClusterResult(sample_n=sample_n, k=best_k, silhouette=best_sil)


def main() -> Path:
    """
    Ejecuta el análisis y genera summary.json.

    Returns
    -------
    Path
        Ruta del summary.json.
    """
    _, processed_dir, reports_dir, tables_dir = ensure_dirs()
    figures_dir = reports_dir / "figures"

    cfg = load_project_config()
    random_state = int(cfg.get("random_state", 42))
    sample_n = int(cfg.get("cluster_sample_n", 800))

    in_csv = processed_dir / "adult_clean.csv"
    df = _load_data(in_csv)
    
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Distribución de hours_per_week por clase (histograma)
    if "hours_per_week" in df.columns and "income" in df.columns:
        plt.figure()
        for label in ["<=50K", ">50K"]:
            s = df.loc[df["income"].astype(str).str.strip() == label, "hours_per_week"]
            s = pd.to_numeric(s, errors="coerce").dropna()
            plt.hist(s.values, bins=30, alpha=0.5, label=label)
        plt.xlabel("hours_per_week")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de horas trabajadas por clase de income")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "hours_per_week_by_income.png", dpi=160)
        plt.close()
    
    # 2) Proporción de education por clase (top 10 categorías)
    if "education" in df.columns and "income" in df.columns:
        tmp = df.copy()
        tmp["income"] = tmp["income"].astype(str).str.strip()
        top_edu = tmp["education"].astype(str).value_counts().head(10).index
    
        ctab = (
            tmp[tmp["education"].astype(str).isin(top_edu)]
            .groupby(["education", "income"])
            .size()
            .unstack(fill_value=0)
        )
    
        # normalizar por education para ver proporciones dentro de cada categoría
        ctab_prop = ctab.div(ctab.sum(axis=1), axis=0)
    
        plt.figure(figsize=(9, 4))
        x = range(len(ctab_prop.index))
        y0 = ctab_prop.get("<=50K", pd.Series([0]*len(ctab_prop), index=ctab_prop.index)).values
        y1 = ctab_prop.get(">50K", pd.Series([0]*len(ctab_prop), index=ctab_prop.index)).values
    
        plt.bar(x, y0, label="<=50K")
        plt.bar(x, y1, bottom=y0, label=">50K")
        plt.xticks(list(x), list(ctab_prop.index), rotation=30, ha="right")
        plt.ylabel("Proporción")
        plt.title("Proporción de income por nivel educativo (top 10)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "education_income_proportions.png", dpi=160)
        plt.close()
    
    y = _binary_target(df)
    dist = _class_distribution(y)

    sup = _supervised(df, figures_dir, tables_dir, random_state)
    hyp = _hypothesis_test(df)
    clu = _clustering(df, tables_dir, random_state, sample_n)

    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "class_distribution": dist,
        "supervised": asdict(sup),
        "hypothesis_test": asdict(hyp),
        "clustering": asdict(clu),
        "notes": {
            "clustering_sample_note": (
                "El análisis no supervisado se realiza sobre una muestra "
                "para mejorar escalabilidad; se reporta el tamaño de muestra."
            )
        },
    }

    out_json = reports_dir / "summary.json"
    out_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[analyze_data] Wrote: {out_json}")
    return out_json


if __name__ == "__main__":
    main()
