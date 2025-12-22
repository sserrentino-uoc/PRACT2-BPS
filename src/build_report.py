"""
build_report.py

Genera un reporte en Markdown con observaciones y conclusiones
basadas en resultados reales del análisis.

Entrada:
- reports/summary.json
- reports/tables/missing_before.csv
- reports/tables/missing_after.csv
- reports/tables/semantic_missing_before.csv (opcional)
- reports/tables/semantic_missing_after.csv (opcional)
- reports/tables/winsor_caps.csv
- reports/tables/classification_report.csv (opcional)

Salida:
- reports/report.md
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import ensure_dirs, load_project_config


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _members_block(cfg: Dict[str, Any]) -> str:
    members = cfg.get("members", [])
    repo_url = cfg.get("repo_url", "PENDIENTE")
    video_url = cfg.get("video_url", "PENDIENTE")

    members_lines = "\n".join([f"- {m}" for m in members]) or "- PENDIENTE"
    return (
        "**Integrantes**\n"
        f"{members_lines}\n\n"
        f"**Repositorio**: {repo_url}\n\n"
        f"**Vídeo**: {video_url}\n"
    )


def _contrib_table(cfg: Dict[str, Any]) -> str:
    """
    Tabla de contribuciones según el enunciado:
    investigación previa, redacción, código, vídeo; firmada con iniciales.
    """
    initials = cfg.get("members_initials", ["AA", "BB"])
    a = initials[0] if len(initials) >= 1 else "AA"
    b = initials[1] if len(initials) >= 2 else "BB"
    sign = f"{a}, {b}"
    return (
        "| Contribuciones | Firma |\n"
        "|---|---|\n"
        f"| Investigación previa | {sign} |\n"
        f"| Redacción de las respuestas | {sign} |\n"
        f"| Desarrollo del código | {sign} |\n"
        f"| Participación en el vídeo | {sign} |\n"
    )


def _safe_read_csv(path: Path, **kwargs: Any) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


def main() -> Path:
    """
    Construye reports/report.md.

    Returns
    -------
    Path
        Ruta del Markdown generado.
    """
    _, processed_dir, reports_dir, tables_dir = ensure_dirs()
    cfg = load_project_config()

    summary_path = reports_dir / "summary.json"
    summary = _load_summary(summary_path)

        # --- Extraer métricas del summary.json de forma robusta ---
    sup = summary.get("supervised", {}) or summary.get("supervised_model", {}) or {}
    sup_auc = sup.get("roc_auc") or sup.get("auc") or sup.get("ROC_AUC")
    sup_acc = sup.get("accuracy") or sup.get("acc") or sup.get("ACC")
    
    # métricas de la clase positiva (>50K), si existen
    sup_pos = sup.get("positive_class", {}) or sup.get(">50K", {}) or sup.get("class_1", {}) or {}
    sup_rec = sup_pos.get("recall") or sup.get("recall_pos") or sup.get("recall")
    
    # si recall no está en positivo_class, intentar buscar en un classification_report serializado
    if sup_rec is None:
        rep = sup.get("classification_report", {}) or {}
        if isinstance(rep, dict):
            # nombres típicos
            for key in ["1", ">50K", "True", "pos"]:
                if key in rep and isinstance(rep[key], dict) and "recall" in rep[key]:
                    sup_rec = rep[key]["recall"]
                    break

    # no supervisado
    uns = summary.get("unsupervised", {}) or {}
    cl_k = uns.get("k") or uns.get("n_clusters") or uns.get("K")
    cl_sil = uns.get("silhouette") or uns.get("silhouette_score")
    
    # contraste hipótesis
    hyp = summary.get("hypothesis_test", {}) or summary.get("hypothesis", {}) or {}
    hyp_var = hyp.get("variable") or hyp.get("feature") or "hours_per_week"
    hyp_mean0 = hyp.get("mean_group0") or hyp.get("mean_<=50K") or hyp.get("mean0")
    hyp_mean1 = hyp.get("mean_group1") or hyp.get("mean_>50K") or hyp.get("mean1")
    ci = hyp.get("bootstrap_ci") or hyp.get("ci95") or {}
    hyp_ci_lo = ci.get("low") or ci.get("lo") or ci.get("lower")
    hyp_ci_hi = ci.get("high") or ci.get("hi") or ci.get("upper")
    
    # Defaults razonables para evitar NameError si faltan claves
    sup_auc = float(sup_auc) if sup_auc is not None else float("nan")
    sup_acc = float(sup_acc) if sup_acc is not None else float("nan")
    sup_rec = float(sup_rec) if sup_rec is not None else float("nan")
    cl_k = int(cl_k) if cl_k is not None else None
    cl_sil = float(cl_sil) if cl_sil is not None else float("nan")
    hyp_mean0 = float(hyp_mean0) if hyp_mean0 is not None else float("nan")
    hyp_mean1 = float(hyp_mean1) if hyp_mean1 is not None else float("nan")
    hyp_ci_lo = float(hyp_ci_lo) if hyp_ci_lo is not None else float("nan")
    hyp_ci_hi = float(hyp_ci_hi) if hyp_ci_hi is not None else float("nan")


    missing_before = pd.read_csv(tables_dir / "missing_before.csv", index_col=0)
    missing_after = pd.read_csv(tables_dir / "missing_after.csv", index_col=0)
    winsor = pd.read_csv(tables_dir / "winsor_caps.csv")

    semantic_before = _safe_read_csv(
        tables_dir / "semantic_missing_before.csv", index_col=0
    )
    semantic_after = _safe_read_csv(
        tables_dir / "semantic_missing_after.csv", index_col=0
    )

    cls_rep = _safe_read_csv(tables_dir / "classification_report.csv", index_col=0)

    dist = summary["class_distribution"]
    counts = dist.get("counts", {})
    pct = dist.get("pct", {})

    # En JSON, las keys de dict se serializan como strings.
    c0 = int(counts.get("0", 0))
    c1 = int(counts.get("1", 0))
    p0 = float(pct.get("0", 0.0))
    p1 = float(pct.get("1", 0.0))
    total = int(dist.get("total", c0 + c1))

    sup = summary["supervised"]
    hyp = summary["hypothesis_test"]
    clu = summary["clustering"]

    # Formateo p-value (evitar "0.0" por underflow)
    pval = float(hyp.get("p_value", 1.0))
    pval_txt = "< 1e-300" if pval == 0.0 else f"{pval:.3e}"

    today = date.today().isoformat()

    # Baseline: predecir siempre clase mayoritaria
    maj_acc = max(p0, p1)

    md: List[str] = []
    md.append("# Práctica 2 — Análisis del dataset Adult Income (Python)")
    md.append("")
    md.append(_members_block(cfg))
    md.append(f"Fecha de generación: **{today}**")
    md.append("")

# 1
    md.append("## 1. Descripción del dataset")
    
    md.append(
        "Trabajamos con el dataset **Adult Income** (UCI), cuyo objetivo es "
        "analizar qué variables socio-demográficas y laborales se asocian con "
        "la probabilidad de percibir ingresos **>50K**."
    )
    md.append("")
    
    md.append(
        f"El dataset integrado contiene **{total:,}** registros. "
        "La variable objetivo es `income` (<=50K vs >50K)."
    )
    md.append(
        f"Distribución de clases: `<=50K` = **{c0:,}** ({_pct(p0)}), "
        f"`>50K` = **{c1:,}** ({_pct(p1)})."
    )
    md.append("")
    
    md.append(
        "Este dataset resulta especialmente adecuado para un análisis estadístico "
        "y de ciencia de datos porque combina **variables numéricas y categóricas**, "
        "presenta **valores faltantes semánticos** (p. ej. `?`) y contiene "
        "**valores extremos** en variables financieras (p. ej. `capital_gain`, `capital_loss`). "
        "Estas características permiten aplicar de forma natural técnicas de integración, "
        "limpieza, validación, análisis supervisado y no supervisado, además de contrastes de hipótesis."
    )
    md.append("")
    
    md.append("**Estructura de variables (resumen):**")
    md.append(
        "- **Numéricas**: `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`.\n"
        "- **Categóricas**: `workclass`, `education`, `marital_status`, `occupation`, "
        "`relationship`, `race`, `sex`, `native_country`.\n"
        "- **Objetivo**: `income`."
    )
    md.append("")
    
    md.append(
        "Dado el desbalance aproximado 3:1, además de la accuracy se reportan "
        "métricas por clase (precision/recall/F1) y AUC."
    )
    md.append("")

    md.append("")
    md.append("**Alcance del análisis**: el objetivo del trabajo es **descriptivo y predictivo**, no causal. "
              "Por tanto, las asociaciones observadas no deben interpretarse como relaciones causa–efecto.")
    md.append("")
    md.append("**Uso del análisis no supervisado**: las técnicas no supervisadas se emplean con fines **exploratorios**, "
              "para identificar patrones y estructura potencial en los datos, sin asumir grupos “reales” o interpretables a priori.")
    md.append("")


    # 2
    md.append("## 2. Integración y selección de los datos")
    md.append(
        "Se integran los conjuntos train y test del Adult Income y se conservan "
        "las variables estándar del dominio (edad, educación, horas, "
        "capital_gain/capital_loss y categóricas de contexto)."
    )
    md.append("")

    # 3
    md.append("## 3. Limpieza de los datos")

    md.append("### 3.1 Faltantes y/o valores perdidos")
    md.append(
        "Faltantes reales (NaN) antes de la limpieza (top 5 por columna):"
    )
    md.append("")
    md.append(
        missing_before.sort_values("missing_count", ascending=False)
        .head(5)
        .reset_index()
        .to_markdown(index=False)
    )
    md.append("")

    if semantic_before is not None:
        md.append(
            "Faltantes semánticos antes de la limpieza (incluye '?', vacío "
            "y equivalentes) (top 5):"
        )
        md.append("")
        md.append(
            semantic_before.sort_values("missing_count", ascending=False)
            .head(5)
            .reset_index()
            .to_markdown(index=False)
        )
        md.append("")

        # Observaciones
    md.append("**Observaciones:**")
    
    if semantic_before is not None and len(semantic_before) > 0:
        df_sem = semantic_before.copy()
    
        # Detectar el nombre de la columna que contiene el nombre de variable
        col_name = None
        for candidate in ["col", "column", "variable", "feature", "attribute"]:
            if candidate in df_sem.columns:
                col_name = candidate
                break
        if col_name is None:
            # fallback: primera columna
            col_name = df_sem.columns[0]
    
        # Detectar columnas de conteo y porcentaje
        count_name = None
        for candidate in ["missing_count", "count", "n_missing", "missing", "num_missing"]:
            if candidate in df_sem.columns:
                count_name = candidate
                break
    
        pct_name = None
        for candidate in ["missing_pct", "pct", "percent", "missing_percent", "percentage"]:
            if candidate in df_sem.columns:
                pct_name = candidate
                break
    
        # Ordenar por conteo si existe, si no por porcentaje, si no por col
        if count_name:
            top_sem = df_sem.sort_values(count_name, ascending=False).head(3)
        elif pct_name:
            top_sem = df_sem.sort_values(pct_name, ascending=False).head(3)
        else:
            top_sem = df_sem.head(3)
    
        rows = []
        for _, r in top_sem.iterrows():
            var = r[col_name]
    
            if count_name:
                nmiss = int(r[count_name])
            else:
                nmiss = None
    
            if pct_name:
                # puede venir como 5.7 (porcentaje) o 0.057 (proporción)
                raw = float(r[pct_name])
                pct = raw / 100.0 if raw > 1 else raw
            else:
                pct = None
    
            if (nmiss is not None) and (pct is not None):
                rows.append(f"- La variable `{var}` concentra faltantes semánticos: **{nmiss}** registros (**{_pct(pct)}** aprox.).")
            elif (nmiss is not None):
                rows.append(f"- La variable `{var}` concentra faltantes semánticos: **{nmiss}** registros.")
            elif (pct is not None):
                rows.append(f"- La variable `{var}` concentra faltantes semánticos: **{_pct(pct)}** aprox.")
            else:
                rows.append(f"- Se observan faltantes semánticos en `{var}`.")
    
        md.extend(rows)
        md.append("- Este patrón sugiere que la ausencia de información no es uniforme y debe tratarse explícitamente para evitar sesgos.")
    else:
        md.append("- Los faltantes semánticos son bajos o están concentrados en pocas variables; se aplica imputación para garantizar consistencia del análisis.")
    
    md.append("")


    # 4
    md.append("## 4. Análisis y métricas")

    md.append("### 4.1 Supervisado y no supervisado")
    md.append(
        f"**Modelo supervisado (Regresión logística):** ROC-AUC = "
        f"**{sup['roc_auc']:.4f}**, Accuracy = **{sup['accuracy']:.4f}**."
    )
    md.append(
        "Baseline (predecir siempre la clase mayoritaria): "
        f"**{maj_acc:.4f}**."
    )
    md.append(
        "Para la clase `>50K` (positiva): "
        f"Precision = **{sup['precision_pos']:.3f}**, "
        f"Recall = **{sup['recall_pos']:.3f}**, "
        f"F1 = **{sup['f1_pos']:.3f}**."
    )
    md.append(
        "Interpretación: AUC alto indica buena discriminación; el recall "
        "moderado sugiere que el modelo pierde parte de los casos `>50K`, "
        "fenómeno consistente con el desbalance."
    )
    md.append(
        "Matriz de confusión (test): "
        f"TN={sup['tn']}, FP={sup['fp']}, FN={sup['fn']}, TP={sup['tp']}."
    )
    md.append("")

    md.append(
        f"**No supervisado (PCA+KMeans):** muestra n = **{clu['sample_n']}**, "
        f"k = **{clu['k']}**, silhouette = **{clu['silhouette']:.4f}**."
    )
    md.append(
        "Interpretación: el clustering es exploratorio y depende del muestreo; "
        "no se extraen conclusiones predictivas fuertes sin validación de "
        "estabilidad."
    )
    md.append("")

    md.append("### 4.2 Contraste de hipótesis")
    md.append(
        "Contraste entre grupos de `income` sobre `hours_per_week` usando "
        f"**{hyp['test']}** (prueba no paramétrica, no requiere normalidad)."
    )
    md.append(
        f"Medias: <=50K = **{hyp['mean0']:.2f}**, >50K = **{hyp['mean1']:.2f}**. "
        f"Medianas: <=50K = **{hyp['median0']:.2f}**, >50K = **{hyp['median1']:.2f}**."
    )
    md.append(f"p-value = **{pval_txt}**.")
    if "ci_mean_diff_lo" in hyp and "ci_mean_diff_hi" in hyp:
        md.append(
            "IC 95% (bootstrap) para la diferencia de **medias** "
            f"(>50K − <=50K): **[{hyp['ci_mean_diff_lo']:.2f}, {hyp['ci_mean_diff_hi']:.2f}]**."
        )
    md.append(
        "Interpretación: evidencia estadística fuerte de diferencias entre grupos; "
        "esto indica asociación, no causalidad."
    )
    md.append("")

    # 5
    md.append("## 5. Representación de resultados")

    # 5.1 Preview del dataset limpio (estratificada)
    md.append("### 5.1 Vista previa del dataset limpio")
    clean_path = processed_dir / "adult_clean.csv"
    df_clean = pd.read_csv(clean_path)

    preview_cols = [
        "age",
        "workclass",
        "education",
        "hours_per_week",
        "capital_gain",
        "capital_loss",
        "income",
    ]
    preview_cols = [c for c in preview_cols if c in df_clean.columns]

    df_clean["income"] = df_clean["income"].astype("string").str.strip()
    df_lo = df_clean[df_clean["income"] == "<=50K"].head(3)
    df_hi = df_clean[df_clean["income"] == ">50K"].head(2)
    preview_df = pd.concat([df_lo, df_hi], ignore_index=True)

    if len(preview_df) < 5:
        preview_df = pd.concat(
            [preview_df, df_clean.head(5 - len(preview_df))],
            ignore_index=True,
        )

    md.append(
        "Muestra estratificada (3 filas de `<=50K` y 2 filas de `>50K`):"
    )
    md.append("")
    md.append(preview_df[preview_cols].to_markdown(index=False))
    md.append("")

    # 5.2 Tabla de métricas supervisadas (classification report)
    md.append("### 5.2 Métricas del modelo supervisado")
    if cls_rep is not None:
        # Mostrar filas más relevantes
        show_rows = ["0", "1", "macro avg", "weighted avg"]
        cls_df = cls_rep.loc[[r for r in show_rows if r in cls_rep.index], :]
        # Limitar columnas para legibilidad
        cols_keep = [c for c in ["precision", "recall", "f1-score", "support"] if c in cls_df.columns]
        if cols_keep:
            cls_df = cls_df[cols_keep]
        md.append("Tabla resumida (precision/recall/F1/support):")
        md.append("")
        md.append(cls_df.to_markdown())
        md.append("")
    else:
        md.append(
            "No se encontró `reports/tables/classification_report.csv` "
            "(verificar ejecución del pipeline)."
        )
        md.append("")

    # 5.3 Figuras
    md.append("### 5.3 Gráficos generados")
    md.append("Se incluyen las figuras principales del análisis:")
    md.append("")
    md.append("**ROC Curve**")
    md.append("")
    md.append("![ROC Curve](figures/roc_curve.png)")
    md.append("")
    md.append("**Matriz de confusión**")
    md.append("")
    md.append("![Confusion Matrix](figures/confusion_matrix.png)")
    md.append("")

    # 6
    md.append("## 6. Conclusiones")
    
    md.append(
        "A partir del proceso de limpieza y del análisis posterior, se obtienen las siguientes conclusiones principales:"
    )
    md.append("")
    
    md.append("- **Calidad del dato y limpieza**: la presencia de faltantes semánticos (`?`) se concentra en variables específicas (p. ej., `occupation`, `workclass`, `native_country`), por lo que tratarlas explícitamente mejora la consistencia del análisis y evita perder filas.")
    md.append("- **Valores extremos**: variables como `capital_gain` y `capital_loss` presentan colas largas; la winsorización permite estabilizar el análisis sin eliminar observaciones.")
    md.append("")
    
    # Conclusiones supervisado
    md.append(
        f"- **Modelo supervisado**: el clasificador logra un desempeño global sólido "
        f"(ROC-AUC = **{sup_auc:.4f}**, accuracy = **{sup_acc:.4f}**), superando claramente el baseline de clase mayoritaria. "
        f"Sin embargo, la recuperación de la clase `>50K` (recall = **{sup_rec:.3f}**) es moderada, coherente con el desbalance."
    )
    
    # Conclusiones contraste
    md.append(
        f"- **Contraste de hipótesis**: se observan diferencias consistentes entre grupos en `{hyp_var}`. "
        f"La diferencia de medias estimada es aproximadamente **{hyp_mean1 - hyp_mean0:.2f}** horas/semana "
        f"(IC 95% bootstrap: **[{hyp_ci_lo:.2f}, {hyp_ci_hi:.2f}]**), con evidencia estadística muy fuerte."
    )
    
    # Conclusiones no supervisado
    md.append(
        f"- **Modelo no supervisado (exploratorio)**: con PCA + KMeans (k={cl_k}) se obtiene un silhouette ≈ **{cl_sil:.3f}**, "
        "lo que sugiere cierta separación estructural en los datos, sin implicar necesariamente grupos “reales” o interpretables."
    )
    
    md.append("")
    md.append(
        "**Limitaciones**: este análisis es observacional; los resultados describen asociaciones y capacidad predictiva, "
        "pero no permiten afirmar causalidad. El clustering se interpreta como exploratorio."
    )
    md.append("")

    # 7
    md.append("## 7. Código")
    md.append(
        "El código fuente se encuentra en `src/`. Para ejecutar el pipeline: "
        "`python -m src.run_all`."
    )
    md.append("")

    # 8
    md.append("## 8. Vídeo")
    md.append(
        f"Enlace al vídeo (Google Drive UOC): {cfg.get('video_url', 'PENDIENTE')}"
    )
    md.append("")

    md.append("## Tabla de contribuciones")
    md.append(_contrib_table(cfg))

    out_md = reports_dir / "report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[build_report] Wrote: {out_md}")
    return out_md


if __name__ == "__main__":
    main()
