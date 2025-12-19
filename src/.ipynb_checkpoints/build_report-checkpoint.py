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
    _, _, reports_dir, tables_dir = ensure_dirs()
    cfg = load_project_config()

    summary_path = reports_dir / "summary.json"
    summary = _load_summary(summary_path)

    missing_before = pd.read_csv(tables_dir / "missing_before.csv", index_col=0)
    missing_after = pd.read_csv(tables_dir / "missing_after.csv", index_col=0)
    winsor = pd.read_csv(tables_dir / "winsor_caps.csv")

    semantic_before = _safe_read_csv(
        tables_dir / "semantic_missing_before.csv", index_col=0
    )
    semantic_after = _safe_read_csv(
        tables_dir / "semantic_missing_after.csv", index_col=0
    )

    cls_rep = _safe_read_csv(tables_dir / "classification_report.csv")

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
        f"El dataset integrado contiene **{total:,}** registros. "
        "La variable objetivo es `income` (<=50K vs >50K)."
    )
    md.append(
        f"Distribución de clases: `<=50K` = **{c0:,}** ({_pct(p0)}), "
        f"`>50K` = **{c1:,}** ({_pct(p1)})."
    )
    md.append(
        "Se observa desbalance aproximado 3:1. Por tanto, además de la "
        "accuracy se reportan métricas por clase (precision/recall/F1) y AUC."
    )
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

    md.append(
        "Tratamiento aplicado: categóricas imputadas como `Unknown` y numéricas "
        "imputadas con mediana."
    )
    md.append("")

    md.append("### 3.2 Tipos de variables y transformaciones")
    md.append(
        "Se normalizan categóricas (strip) y se tipifican numéricas con "
        "coerción segura (valores inválidos pasan a NA y se imputan)."
    )
    md.append("")

    md.append("### 3.3 Tratamiento de valores extremos")
    md.append(
        "Para `capital_gain` y `capital_loss` se aplica winsorización al "
        "percentil 99.5% para limitar el impacto de colas extremas en modelos "
        "lineales y métricas."
    )
    md.append("")
    md.append(winsor.to_markdown(index=False))
    md.append("")

    md.append("### 3.4 Consideraciones adicionales")
    md.append(
        "Se preserva el tamaño muestral evitando eliminar filas con faltantes; "
        "esto reduce riesgo de sesgo por eliminación y mantiene potencia "
        "estadística."
    )
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
    md.append(
        "Interpretación: evidencia estadística fuerte de diferencias entre "
        "grupos; esto indica asociación, no causalidad."
    )
    md.append("")

    # 5
    md.append("## 5. Representación de resultados")
    md.append(
        "Figuras generadas en `reports/figures/`: `roc_curve.png` y "
        "`confusion_matrix.png`. Tablas en `reports/tables/`."
    )
    md.append("")

    # 6
    md.append("## 6. Conclusiones")
    md.append(
        "El dataset permite construir un clasificador con buen desempeño "
        "(AUC alto) frente al baseline, aunque la recuperación de la clase "
        "`>50K` es moderada por el desbalance. El contraste sugiere diferencias "
        "consistentes en horas trabajadas entre grupos. El análisis no "
        "supervisado se interpreta como exploratorio."
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
