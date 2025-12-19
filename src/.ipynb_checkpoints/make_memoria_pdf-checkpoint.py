"""
make_memoria_pdf.py

Convierte reports/report.md a un PDF simple (memoria) usando reportlab.

Soporta:
- Títulos (#, ##, ###) como encabezados
- Tablas Markdown (líneas que comienzan con '|') como texto monoespaciado con wrap
- Imágenes Markdown: ![alt](ruta.png) insertadas en el PDF
- Párrafos como texto normal

Salida:
- reports/memoria.pdf
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .utils import ensure_dirs


def _wrap_line(text: str, max_width: float, font: str, size: int) -> List[str]:
    """
    Divide una línea en múltiples líneas según ancho máximo.
    """
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = words[0]
    for w in words[1:]:
        candidate = f"{current} {w}"
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def _wrap_preserving_pipes(line: str, max_width: float, font: str, size: int) -> List[str]:
    """
    Wrap para líneas de tabla Markdown sin romper de forma agresiva la estructura.
    Estrategia: wrap por ancho, pero si excede y no hay espacios suficientes,
    fuerza corte por longitud aproximada.
    """
    if stringWidth(line, font, size) <= max_width:
        return [line]

    # Intentar wrap por espacios primero
    if " " in line.strip():
        parts = _wrap_line(line, max_width, font, size)
        return parts

    # Fallback: corte por caracteres aproximado
    avg_char_w = stringWidth("X", font, size) or 6.0
    max_chars = max(20, int(max_width / avg_char_w))
    out = [line[i:i + max_chars] for i in range(0, len(line), max_chars)]
    return out


def main() -> Path:
    """
    Genera reports/memoria.pdf desde reports/report.md.
    """
    _, _, reports_dir, _ = ensure_dirs()
    md_path = reports_dir / "report.md"
    pdf_path = reports_dir / "memoria.pdf"

    lines = md_path.read_text(encoding="utf-8").splitlines()

    canvas = Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin_x = 2.0 * cm
    margin_y = 2.0 * cm
    y = height - margin_y

    img_re = re.compile(r"!\[.*?\]\((.*?)\)")

    def new_page() -> None:
        nonlocal y
        canvas.showPage()
        y = height - margin_y

    def draw_text(text: str, font: str, size: int, leading: int) -> None:
        nonlocal y
        canvas.setFont(font, size)
        max_w = width - 2 * margin_x
        wrapped_lines = _wrap_line(text, max_w, font, size)
        for wline in wrapped_lines:
            if y < margin_y + leading:
                new_page()
                canvas.setFont(font, size)
            canvas.drawString(margin_x, y, wline)
            y -= leading

    def draw_table_line(text: str) -> None:
        """
        Dibuja una línea de tabla Markdown en monoespaciado con wrap.
        """
        nonlocal y
        font = "Courier"
        size = 7
        leading = 10
        canvas.setFont(font, size)
        max_w = width - 2 * margin_x

        for wline in _wrap_preserving_pipes(text, max_w, font, size):
            if y < margin_y + leading:
                new_page()
                canvas.setFont(font, size)
            canvas.drawString(margin_x, y, wline)
            y -= leading

    def draw_image(rel_path: str) -> None:
        """
        Inserta una imagen referenciada desde el Markdown.
        rel_path se interpreta relativo a reports/.
        """
        nonlocal y
        img_path = (reports_dir / rel_path).resolve()

        if not img_path.exists():
            draw_text(f"[Imagen no encontrada: {rel_path}]", "Helvetica", 9, 12)
            return

        max_w = width - 2 * margin_x
        img = ImageReader(str(img_path))
        iw, ih = img.getSize()

        scale = max_w / float(iw)
        draw_w = max_w
        draw_h = float(ih) * scale

        if y - draw_h < margin_y:
            new_page()

        canvas.drawImage(
            img,
            margin_x,
            y - draw_h,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= (draw_h + 12)

    for line in lines:
        # Render de imágenes Markdown: ![alt](figures/x.png)
        m = img_re.search(line.strip())
        if m:
            rel = m.group(1).strip()
            draw_image(rel)
            continue

        # Encabezados
        if line.startswith("# "):
            draw_text(line[2:].strip(), "Helvetica-Bold", 16, 20)
            y -= 6
            continue
        if line.startswith("## "):
            draw_text(line[3:].strip(), "Helvetica-Bold", 13, 18)
            y -= 4
            continue
        if line.startswith("### "):
            draw_text(line[4:].strip(), "Helvetica-Bold", 11, 16)
            y -= 2
            continue

        # Tablas Markdown (líneas que empiezan con '|')
        if line.strip().startswith("|"):
            draw_table_line(line)
            continue

        # Saltos de línea
        if not line.strip():
            y -= 10
            if y < margin_y:
                new_page()
            continue

        # Texto normal
        draw_text(line, "Helvetica", 10, 14)

    canvas.save()
    print(f"[make_memoria_pdf] Wrote: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    main()
