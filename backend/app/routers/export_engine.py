"""Export engine — generates publication-ready APA tables, full reports,
and LaTeX equations from statistical results."""

from __future__ import annotations

import csv
import io
import math
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val: float, decimals: int = 2) -> str:
    """Format number; strip leading zero for values bounded by +/-1."""
    s = f"{val:.{decimals}f}"
    if -1 < val < 1 and val != 0:
        s = s.lstrip("0").lstrip("-0")
        if val < 0:
            s = "-" + s
    return s


def _p_fmt(p: float) -> str:
    if p < 0.001:
        return "< .001"
    return _fmt(p, 3)


def _p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class APATableRequest(BaseModel):
    table_type: str  # "descriptive", "correlation", "regression", "anova", "t_test", "frequencies"
    data: dict       # varies by table_type
    title: str | None = None
    note: str | None = None


class APATableResponse(BaseModel):
    html: str
    latex: str
    markdown: str
    csv: str
    title: str
    note: str


class ReportSection(BaseModel):
    heading: str
    content: str
    results: dict | None = None
    apa_text: str | None = None
    table: dict | None = None


class FullReportRequest(BaseModel):
    title: str
    author: str | None = None
    sections: list[ReportSection]
    format: str = "html"    # "html", "latex", "markdown"
    style: str = "apa7"     # "apa7", "custom"


class FullReportResponse(BaseModel):
    content: str
    word_count: int
    tables_count: int
    figures_referenced: int


class LatexEquationRequest(BaseModel):
    model_type: str  # "regression", "hlm", "sem", "anova", "mediation"
    parameters: dict  # model-specific


class LatexEquationResponse(BaseModel):
    equation: str
    description: str


# ---------------------------------------------------------------------------
# APA table builders
# ---------------------------------------------------------------------------

_APA_CSS = (
    "font-family: 'Times New Roman', Times, serif; border-collapse: collapse; "
    "width: 100%; font-size: 12pt;"
)
_TH_CSS = "border-top: 2px solid black; border-bottom: 1px solid black; padding: 4px 8px; text-align: center;"
_TD_CSS = "padding: 4px 8px; text-align: center;"
_BOTTOM_CSS = "border-bottom: 2px solid black;"


def _build_descriptive_table(data: dict, title: str, note: str) -> APATableResponse:
    """Descriptive statistics table: M, SD, N, Min, Max per variable."""
    variables: list[dict] = data.get("variables", [])
    if not variables:
        raise HTTPException(status_code=400, detail="data.variables required for descriptive table")

    # ── HTML ──
    rows_html = ""
    for v in variables:
        rows_html += (
            f"<tr>"
            f"<td style=\"{_TD_CSS} text-align:left;\">{v['name']}</td>"
            f"<td style=\"{_TD_CSS}\">{v.get('n', '')}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(v['mean'])}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(v['sd'])}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(v.get('min', 0))}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(v.get('max', 0))}</td>"
            f"</tr>\n"
        )
    html = (
        f"<table style=\"{_APA_CSS}\">\n"
        f"<caption style=\"text-align:left; font-style:italic;\">{title}</caption>\n"
        f"<thead><tr>"
        f"<th style=\"{_TH_CSS} text-align:left;\">Variable</th>"
        f"<th style=\"{_TH_CSS}\"><em>N</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>M</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>SD</em></th>"
        f"<th style=\"{_TH_CSS}\">Min</th>"
        f"<th style=\"{_TH_CSS}\">Max</th>"
        f"</tr></thead>\n<tbody>\n{rows_html}</tbody>\n"
        f"<tfoot><tr><td colspan=\"6\" style=\"{_BOTTOM_CSS} text-align:left; font-size:10pt;\">"
        f"<em>Note.</em> {note}</td></tr></tfoot>\n</table>"
    )

    # ── Markdown ──
    md_lines = [
        f"*{title}*\n",
        "| Variable | *N* | *M* | *SD* | Min | Max |",
        "|:---------|----:|----:|-----:|----:|----:|",
    ]
    for v in variables:
        md_lines.append(
            f"| {v['name']} | {v.get('n', '')} | {_fmt(v['mean'])} | "
            f"{_fmt(v['sd'])} | {_fmt(v.get('min', 0))} | {_fmt(v.get('max', 0))} |"
        )
    md_lines.append(f"\n*Note.* {note}")
    markdown = "\n".join(md_lines)

    # ── LaTeX ──
    latex_rows = ""
    for v in variables:
        latex_rows += (
            f"  {v['name']} & {v.get('n', '')} & {_fmt(v['mean'])} & "
            f"{_fmt(v['sd'])} & {_fmt(v.get('min', 0))} & {_fmt(v.get('max', 0))} \\\\\n"
        )
    latex = (
        "\\begin{table}[ht]\n\\caption{" + title + "}\n"
        "\\begin{tabular}{l r r r r r}\n\\toprule\n"
        "Variable & $N$ & $M$ & $SD$ & Min & Max \\\\\n\\midrule\n"
        + latex_rows +
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n\\item \\textit{Note.} " + note +
        "\n\\end{tablenotes}\n\\end{table}"
    )

    # ── CSV ──
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Variable", "N", "M", "SD", "Min", "Max"])
    for v in variables:
        writer.writerow([v["name"], v.get("n", ""), _fmt(v["mean"]),
                         _fmt(v["sd"]), _fmt(v.get("min", 0)), _fmt(v.get("max", 0))])
    csv_str = buf.getvalue()

    return APATableResponse(html=html, latex=latex, markdown=markdown,
                            csv=csv_str, title=title, note=note)


def _build_correlation_table(data: dict, title: str, note: str) -> APATableResponse:
    """Correlation matrix — lower triangle with significance stars."""
    variables: list[str] = data.get("variables", [])
    matrix: list[list[float]] = data.get("matrix", [])
    p_matrix: list[list[float]] = data.get("p_matrix", [])
    means: list[float] = data.get("means", [])
    sds: list[float] = data.get("sds", [])

    if not variables or not matrix:
        raise HTTPException(status_code=400, detail="data.variables and data.matrix required")

    n = len(variables)

    # ── HTML ──
    header_cells = "".join(f"<th style=\"{_TH_CSS}\">{i+1}</th>" for i in range(n))
    extra_hdr = ""
    if means:
        extra_hdr = f"<th style=\"{_TH_CSS}\"><em>M</em></th><th style=\"{_TH_CSS}\"><em>SD</em></th>"
    rows_html = ""
    for i in range(n):
        cells = f"<td style=\"{_TD_CSS} text-align:left;\">{i+1}. {variables[i]}</td>"
        for j in range(n):
            if j > i:
                cells += f"<td style=\"{_TD_CSS}\"> </td>"
            elif j == i:
                cells += f"<td style=\"{_TD_CSS}\">&mdash;</td>"
            else:
                r_val = matrix[i][j]
                p_val = p_matrix[i][j] if p_matrix else 1.0
                cells += f"<td style=\"{_TD_CSS}\">{_fmt(r_val)}{_p_stars(p_val)}</td>"
        if means:
            cells += f"<td style=\"{_TD_CSS}\">{_fmt(means[i])}</td>"
            cells += f"<td style=\"{_TD_CSS}\">{_fmt(sds[i])}</td>"
        rows_html += f"<tr>{cells}</tr>\n"

    html = (
        f"<table style=\"{_APA_CSS}\">\n"
        f"<caption style=\"text-align:left; font-style:italic;\">{title}</caption>\n"
        f"<thead><tr><th style=\"{_TH_CSS} text-align:left;\">Variable</th>"
        f"{header_cells}{extra_hdr}</tr></thead>\n"
        f"<tbody>\n{rows_html}</tbody>\n"
        f"<tfoot><tr><td colspan=\"{n + 1 + (2 if means else 0)}\" "
        f"style=\"{_BOTTOM_CSS} text-align:left; font-size:10pt;\">"
        f"<em>Note.</em> {note}</td></tr></tfoot>\n</table>"
    )

    # ── Markdown ──
    hdr = "| Variable | " + " | ".join(str(i+1) for i in range(n))
    if means:
        hdr += " | *M* | *SD*"
    hdr += " |"
    sep = "|:---------|" + "|".join("----:" for _ in range(n))
    if means:
        sep += "|----:|----:"
    sep += "|"
    md_rows = [f"*{title}*\n", hdr, sep]
    for i in range(n):
        row = f"| {i+1}. {variables[i]} |"
        for j in range(n):
            if j > i:
                row += "   |"
            elif j == i:
                row += " --- |"
            else:
                r_val = matrix[i][j]
                p_val = p_matrix[i][j] if p_matrix else 1.0
                row += f" {_fmt(r_val)}{_p_stars(p_val)} |"
        if means:
            row += f" {_fmt(means[i])} | {_fmt(sds[i])} |"
        md_rows.append(row)
    md_rows.append(f"\n*Note.* {note}")
    markdown = "\n".join(md_rows)

    # ── LaTeX ──
    col_spec = "l " + " ".join(["r"] * n)
    if means:
        col_spec += " r r"
    lhdr = "Variable & " + " & ".join(str(i+1) for i in range(n))
    if means:
        lhdr += " & $M$ & $SD$"
    lhdr += " \\\\"
    lrows = ""
    for i in range(n):
        lr = f"  {i+1}. {variables[i]}"
        for j in range(n):
            if j > i:
                lr += " & "
            elif j == i:
                lr += " & ---"
            else:
                r_val = matrix[i][j]
                p_val = p_matrix[i][j] if p_matrix else 1.0
                lr += f" & {_fmt(r_val)}{_p_stars(p_val)}"
        if means:
            lr += f" & {_fmt(means[i])} & {_fmt(sds[i])}"
        lr += " \\\\\n"
        lrows += lr
    latex = (
        "\\begin{table}[ht]\n\\caption{" + title + "}\n"
        "\\begin{tabular}{" + col_spec + "}\n\\toprule\n"
        + lhdr + "\n\\midrule\n" + lrows +
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n\\item \\textit{Note.} " + note +
        "\n\\end{tablenotes}\n\\end{table}"
    )

    # ── CSV ──
    buf = io.StringIO()
    writer = csv.writer(buf)
    header_row = ["Variable"] + [str(i+1) for i in range(n)]
    if means:
        header_row += ["M", "SD"]
    writer.writerow(header_row)
    for i in range(n):
        row_data = [f"{i+1}. {variables[i]}"]
        for j in range(n):
            if j > i:
                row_data.append("")
            elif j == i:
                row_data.append("---")
            else:
                r_val = matrix[i][j]
                p_val = p_matrix[i][j] if p_matrix else 1.0
                row_data.append(f"{_fmt(r_val)}{_p_stars(p_val)}")
        if means:
            row_data += [_fmt(means[i]), _fmt(sds[i])]
        writer.writerow(row_data)
    csv_str = buf.getvalue()

    return APATableResponse(html=html, latex=latex, markdown=markdown,
                            csv=csv_str, title=title, note=note)


def _build_regression_table(data: dict, title: str, note: str) -> APATableResponse:
    """Regression results: B, SE, beta, t, p, 95% CI per predictor."""
    predictors: list[dict] = data.get("predictors", [])
    model_stats: dict = data.get("model_stats", {})

    if not predictors:
        raise HTTPException(status_code=400, detail="data.predictors required for regression table")

    # ── HTML ──
    rows_html = ""
    for p in predictors:
        ci = p.get("ci", [None, None])
        ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else ""
        rows_html += (
            f"<tr>"
            f"<td style=\"{_TD_CSS} text-align:left;\">{p['name']}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(p['B'])}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(p['SE'])}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(p.get('beta', 0))}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(p['t'])}</td>"
            f"<td style=\"{_TD_CSS}\">{_p_fmt(p['p'])}</td>"
            f"<td style=\"{_TD_CSS}\">{ci_str}</td>"
            f"</tr>\n"
        )
    model_note = note
    if model_stats:
        r2 = model_stats.get("R2", "")
        adj_r2 = model_stats.get("adj_R2", "")
        f_val = model_stats.get("F", "")
        f_p = model_stats.get("F_p", "")
        model_note += (
            f" R\\u00B2 = {_fmt(r2) if isinstance(r2, (int, float)) else r2}, "
            f"Adjusted R\\u00B2 = {_fmt(adj_r2) if isinstance(adj_r2, (int, float)) else adj_r2}, "
            f"F = {_fmt(f_val) if isinstance(f_val, (int, float)) else f_val}, "
            f"p {_p_fmt(f_p) if isinstance(f_p, (int, float)) else f_p}."
        )
    html = (
        f"<table style=\"{_APA_CSS}\">\n"
        f"<caption style=\"text-align:left; font-style:italic;\">{title}</caption>\n"
        f"<thead><tr>"
        f"<th style=\"{_TH_CSS} text-align:left;\">Predictor</th>"
        f"<th style=\"{_TH_CSS}\"><em>B</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>SE</em></th>"
        f"<th style=\"{_TH_CSS}\">&beta;</th>"
        f"<th style=\"{_TH_CSS}\"><em>t</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>p</em></th>"
        f"<th style=\"{_TH_CSS}\">95% CI</th>"
        f"</tr></thead>\n<tbody>\n{rows_html}</tbody>\n"
        f"<tfoot><tr><td colspan=\"7\" style=\"{_BOTTOM_CSS} text-align:left; font-size:10pt;\">"
        f"<em>Note.</em> {model_note}</td></tr></tfoot>\n</table>"
    )

    # ── Markdown ──
    md_lines = [
        f"*{title}*\n",
        "| Predictor | *B* | *SE* | *beta* | *t* | *p* | 95% CI |",
        "|:----------|----:|-----:|-------:|----:|----:|-------:|",
    ]
    for p in predictors:
        ci = p.get("ci", [None, None])
        ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else ""
        md_lines.append(
            f"| {p['name']} | {_fmt(p['B'])} | {_fmt(p['SE'])} | "
            f"{_fmt(p.get('beta', 0))} | {_fmt(p['t'])} | {_p_fmt(p['p'])} | {ci_str} |"
        )
    md_lines.append(f"\n*Note.* {model_note}")
    markdown = "\n".join(md_lines)

    # ── LaTeX ──
    lrows = ""
    for p in predictors:
        ci = p.get("ci", [None, None])
        ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else ""
        lrows += (
            f"  {p['name']} & {_fmt(p['B'])} & {_fmt(p['SE'])} & "
            f"{_fmt(p.get('beta', 0))} & {_fmt(p['t'])} & {_p_fmt(p['p'])} & {ci_str} \\\\\n"
        )
    latex = (
        "\\begin{table}[ht]\n\\caption{" + title + "}\n"
        "\\begin{tabular}{l r r r r r l}\n\\toprule\n"
        "Predictor & $B$ & $SE$ & $\\beta$ & $t$ & $p$ & 95\\% CI \\\\\n\\midrule\n"
        + lrows +
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n\\item \\textit{Note.} " + model_note +
        "\n\\end{tablenotes}\n\\end{table}"
    )

    # ── CSV ──
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Predictor", "B", "SE", "beta", "t", "p", "CI_lower", "CI_upper"])
    for p in predictors:
        ci = p.get("ci", [None, None])
        writer.writerow([p["name"], _fmt(p["B"]), _fmt(p["SE"]),
                         _fmt(p.get("beta", 0)), _fmt(p["t"]), _p_fmt(p["p"]),
                         _fmt(ci[0]) if ci[0] is not None else "",
                         _fmt(ci[1]) if ci[1] is not None else ""])
    csv_str = buf.getvalue()

    return APATableResponse(html=html, latex=latex, markdown=markdown,
                            csv=csv_str, title=title, note=model_note)


def _build_anova_table(data: dict, title: str, note: str) -> APATableResponse:
    """ANOVA source table: Source, SS, df, MS, F, p, eta-squared."""
    sources: list[dict] = data.get("sources", [])
    if not sources:
        raise HTTPException(status_code=400, detail="data.sources required for ANOVA table")

    # ── HTML ──
    rows_html = ""
    for s in sources:
        p_str = _p_fmt(s["p"]) if "p" in s and s["p"] is not None else ""
        f_str = _fmt(s["F"]) if "F" in s and s["F"] is not None else ""
        eta_str = _fmt(s.get("eta_sq", 0)) if s.get("eta_sq") is not None else ""
        rows_html += (
            f"<tr>"
            f"<td style=\"{_TD_CSS} text-align:left;\">{s['source']}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(s['SS'])}</td>"
            f"<td style=\"{_TD_CSS}\">{s['df']}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(s['MS'])}</td>"
            f"<td style=\"{_TD_CSS}\">{f_str}</td>"
            f"<td style=\"{_TD_CSS}\">{p_str}</td>"
            f"<td style=\"{_TD_CSS}\">{eta_str}</td>"
            f"</tr>\n"
        )
    html = (
        f"<table style=\"{_APA_CSS}\">\n"
        f"<caption style=\"text-align:left; font-style:italic;\">{title}</caption>\n"
        f"<thead><tr>"
        f"<th style=\"{_TH_CSS} text-align:left;\">Source</th>"
        f"<th style=\"{_TH_CSS}\"><em>SS</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>df</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>MS</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>F</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>p</em></th>"
        f"<th style=\"{_TH_CSS}\">&eta;&sup2;</th>"
        f"</tr></thead>\n<tbody>\n{rows_html}</tbody>\n"
        f"<tfoot><tr><td colspan=\"7\" style=\"{_BOTTOM_CSS} text-align:left; font-size:10pt;\">"
        f"<em>Note.</em> {note}</td></tr></tfoot>\n</table>"
    )

    # ── Markdown ──
    md_lines = [
        f"*{title}*\n",
        "| Source | *SS* | *df* | *MS* | *F* | *p* | eta-sq |",
        "|:-------|-----:|-----:|-----:|----:|----:|-------:|",
    ]
    for s in sources:
        p_str = _p_fmt(s["p"]) if "p" in s and s["p"] is not None else ""
        f_str = _fmt(s["F"]) if "F" in s and s["F"] is not None else ""
        eta_str = _fmt(s.get("eta_sq", 0)) if s.get("eta_sq") is not None else ""
        md_lines.append(
            f"| {s['source']} | {_fmt(s['SS'])} | {s['df']} | "
            f"{_fmt(s['MS'])} | {f_str} | {p_str} | {eta_str} |"
        )
    md_lines.append(f"\n*Note.* {note}")
    markdown = "\n".join(md_lines)

    # ── LaTeX ──
    lrows = ""
    for s in sources:
        p_str = _p_fmt(s["p"]) if "p" in s and s["p"] is not None else ""
        f_str = _fmt(s["F"]) if "F" in s and s["F"] is not None else ""
        eta_str = _fmt(s.get("eta_sq", 0)) if s.get("eta_sq") is not None else ""
        lrows += (
            f"  {s['source']} & {_fmt(s['SS'])} & {s['df']} & "
            f"{_fmt(s['MS'])} & {f_str} & {p_str} & {eta_str} \\\\\n"
        )
    latex = (
        "\\begin{table}[ht]\n\\caption{" + title + "}\n"
        "\\begin{tabular}{l r r r r r r}\n\\toprule\n"
        "Source & $SS$ & $df$ & $MS$ & $F$ & $p$ & $\\eta^2$ \\\\\n\\midrule\n"
        + lrows +
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n\\item \\textit{Note.} " + note +
        "\n\\end{tablenotes}\n\\end{table}"
    )

    # ── CSV ──
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Source", "SS", "df", "MS", "F", "p", "eta_sq"])
    for s in sources:
        writer.writerow([
            s["source"], _fmt(s["SS"]), s["df"], _fmt(s["MS"]),
            _fmt(s["F"]) if "F" in s and s["F"] is not None else "",
            _p_fmt(s["p"]) if "p" in s and s["p"] is not None else "",
            _fmt(s.get("eta_sq", 0)) if s.get("eta_sq") is not None else "",
        ])
    csv_str = buf.getvalue()

    return APATableResponse(html=html, latex=latex, markdown=markdown,
                            csv=csv_str, title=title, note=note)


def _build_t_test_table(data: dict, title: str, note: str) -> APATableResponse:
    """T-test results: group means, SDs, t, df, p, Cohen's d."""
    groups: list[dict] = data.get("groups", [])
    test_stats: dict = data.get("test_stats", {})

    if not groups or not test_stats:
        raise HTTPException(status_code=400, detail="data.groups and data.test_stats required")

    # ── HTML ──
    group_rows = ""
    for g in groups:
        group_rows += (
            f"<tr>"
            f"<td style=\"{_TD_CSS} text-align:left;\">{g['name']}</td>"
            f"<td style=\"{_TD_CSS}\">{g.get('n', '')}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(g['mean'])}</td>"
            f"<td style=\"{_TD_CSS}\">{_fmt(g['sd'])}</td>"
            f"</tr>\n"
        )
    t_val = _fmt(test_stats["t"])
    df_val = test_stats["df"]
    p_val = _p_fmt(test_stats["p"])
    d_val = _fmt(test_stats.get("d", 0))
    stats_note = f"{note} t({df_val}) = {t_val}, p {p_val}, d = {d_val}."

    html = (
        f"<table style=\"{_APA_CSS}\">\n"
        f"<caption style=\"text-align:left; font-style:italic;\">{title}</caption>\n"
        f"<thead><tr>"
        f"<th style=\"{_TH_CSS} text-align:left;\">Group</th>"
        f"<th style=\"{_TH_CSS}\"><em>n</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>M</em></th>"
        f"<th style=\"{_TH_CSS}\"><em>SD</em></th>"
        f"</tr></thead>\n<tbody>\n{group_rows}</tbody>\n"
        f"<tfoot><tr><td colspan=\"4\" style=\"{_BOTTOM_CSS} text-align:left; font-size:10pt;\">"
        f"<em>Note.</em> {stats_note}</td></tr></tfoot>\n</table>"
    )

    # ── Markdown ──
    md_lines = [
        f"*{title}*\n",
        "| Group | *n* | *M* | *SD* |",
        "|:------|----:|----:|-----:|",
    ]
    for g in groups:
        md_lines.append(f"| {g['name']} | {g.get('n', '')} | {_fmt(g['mean'])} | {_fmt(g['sd'])} |")
    md_lines.append(f"\n*Note.* {stats_note}")
    markdown = "\n".join(md_lines)

    # ── LaTeX ──
    lrows = ""
    for g in groups:
        lrows += f"  {g['name']} & {g.get('n', '')} & {_fmt(g['mean'])} & {_fmt(g['sd'])} \\\\\n"
    latex = (
        "\\begin{table}[ht]\n\\caption{" + title + "}\n"
        "\\begin{tabular}{l r r r}\n\\toprule\n"
        "Group & $n$ & $M$ & $SD$ \\\\\n\\midrule\n"
        + lrows +
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n\\item \\textit{Note.} " + stats_note +
        "\n\\end{tablenotes}\n\\end{table}"
    )

    # ── CSV ──
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Group", "n", "M", "SD"])
    for g in groups:
        writer.writerow([g["name"], g.get("n", ""), _fmt(g["mean"]), _fmt(g["sd"])])
    writer.writerow([])
    writer.writerow(["t", "df", "p", "d"])
    writer.writerow([t_val, df_val, p_val, d_val])
    csv_str = buf.getvalue()

    return APATableResponse(html=html, latex=latex, markdown=markdown,
                            csv=csv_str, title=title, note=stats_note)


def _build_frequencies_table(data: dict, title: str, note: str) -> APATableResponse:
    """Frequency / contingency table."""
    categories: list[dict] = data.get("categories", [])
    if not categories:
        raise HTTPException(status_code=400, detail="data.categories required for frequencies table")

    total_n = sum(c.get("n", 0) for c in categories)

    # ── HTML ──
    rows_html = ""
    for c in categories:
        pct = (c["n"] / total_n * 100) if total_n > 0 else 0
        rows_html += (
            f"<tr>"
            f"<td style=\"{_TD_CSS} text-align:left;\">{c['label']}</td>"
            f"<td style=\"{_TD_CSS}\">{c['n']}</td>"
            f"<td style=\"{_TD_CSS}\">{pct:.1f}%</td>"
            f"</tr>\n"
        )
    html = (
        f"<table style=\"{_APA_CSS}\">\n"
        f"<caption style=\"text-align:left; font-style:italic;\">{title}</caption>\n"
        f"<thead><tr>"
        f"<th style=\"{_TH_CSS} text-align:left;\">Category</th>"
        f"<th style=\"{_TH_CSS}\"><em>n</em></th>"
        f"<th style=\"{_TH_CSS}\">%</th>"
        f"</tr></thead>\n<tbody>\n{rows_html}</tbody>\n"
        f"<tfoot><tr><td colspan=\"3\" style=\"{_BOTTOM_CSS} text-align:left; font-size:10pt;\">"
        f"<em>Note.</em> {note} N = {total_n}.</td></tr></tfoot>\n</table>"
    )

    # ── Markdown ──
    md_lines = [f"*{title}*\n", "| Category | *n* | % |", "|:---------|----:|--:|"]
    for c in categories:
        pct = (c["n"] / total_n * 100) if total_n > 0 else 0
        md_lines.append(f"| {c['label']} | {c['n']} | {pct:.1f}% |")
    md_lines.append(f"\n*Note.* {note} N = {total_n}.")
    markdown = "\n".join(md_lines)

    # ── LaTeX ──
    lrows = ""
    for c in categories:
        pct = (c["n"] / total_n * 100) if total_n > 0 else 0
        lrows += f"  {c['label']} & {c['n']} & {pct:.1f}\\% \\\\\n"
    latex = (
        "\\begin{table}[ht]\n\\caption{" + title + "}\n"
        "\\begin{tabular}{l r r}\n\\toprule\n"
        "Category & $n$ & \\% \\\\\n\\midrule\n"
        + lrows +
        "\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\small\n\\item \\textit{Note.} " + note + f" $N$ = {total_n}." +
        "\n\\end{tablenotes}\n\\end{table}"
    )

    # ── CSV ──
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Category", "n", "%"])
    for c in categories:
        pct = (c["n"] / total_n * 100) if total_n > 0 else 0
        writer.writerow([c["label"], c["n"], f"{pct:.1f}"])
    csv_str = buf.getvalue()

    return APATableResponse(html=html, latex=latex, markdown=markdown,
                            csv=csv_str, title=title, note=f"{note} N = {total_n}.")


# ---------------------------------------------------------------------------
# Table builder dispatch
# ---------------------------------------------------------------------------

_TABLE_BUILDERS = {
    "descriptive": _build_descriptive_table,
    "correlation": _build_correlation_table,
    "regression": _build_regression_table,
    "anova": _build_anova_table,
    "t_test": _build_t_test_table,
    "frequencies": _build_frequencies_table,
}


# ---------------------------------------------------------------------------
# POST /apa-table
# ---------------------------------------------------------------------------

@router.post("/apa-table", response_model=APATableResponse)
def apa_table(req: APATableRequest):
    """Generate an APA-formatted results table in HTML, LaTeX, Markdown, and CSV."""
    builder = _TABLE_BUILDERS.get(req.table_type)
    if builder is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown table_type '{req.table_type}'. "
                   f"Supported: {', '.join(_TABLE_BUILDERS.keys())}",
        )

    title = req.title or f"Table: {req.table_type.replace('_', ' ').title()} Results"
    note = req.note or ""

    return builder(req.data, title, note)


# ---------------------------------------------------------------------------
# POST /full-report
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(text.split())


def _build_report_html(req: FullReportRequest) -> str:
    """Build a full APA 7 styled HTML report."""
    sections_html = ""
    tables_count = 0
    for sec in req.sections:
        sections_html += f"<h2 style=\"font-weight:bold; text-align:center;\">{sec.heading}</h2>\n"
        sections_html += f"<p style=\"text-indent: 0.5in;\">{sec.content}</p>\n"
        if sec.apa_text:
            sections_html += f"<p style=\"text-indent: 0.5in;\">{sec.apa_text}</p>\n"
        if sec.table:
            tables_count += 1
            sections_html += f"<p><em>[Insert Table {tables_count} here]</em></p>\n"

    author_line = f"<p style=\"text-align:center;\">{req.author}</p>" if req.author else ""

    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "<style>\n"
        "  body { font-family: 'Times New Roman', Times, serif; font-size: 12pt; "
        "line-height: 2; margin: 1in; max-width: 6.5in; }\n"
        "  h1 { font-weight: bold; text-align: center; }\n"
        "</style>\n</head>\n<body>\n"
        f"<h1>{req.title}</h1>\n{author_line}\n"
        f"{sections_html}\n"
        "</body>\n</html>"
    )


def _build_report_latex(req: FullReportRequest) -> str:
    """Build a LaTeX document."""
    sections_latex = ""
    tables_count = 0
    for sec in req.sections:
        sections_latex += f"\\section{{{sec.heading}}}\n\n"
        sections_latex += f"{sec.content}\n\n"
        if sec.apa_text:
            sections_latex += f"{sec.apa_text}\n\n"
        if sec.table:
            tables_count += 1
            sections_latex += f"[Insert Table {tables_count} here]\n\n"

    author_line = f"\\author{{{req.author}}}" if req.author else "\\author{}"

    return (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{setspace}\n"
        "\\doublespacing\n\n"
        f"\\title{{{req.title}}}\n"
        f"{author_line}\n"
        "\\date{}\n\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        f"{sections_latex}"
        "\\end{document}"
    )


def _build_report_markdown(req: FullReportRequest) -> str:
    """Build a Markdown document."""
    parts = [f"# {req.title}\n"]
    if req.author:
        parts.append(f"*{req.author}*\n")

    tables_count = 0
    for sec in req.sections:
        parts.append(f"## {sec.heading}\n")
        parts.append(f"{sec.content}\n")
        if sec.apa_text:
            parts.append(f"{sec.apa_text}\n")
        if sec.table:
            tables_count += 1
            parts.append(f"*[Insert Table {tables_count} here]*\n")

    return "\n".join(parts)


@router.post("/full-report", response_model=FullReportResponse)
def full_report(req: FullReportRequest):
    """Generate a complete analysis report in the requested format."""
    if req.format not in ("html", "latex", "markdown"):
        raise HTTPException(status_code=400, detail="format must be 'html', 'latex', or 'markdown'")

    builders = {
        "html": _build_report_html,
        "latex": _build_report_latex,
        "markdown": _build_report_markdown,
    }

    content = builders[req.format](req)
    tables_count = sum(1 for s in req.sections if s.table)

    # Count figure references in all text
    all_text = " ".join(s.content + (s.apa_text or "") for s in req.sections)
    import re
    figures_referenced = len(re.findall(r"[Ff]igure\s+\d+", all_text))

    return FullReportResponse(
        content=content,
        word_count=_count_words(content),
        tables_count=tables_count,
        figures_referenced=figures_referenced,
    )


# ---------------------------------------------------------------------------
# POST /latex-equation
# ---------------------------------------------------------------------------

_EQUATION_BUILDERS: dict[str, callable] = {}


def _eq_regression(params: dict) -> tuple[str, str]:
    predictors = params.get("predictors", ["X_1", "X_2"])
    outcome = params.get("outcome", "Y")
    n = len(predictors)

    terms = " + ".join(f"\\beta_{{{i+1}}} {p}" for i, p in enumerate(predictors))
    eq = f"{outcome}_i = \\beta_0 + {terms} + \\epsilon_i"

    desc = (
        f"Multiple linear regression predicting {outcome} from "
        f"{', '.join(predictors)}. "
        f"beta_0 is the intercept, beta_1 through beta_{n} are the "
        f"regression coefficients, and epsilon is the residual error term."
    )
    return eq, desc


def _eq_hlm(params: dict) -> tuple[str, str]:
    outcome = params.get("outcome", "Y")
    level1_pred = params.get("level1_predictors", ["X"])
    level2_pred = params.get("level2_predictors", ["W"])

    l1_terms = " + ".join(f"\\beta_{{1{i}j}} {p}" for i, p in enumerate(level1_pred))
    eq_l1 = f"{outcome}_{{ij}} = \\beta_{{0j}} + {l1_terms} + r_{{ij}}"

    l2_intercept = f"\\beta_{{0j}} = \\gamma_{{00}}"
    if level2_pred:
        l2_intercept += " + " + " + ".join(f"\\gamma_{{0{k+1}}} {w}" for k, w in enumerate(level2_pred))
    l2_intercept += " + u_{0j}"

    eq = f"{eq_l1} \\\\[6pt]\n{l2_intercept}"

    desc = (
        f"Two-level hierarchical linear model. Level 1: {outcome} predicted by "
        f"{', '.join(level1_pred)} within clusters (j). Level 2: the intercept "
        f"varies across clusters as a function of {', '.join(level2_pred) if level2_pred else 'no predictors'}. "
        f"r_ij is the within-cluster residual, u_0j is the random intercept variance."
    )
    return eq, desc


def _eq_sem(params: dict) -> tuple[str, str]:
    latent = params.get("latent_variables", {"eta": ["Y1", "Y2", "Y3"]})
    structural = params.get("structural", [])

    # Measurement model
    meas_lines = []
    for lv, indicators in latent.items():
        loadings = " + ".join(f"\\lambda_{{{i+1}}} {ind}" for i, ind in enumerate(indicators))
        meas_lines.append(f"{lv} = {loadings}")

    # Structural model
    struct_lines = []
    for rel in structural:
        struct_lines.append(f"{rel.get('dv', 'eta_1')} = "
                           f"{'+ '.join(rel.get('ivs', ['eta_2']))} + \\zeta")

    eq = " \\\\[4pt]\n".join(meas_lines + struct_lines)

    desc = (
        f"Structural equation model with {len(latent)} latent variable(s). "
        "Lambda values are factor loadings, zeta is the structural disturbance term."
    )
    return eq, desc


def _eq_anova(params: dict) -> tuple[str, str]:
    outcome = params.get("outcome", "Y")
    factors = params.get("factors", ["A"])

    if len(factors) == 1:
        eq = f"{outcome}_{{ij}} = \\mu + \\alpha_i + \\epsilon_{{ij}}"
        desc = (
            f"One-way ANOVA model. mu is the grand mean, alpha_i is the effect "
            f"of level i of factor {factors[0]}, epsilon is the error term."
        )
    else:
        eq = (
            f"{outcome}_{{ijk}} = \\mu + \\alpha_i + \\beta_j + "
            f"(\\alpha\\beta)_{{ij}} + \\epsilon_{{ijk}}"
        )
        desc = (
            f"Two-way factorial ANOVA. mu is the grand mean, alpha_i is the main "
            f"effect of {factors[0]}, beta_j is the main effect of {factors[1]}, "
            f"(alpha*beta)_ij is their interaction, epsilon is the error term."
        )
    return eq, desc


def _eq_mediation(params: dict) -> tuple[str, str]:
    x = params.get("x", "X")
    m = params.get("mediator", "M")
    y = params.get("y", "Y")

    eq = (
        f"{m} = a \\cdot {x} + e_1 \\\\[4pt]\n"
        f"{y} = c' \\cdot {x} + b \\cdot {m} + e_2 \\\\[4pt]\n"
        f"\\text{{Indirect effect}} = a \\times b"
    )
    desc = (
        f"Mediation model: {x} -> {m} -> {y}. 'a' is the effect of {x} on {m}, "
        f"'b' is the effect of {m} on {y} controlling for {x}, c' is the direct "
        f"effect, and a*b is the indirect (mediated) effect."
    )
    return eq, desc


_EQUATION_BUILDERS = {
    "regression": _eq_regression,
    "hlm": _eq_hlm,
    "sem": _eq_sem,
    "anova": _eq_anova,
    "mediation": _eq_mediation,
}


@router.post("/latex-equation", response_model=LatexEquationResponse)
def latex_equation(req: LatexEquationRequest):
    """Generate a LaTeX equation string for a statistical model."""
    builder = _EQUATION_BUILDERS.get(req.model_type)
    if builder is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_type '{req.model_type}'. "
                   f"Supported: {', '.join(_EQUATION_BUILDERS.keys())}",
        )

    equation, description = builder(req.parameters)

    return LatexEquationResponse(equation=equation, description=description)
