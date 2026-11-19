"""Security limits for data import, report generation and the export engine.

Uploads and pasted text are bounded before parsing (zip bombs, huge sheets, regex separators),
error responses never carry exception text, and every user value placed in generated HTML or
LaTeX is escaped.
"""

import io
import zipfile

import pytest

XSS = '<script>alert("x")</script>'


# ── Data import ────────────────────────────────────────────────────────────

def _upload(client, name, content):
    return client.post("/api/import/upload", files={"file": (name, content, "application/octet-stream")})


def test_csv_upload_still_works(client):
    r = _upload(client, "d.csv", b"a,b\n1,2\n3,4\n")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] == 2 and body["columns"] == ["a", "b"]


def test_parse_text_still_works(client):
    r = client.post("/api/import/parse-text", json={"text": "x y\n1 2\n3 4\n"})
    assert r.status_code == 200, r.text
    assert r.json()["columns"] == ["x", "y"]
    r = client.post("/api/import/parse-text", json={"text": "x;y\n1;2\n"})
    assert r.json()["columns"] == ["x", "y"]


def test_csv_too_many_columns_rejected(client):
    header = ",".join(f"c{i}" for i in range(1_500)).encode()
    r = _upload(client, "wide.csv", header + b"\n" + b",".join([b"1"] * 1_500) + b"\n")
    assert r.status_code == 422
    assert "1,000 columns" in r.json()["detail"]


def test_csv_too_many_rows_rejected(client):
    r = _upload(client, "long.csv", b"a\n" + b"1\n" * 100_001)
    assert r.status_code == 422
    assert "100,000 rows" in r.json()["detail"]


def test_upload_parse_error_does_not_echo_exception(client):
    r = _upload(client, "bad.json", b'{"a": [1, 2,, }')
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "Expecting" not in detail and "line 1" not in detail and "Traceback" not in detail


def test_unsupported_extension_not_echoed(client):
    r = _upload(client, f"x.{'<b>evil</b>'}", b"data")
    assert r.status_code == 400
    assert "evil" not in r.json()["detail"]


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_xlsx_zip_bomb_rejected(client):
    # 60 MB of zeros compresses to ~60 KB: refused from the zip directory, before inflating.
    bomb = _zip_bytes({"xl/worksheets/sheet1.xml": b"0" * (60 * 1024 * 1024)})
    assert len(bomb) < 1_000_000
    r = _upload(client, "bomb.xlsx", bomb)
    assert r.status_code == 422
    assert "MB uncompressed" in r.json()["detail"]


def test_xlsx_not_a_zip_rejected(client):
    r = _upload(client, "fake.xlsx", b"not a zip at all")
    assert r.status_code == 422
    assert "valid .xlsx" in r.json()["detail"]


def _xlsx(rows_xml: str) -> bytes:
    """A minimal workbook whose first sheet has the given <sheetData> rows."""
    return _zip_bytes({
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>').encode(),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>').encode(),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>').encode(),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>').encode(),
        "xl/worksheets/sheet1.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            + rows_xml + '</sheetData></worksheet>').encode(),
    })


def test_xlsx_small_sheet_still_works(client):
    rows = ('<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is><t>b</t></is></c></row>'
            '<row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2</v></c></row>'
            '<row r="3"><c r="A3"><v>3</v></c><c r="B3"><v>4</v></c></row>')
    r = _upload(client, "ok.xlsx", _xlsx(rows))
    assert r.status_code == 200, r.text
    assert r.json()["n_rows"] == 2 and r.json()["columns"] == ["a", "b"]


def test_xlsx_far_column_rejected_before_padding(client):
    # One cell in column XFD (16,384) would make openpyxl pad every row to 16,384 cells.
    rows = '<row r="1"><c r="A1"><v>1</v></c><c r="XFD1"><v>1</v></c></row>'
    rows += "".join(f'<row r="{i}"><c r="A{i}"><v>{i}</v></c></row>' for i in range(2, 50))
    r = _upload(client, "wide.xlsx", _xlsx(rows))
    assert r.status_code == 422
    assert "1,000 columns" in r.json()["detail"]


def test_json_dict_of_dicts_bomb_rejected(client):
    # 2,000 columns x a 1,000-key union index would expand to 2M+ cells... and more columns than allowed.
    body = "{" + ",".join(f'"c{i}": {{"r{i}": 1}}' for i in range(2_000)) + "}"
    r = _upload(client, "wide.json", body.encode())
    assert r.status_code == 422


def test_parse_text_regex_separator_is_literal(client):
    # "(a+)+$" as a regex would backtrack catastrophically on the text below; as a literal it never matches.
    text = "col\n" + "a" * 5_000 + "!\n"
    r = client.post("/api/import/parse-text", json={"text": text, "separator": "(a+)+$"})
    assert r.status_code == 200, r.text
    assert r.json()["n_columns"] == 1
    r = client.post("/api/import/parse-text", json={"text": "a||b\n1||2\n", "separator": "||"})
    assert r.status_code == 200, r.text
    assert r.json()["columns"] == ["a", "b"]


def test_parse_text_limits(client):
    r = client.post("/api/import/parse-text", json={"text": "a,b\n1,2\n", "separator": "x" * 11})
    assert r.status_code == 422
    r = client.post("/api/import/parse-text", json={"text": "a\n" + "1\n" * 100_001})
    assert r.status_code == 422


def test_parse_text_error_does_not_echo_exception(client):
    r = client.post("/api/import/parse-text", json={"text": 'a,b\n"1,2\n', "separator": ","})
    assert r.status_code == 422
    assert "EOF" not in r.json()["detail"] and "Error tokenizing" not in r.json()["detail"]


# ── Reports ────────────────────────────────────────────────────────────────

def test_report_escapes_user_html(client):
    r = client.post("/api/reports/generate", json={
        "title": XSS, "author": XSS,
        "sections": [{"title": XSS, "content": f"{XSS} t(10) = 2.1, p < .05",
                      "chart_data": {"title": XSS}}],
        "apa_results": [{"test_type": XSS, "apa_text": XSS, "raw_results": {XSS: XSS}}],
    })
    assert r.status_code == 200, r.text
    html = r.json()["html"]
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    # Intentional formatting generated by the report code survives.
    assert "<em>p</em> &lt; .05" in html


def test_report_limits(client):
    r = client.post("/api/reports/generate", json={
        "title": "t", "sections": [{"title": "s", "content": "x"}] * 201})
    assert r.status_code == 422
    r = client.post("/api/reports/generate", json={
        "title": "t", "sections": [{"title": "s", "content": "x" * 100_001}]})
    assert r.status_code == 422


# ── Export engine ──────────────────────────────────────────────────────────

TEX_ATTACK = r"\input{/etc/passwd} $x$ & 100% #1 ^ _ ~ {}"


def test_apa_table_escapes_html_and_latex(client):
    r = client.post("/api/export/apa-table", json={
        "table_type": "descriptive", "title": XSS + TEX_ATTACK, "note": XSS,
        "data": {"variables": [{"name": XSS + TEX_ATTACK, "n": XSS, "mean": 1.5, "sd": 0.5}]},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "<script>" not in body["html"] and "&lt;script&gt;" in body["html"]
    assert r"\input" not in body["latex"]
    assert r"\textbackslash{}input\{/etc/passwd\} \$x\$ \& 100\% \#1" in body["latex"]
    assert r"\begin{table}" in body["latex"]  # generated LaTeX is untouched


def test_apa_table_malformed_data_is_422_without_exception_text(client):
    r = client.post("/api/export/apa-table", json={
        "table_type": "descriptive", "data": {"variables": [{"name": "x", "mean": "not-a-number", "sd": 1}]}})
    assert r.status_code == 422
    assert "format code" not in r.json()["detail"]
    r = client.post("/api/export/apa-table", json={
        "table_type": "regression", "data": {"predictors": [{"name": "x"}]}})
    assert r.status_code == 422


def test_apa_table_size_limits(client):
    r = client.post("/api/export/apa-table", json={
        "table_type": "correlation",
        "data": {"variables": [f"v{i}" for i in range(201)], "matrix": [[1.0]]}})
    assert r.status_code == 422
    assert "200 variables" in r.json()["detail"]
    r = client.post("/api/export/apa-table", json={
        "table_type": "<b>bogus</b>", "data": {}})
    assert r.status_code == 400 and "bogus" not in r.json()["detail"]


@pytest.mark.parametrize("fmt", ["html", "latex"])
def test_full_report_escapes(client, fmt):
    r = client.post("/api/export/full-report", json={
        "title": XSS + TEX_ATTACK, "author": XSS + TEX_ATTACK, "format": fmt,
        "sections": [{"heading": XSS + TEX_ATTACK, "content": XSS + TEX_ATTACK + "\n\nSecond paragraph",
                      "apa_text": XSS + TEX_ATTACK}],
    })
    assert r.status_code == 200, r.text
    content = r.json()["content"]
    if fmt == "html":
        assert "<script>" not in content and "&lt;script&gt;" in content
        assert "<h1>" in content
    else:
        assert r"\input" not in content and "<script>" in content  # HTML is inert in LaTeX
        assert r"\section{" in content and "\n\nSecond paragraph" in content


def test_latex_equation_names_are_escaped(client):
    r = client.post("/api/export/latex-equation", json={
        "model_type": "regression",
        "parameters": {"outcome": r"\input{/etc/passwd}", "predictors": ["X_1", "age group", r"$\write18{rm}$"]},
    })
    assert r.status_code == 200, r.text
    eq = r.json()["equation"]
    assert r"\input" not in eq and r"\write18" not in eq
    assert r"\beta_{1} X_1" in eq  # simple identifiers keep their math formatting
    assert r"\text{age group}" in eq


def test_latex_equation_limits_and_malformed(client):
    r = client.post("/api/export/latex-equation", json={
        "model_type": "regression", "parameters": {"predictors": [f"x{i}" for i in range(101)]}})
    assert r.status_code == 422
    r = client.post("/api/export/latex-equation", json={
        "model_type": "sem", "parameters": {"latent_variables": ["not", "a", "dict"]}})
    assert r.status_code == 422
