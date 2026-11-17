import data_init
from table_chunks import detect_tables, split_table, to_markdown


def _word(text, x0, x1, top):
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": top + 6}


def _amount_table():
    """Two right-aligned amount columns plus a year/$m header. No PDF involved."""
    words = [
        _word("Note", 40, 70, 10),
        _word("2025", 170, 200, 10),
        _word("2024", 250, 280, 10),
        _word("$m", 188, 200, 22),
        _word("$m", 268, 280, 22),
    ]
    for index, (label, left, right) in enumerate([
        ("Revenue", "10,333", "7,189"),
        ("Costs", "(4,000)", "(3,886)"),
        ("Profit", "6,333", "3,303"),
        ("Tax", "(1,000)", "(800)"),
    ]):
        top = 40 + index * 12
        words.append(_word(label, 40, 100, top))
        words.append(_word(left, 200 - len(left) * 4, 200, top))
        words.append(_word(right, 280 - len(right) * 4, 280, top))
    return words


def test_markdown_escapes_pipes_and_keeps_header():
    markdown = to_markdown(["Item|name", "2025"], [["Revenue", "10,333"]])
    assert markdown.splitlines() == [
        r"| Item\|name | 2025 |",
        "| --- | --- |",
        "| Revenue | 10,333 |",
    ]


def test_long_table_splits_by_row_and_repeats_header():
    header = ["Item", "2025 $m"]
    rows = [[f"row-{i:02d}", f"{i},000"] for i in range(12)]
    chunks = split_table(header, rows, token_limit=40, token_len=lambda text: len(text.split()), prefix="HKEX FY2025 p.1")

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("HKEX FY2025 p.1\n")
        assert "| Item | 2025 $m |" in chunk
        assert "| --- | --- |" in chunk
        assert len(chunk.split()) <= 40
    body = "\n".join(chunks)
    for index in range(12):
        assert body.count(f"row-{index:02d}") == 1


def test_detect_tables_puts_header_and_figures_in_one_table():
    tables = detect_tables(_amount_table())

    assert len(tables) == 1
    assert "2025" in tables[0].header[1] and "$m" in tables[0].header[1]
    assert "2024" in tables[0].header[2]
    assert tables[0].body_rows[0] == ["Revenue", "10,333", "7,189"]
    assert tables[0].body_rows[1][1] == "(4,000)"


def test_table_chunks_off_leaves_page_text_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("TABLE_CHUNKS", raising=False)
    (tmp_path / "0388_2025_en.pdf").write_bytes(b"%PDF-stub")
    monkeypatch.setattr(data_init, "_read_pdf_pages", lambda path: ["Revenue table", "", "Chairman's statement"])
    manifest = {"0388_2025_en.pdf": {
        "file_name": "0388_2025_en.pdf", "ticker": "0388", "company": "HKEX", "fiscal_year": "2025",
        "language": "en", "source_url": "https://www.hkexnews.hk/example.pdf", "download_date": "2026-11-13",
        "sha256": "",
    }}

    docs = data_init.load_documents(tmp_path, manifest)

    assert [doc.text for doc in docs] == ["Revenue table", "Chairman's statement"]
    assert all("chunk_type" not in doc.metadata for doc in docs)
    nodes = data_init.semantic_chunk_documents(docs)
    assert [node.get_content() for node in nodes] == ["Revenue table", "Chairman's statement"]
    assert all("chunk_type" not in node.metadata for node in nodes)


def test_table_chunks_on_does_not_resplit_a_table(tmp_path, monkeypatch):
    monkeypatch.setenv("TABLE_CHUNKS", "on")
    (tmp_path / "0388_2025_en.pdf").write_bytes(b"%PDF-stub")
    monkeypatch.setattr(data_init, "_read_pdf_pages", lambda path: ["narrative around the table"])
    monkeypatch.setattr(data_init, "_page_words", lambda path: [_amount_table()])
    manifest = {"0388_2025_en.pdf": {
        "file_name": "0388_2025_en.pdf", "ticker": "0388", "company": "HKEX", "fiscal_year": "2025",
        "language": "en", "source_url": "", "download_date": "", "sha256": "",
    }}

    docs = data_init.load_documents(tmp_path, manifest)
    tables = [doc for doc in docs if doc.metadata.get("chunk_type") == "table"]

    assert len(tables) == 1
    assert "Revenue" in tables[0].text and "10,333" in tables[0].text and "2025" in tables[0].text
    assert tables[0].text.startswith("HKEX FY2025 p.1")
    nodes = data_init.semantic_chunk_documents(docs)
    table_nodes = [node for node in nodes if node.metadata.get("chunk_type") == "table"]
    assert len(table_nodes) == 1
    assert table_nodes[0].get_content() == tables[0].text
    assert "chunk_type" in table_nodes[0].excluded_embed_metadata_keys
