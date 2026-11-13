import data_init


def test_pdf_pages_become_documents_with_manifest_metadata(tmp_path, monkeypatch):
    (tmp_path / "0388_2025_en.pdf").write_bytes(b"%PDF-stub")
    monkeypatch.setattr(data_init, "_read_pdf_pages", lambda path: ["Revenue table", "", "Chairman's statement"])
    manifest = {"0388_2025_en.pdf": {
        "file_name": "0388_2025_en.pdf", "ticker": "0388", "company": "HKEX", "fiscal_year": "2025",
        "language": "en", "source_url": "https://www.hkexnews.hk/example.pdf", "download_date": "2026-11-13",
        "sha256": "",
    }}

    docs = data_init.load_documents(tmp_path, manifest)

    assert [d.metadata["page"] for d in docs] == [1, 3]  # empty page 2 skipped, PDF numbering kept
    meta = docs[0].metadata
    assert (meta["ticker"], meta["company"], meta["fiscal_year"]) == ("0388", "HKEX", 2025)
    assert meta["section"] == "HKEX FY2025 p.1"
    assert "source_url" in docs[0].excluded_embed_metadata_keys
    assert "page" not in docs[0].excluded_embed_metadata_keys

    nodes = data_init.semantic_chunk_documents(docs)
    assert {n.metadata["page"] for n in nodes} == {1, 3}
    assert all(n.metadata["section"].startswith("HKEX FY2025 p.") for n in nodes)
    assert all("source_url" in n.excluded_embed_metadata_keys for n in nodes)
