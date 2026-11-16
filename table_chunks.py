"""Turn ruled-or-aligned PDF tables into Markdown chunks.

Detection uses word positions (right-aligned numbers), not a hard-coded page.
TABLE_CHUNKS=off never imports this module's results into the index.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

TABLE_TOKEN_LIMIT = 400
_COLUMN_TOLERANCE = 8.0  # points; right edges of the same column sit this close
_LINE_TOLERANCE = 3.0
_ROW_GAP = 42.0  # a bigger gap starts a new table; section breaks inside a statement stay together
_MIN_BODY_ROWS = 3
_MIN_COLUMNS = 2
_MIN_COLUMN_HITS = 3
_CONTINUATION_ENDINGS = {
    "a", "an", "and", "at", "for", "from", "in", "of", "on", "or", "the", "through", "to",
}

# A real amount: 23,745, (4,626), (955), 14.05, 9.5%.
# Bare integers (note 8, year 2025) do not define a column; a bare integer can still
# fall into a column that real amounts already established.
_AMOUNT = re.compile(
    r"^\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?$"
    r"|^\(?-?\d+\.\d+\)?%?$"
    r"|^\(?\d+%\)?$"
    r"|^\(\d[\d,]*\)$"
)
_BARE_INTEGER = re.compile(r"^\d{1,3}$")
_NUMBERISH = re.compile(
    r"^\(?-?[\d,]+(?:\.\d+)?%?\)?$"  # 2025, 20,21, 23,745, (4,626), 14.05
    r"|^(?:US|HK)?\$m$"
    r"|^(?:–|—|-)$"
    r"|^\d+(?:\([A-Za-z0-9]+\))+$",  # 5(a), 28(a)(i)
    re.IGNORECASE,
)
_COLUMN_TOKEN = re.compile(r"^(?:US|HK)?\$m$|^(?:–|—|-)$", re.IGNORECASE)


def table_chunks_enabled() -> bool:
    return os.getenv("TABLE_CHUNKS", "off").strip().lower() in {"1", "on", "true", "yes"}


_CURRENCY = re.compile(r"^(?:HK|US|RMB|CNY)?\$")
_HEADER_WORD = re.compile(
    r"^(?:current|non-current|total|note|notes|year|ended|december|q[1-4])$|^(?:us|hk)?\$m$|^\d{4}$",
    re.IGNORECASE,
)
_YEAR = re.compile(r"^\d{4}$")


def _core(text: str) -> str:
    """Drop a leading currency sign and a trailing comma so '$14.05' and '8,' still count.

    '$m' keeps its dollar sign; stripping it would leave a bare 'm'.
    """
    token = text.strip().rstrip(",")
    if _COLUMN_TOKEN.match(token):
        return token
    return _CURRENCY.sub("", token)


def _is_numberish(text: str) -> bool:
    return bool(_NUMBERISH.match(_core(text)))


def _is_amount(text: str) -> bool:
    return bool(_AMOUNT.match(_core(text)))


def _builds_column(text: str) -> bool:
    """Tokens that repeat down a numeric column. Note references and years do not."""
    return _is_amount(text) or bool(_COLUMN_TOKEN.match(_core(text)))


def _display_token(text: str) -> str:
    # '8,' is a note reference split from '24'; words like 'receivable,' keep the comma.
    if _is_numberish(text):
        return text.strip().rstrip(",")
    return text.strip()


def _is_continuation(text: str) -> bool:
    """A wrapped label ('... through' / 'profit or loss'), not a section heading ('Assets')."""
    words = [word for word in text.split() if not _is_numberish(word)]
    if not words:
        return False
    if words[0][:1].islower():
        return True
    return words[-1].lower().strip(",:;") in _CONTINUATION_ENDINGS


def _is_column_header(cells: list[str]) -> bool:
    """A row of column titles (years, $m, Current/Total), not a line of the statement."""
    values = [cell for cell in cells[1:] if cell]
    if len(values) < 2:
        return False
    def ok(cell: str) -> bool:
        parts = cell.split()
        return bool(parts) and all(_HEADER_WORD.match(part) for part in parts)
    return all(ok(cell) for cell in values)


def _is_data_line(words: list[dict], columns: list[float]) -> bool:
    """A body row: two or more cells in the amount columns. A pair of years is a header, not data."""
    hits = [
        word for word in words
        if _fills_column(word, columns)
        and not _YEAR.match(word["text"])
        and (
            _is_amount(word["text"])
            or _BARE_INTEGER.match(_core(word["text"]))
            or _core(word["text"]) in {"-", "–", "—"}
        )
    ]
    return len(hits) >= 2


def _header_cells(words: list[dict], columns: list[float]) -> list[str]:
    """Place column titles. Years and words such as Current/$m/Q1; other words stay in the label."""
    cells = [""] * (len(columns) + 1)
    for word in words:
        token = _display_token(word["text"])
        headerish = bool(_HEADER_WORD.match(token))
        if not headerish:
            continue
        if token.lower() in {"note", "notes"}:
            cells[0] = (cells[0] + " " + token).strip()
            continue
        nearest = min(range(len(columns)), key=lambda i: abs(columns[i] - word["x1"]))
        if abs(columns[nearest] - word["x1"]) <= 40:
            cells[nearest + 1] = (cells[nearest + 1] + " " + token).strip()
        elif word["x1"] < columns[0]:
            cells[0] = (cells[0] + " " + token).strip()
    return cells


def _spread_years(header: list[str], columns: list[float]) -> list[str]:
    """Copy a year onto neighbouring Current/Non-current/Total columns that sit under the same year."""
    located = {}
    for index, cell in enumerate(header[1:], start=1):
        years = [token for token in cell.split() if _YEAR.match(token)]
        if years:
            located[index] = years[-1]
    if not located:
        return header
    spread = list(header)
    for index in range(1, len(spread)):
        if any(_YEAR.match(token) for token in spread[index].split()):
            continue
        nearest = min(located, key=lambda other: abs(columns[other - 1] - columns[index - 1]))
        if abs(columns[nearest - 1] - columns[index - 1]) <= 70:
            spread[index] = (located[nearest] + " " + spread[index]).strip()
    return spread


def _merge_header(lines: list[dict], columns: list[float]) -> list[str]:
    merged = [""] * (len(columns) + 1)
    for line in lines:
        for index, cell in enumerate(_header_cells(line["words"], columns)):
            for token in cell.split():
                if token not in merged[index].split():
                    merged[index] = (merged[index] + " " + token).strip()
    if not any(merged[1:]):
        merged = ["Item", *(f"col {i}" for i in range(1, len(columns) + 1))]
    elif not merged[0]:
        merged[0] = "Item"
    return _spread_years(merged, columns)


def _restarts_header(line: dict, columns: list[float]) -> bool:
    """A second year/quarter banner inside a table starts a new table (2025 block, then 2024)."""
    if _has_amount(line["words"], columns):
        return False
    if not _is_column_header(_header_cells(line["words"], columns)):
        return False
    return any(_YEAR.match(word["text"]) or re.fullmatch(r"Q[1-4]", word["text"], re.IGNORECASE) for word in line["words"])


@dataclass
class DetectedTable:
    """One table on a page. `header` is the Markdown header; `captions` sit above it."""

    title: str
    header: list[str]
    body_rows: list[list[str]]
    captions: list[str] = field(default_factory=list)
    word_ids: set[int] = field(default_factory=set)


def _cluster_lines(words: list[dict]) -> list[dict]:
    ordered = sorted(words, key=lambda w: ((w["top"] + w.get("bottom", w["top"])) / 2, w["x0"]))
    lines: list[dict] = []
    for word in ordered:
        center = (word["top"] + word.get("bottom", word["top"])) / 2
        if lines and abs(center - lines[-1]["yc"]) <= _LINE_TOLERANCE:
            lines[-1]["words"].append(word)
        else:
            lines.append({"yc": center, "words": [word]})
    for line in lines:
        line["words"].sort(key=lambda w: w["x0"])
    return lines


def _cluster_edges(edges: list[float], tolerance: float = _COLUMN_TOLERANCE) -> list[float]:
    if not edges:
        return []
    groups: list[list[float]] = [[sorted(edges)[0]]]
    for edge in sorted(edges)[1:]:
        if edge - groups[-1][-1] <= tolerance:
            groups[-1].append(edge)
        else:
            groups.append([edge])
    return [sum(group) / len(group) for group in groups]


def _column_of(edge: float, columns: list[float], tolerance: float = _COLUMN_TOLERANCE) -> int | None:
    if not columns:
        return None
    index = min(range(len(columns)), key=lambda i: abs(columns[i] - edge))
    if abs(columns[index] - edge) <= tolerance:
        return index
    return None


def _cell_of(word: dict, columns: list[float]) -> int | None:
    """Value-column index for a number. Prose stays in the row label; titles use _header_cells."""
    if _is_numberish(word["text"]):
        return _column_of(word["x1"], columns, _COLUMN_TOLERANCE)
    return None


def _line_text(words: list[dict]) -> str:
    return " ".join(word["text"] for word in words).strip()


def _row_cells(words: list[dict], columns: list[float]) -> list[str]:
    """Label plus one cell per numeric column."""
    cells = [""] * (len(columns) + 1)
    for word in words:
        column = _cell_of(word, columns)
        text = _display_token(word["text"])
        if column is None:
            cells[0] = (cells[0] + " " + text).strip()
        else:
            cells[column + 1] = (cells[column + 1] + " " + text).strip()
    return cells


def _has_amount(words: list[dict], columns: list[float]) -> bool:
    return any(
        _is_amount(word["text"]) and _column_of(word["x1"], columns) is not None
        for word in words
    )


def _fills_column(word: dict, columns: list[float]) -> bool:
    return _is_numberish(word["text"]) and _column_of(word["x1"], columns) is not None


def detect_tables(words: list[dict]) -> list[DetectedTable]:
    """Find aligned numeric tables in one page of words.

    `words` items need text, x0, x1, top, and an optional bottom. Anything already
    consumed stays in word_ids so the caller can drop it from the prose.
    """
    lines = _cluster_lines(words)
    candidates = []
    for line in lines:
        numeric = [word for word in line["words"] if _builds_column(word["text"])]
        if len(numeric) >= 2:
            candidates.extend(word["x1"] for word in numeric)
    columns = [
        edge for edge in _cluster_edges(candidates)
        if sum(
            1
            for line in lines
            for word in line["words"]
            if _builds_column(word["text"]) and _column_of(word["x1"], [edge]) is not None
        ) >= _MIN_COLUMN_HITS
    ]
    if len(columns) < _MIN_COLUMNS:
        return []

    data_indexes = [index for index, line in enumerate(lines) if _is_data_line(line["words"], columns)]
    if not data_indexes:
        return []

    runs: list[list[int]] = [[data_indexes[0]]]
    for index in data_indexes[1:]:
        if lines[index]["yc"] - lines[runs[-1][-1]]["yc"] > _ROW_GAP:
            runs.append([index])
        else:
            runs[-1].append(index)

    tables: list[DetectedTable] = []
    previous_last = -1
    for run in runs:
        first, last = run[0], run[-1]
        start = first
        while start - 1 > previous_last and lines[start]["yc"] - lines[start - 1]["yc"] <= _ROW_GAP:
            start -= 1
        region = lines[start:last + 1]
        built = _tables_in_region(region, columns)
        if not built:
            continue
        previous_last = last
        title_lines = _title_lines(lines, start, columns)
        title = ". ".join(_line_text(line["words"]) for line in title_lines)
        used = {id(word) for line in region for word in line["words"]}
        used.update(id(word) for line in title_lines for word in line["words"])
        for header, captions, body in built:
            tables.append(DetectedTable(
                title=title,
                header=header,
                body_rows=body,
                captions=captions,
                word_ids=used,
            ))
    return tables


def _title_lines(lines: list[dict], start: int, columns: list[float]) -> list[dict]:
    found: list[dict] = []
    previous = lines[start]["yc"]
    for line in reversed(lines[:start]):
        if previous - line["yc"] > 80 or len(found) >= 2:
            break
        text = _line_text(line["words"])
        previous = line["yc"]
        if re.fullmatch(r"\d{1,3}", text or ""):
            continue
        if not text or len(line["words"]) > 14 or _has_amount(line["words"], columns):
            break
        found.append(line)
    return list(reversed(found))


def _tables_in_region(region: list[dict], columns: list[float]) -> list[tuple[list[str], list[str], list[list[str]]]]:
    """Split one vertical run when a second year/quarter banner starts another table."""
    groups: list[dict] = []
    current = {"headers": [], "captions": [], "body": [], "amounts": 0, "pending": ""}

    def flush() -> None:
        if current["pending"]:
            current["body"].append([current["pending"]] + [""] * len(columns))
            current["pending"] = ""
        if current["amounts"] >= (2 if current["headers"] else _MIN_BODY_ROWS) and current["body"]:
            groups.append(current)

    for line in region:
        if _restarts_header(line, columns) and current["amounts"]:
            flush()
            current = {"headers": [line], "captions": [], "body": [], "amounts": 0, "pending": ""}
            continue
        if not current["amounts"] and _is_column_header(_header_cells(line["words"], columns)) and not _has_amount(line["words"], columns):
            current["headers"].append(line)
            continue
        cells = _row_cells(line["words"], columns)
        if not any(cells[1:]):
            if not current["amounts"] and len(cells[0].split()) > 2:
                current["captions"].append(_line_text(line["words"]))
                continue
            if current["pending"] and _is_continuation(current["pending"]):
                current["pending"] = (current["pending"] + " " + cells[0]).strip()
            elif _is_continuation(cells[0]):
                current["pending"] = (current["pending"] + " " + cells[0]).strip()
            else:
                if current["pending"]:
                    current["body"].append([current["pending"]] + [""] * len(columns))
                    current["pending"] = ""
                current["body"].append(cells)
            continue
        if current["pending"]:
            cells[0] = (current["pending"] + " " + cells[0]).strip()
            current["pending"] = ""
        current["body"].append(cells)
        if _has_amount(line["words"], columns):
            current["amounts"] += 1
    flush()
    built = []
    carried: list[str] | None = None
    for group in groups:
        header = _merge_header(group["headers"], columns)
        if group["headers"]:
            carried = header
        elif carried:
            header = carried  # a section break kept the figures but not the banner; repeat it
        built.append((header, group["captions"], group["body"]))
    return [item for item in built if _usable_header(item[0])]


def _usable_header(header: list[str]) -> bool:
    """Keep tables whose columns are labelled (year, unit, quarter). Prose with stray numbers is not a table."""
    text = " ".join(header)
    return bool(re.search(r"\b(?:19|20)\d{2}\b|\$m|\bQ[1-4]\b|\b(?:Current|Non-current)\b", text))


def escape_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def to_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Render one Markdown table. `header` is repeated by the caller when a table is split."""
    width = max([len(header), *(len(row) for row in rows)])
    def fit(row: list[str]) -> list[str]:
        return [*row, *([""] * (width - len(row)))][:width]
    header = fit(header)
    def line(row: list[str]) -> str:
        return "| " + " | ".join(escape_cell(cell) for cell in fit(row)) + " |"
    separator = "| " + " | ".join("---" for _ in header) + " |"
    return "\n".join([line(header), separator, *(line(row) for row in rows)])


def _default_token_len(text: str) -> int:
    try:
        from llama_index.core.utils import get_tokenizer
        return len(get_tokenizer()(text))
    except Exception:
        return max(1, len(text) // 4)


def split_table(
    header: list[str],
    rows: list[list[str]],
    *,
    token_limit: int = TABLE_TOKEN_LIMIT,
    token_len=_default_token_len,
    prefix: str = "",
) -> list[str]:
    """Split a long table by rows. Every piece repeats `prefix` and the header row."""
    if not rows:
        return []
    pieces: list[str] = []
    current: list[list[str]] = []

    def render(body: list[list[str]]) -> str:
        table = to_markdown(header, body)
        return f"{prefix}\n{table}" if prefix else table

    for row in rows:
        trial = current + [row]
        if current and token_len(render(trial)) > token_limit:
            pieces.append(render(current))
            current = [row]
        else:
            current = trial
    if current:
        pieces.append(render(current))
    return pieces


def table_to_chunks(
    table: DetectedTable,
    *,
    context: str,
    token_limit: int = TABLE_TOKEN_LIMIT,
    token_len=_default_token_len,
) -> list[str]:
    """Context line, the statement title, then a Markdown table small enough to embed."""
    caption = [context]
    if table.title:
        caption.append(table.title)
    caption.extend(table.captions)
    prefix = "\n".join(line for line in caption if line)
    return split_table(table.header, table.body_rows, token_limit=token_limit, token_len=token_len, prefix=prefix)


def _extract_words(page) -> list[dict]:
    return page.extract_words(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False) or []


def read_page_words(path, page_index: int) -> list[dict]:
    """Words of one PDF page via pdfplumber. Used when TABLE_CHUNKS is on."""
    import pdfplumber

    with pdfplumber.open(str(path)) as document:
        return _extract_words(document.pages[page_index])


def read_document_words(path) -> list[list[dict]]:
    """Words for every page, opening the PDF once."""
    import pdfplumber

    with pdfplumber.open(str(path)) as document:
        return [_extract_words(page) for page in document.pages]


def prose_without_tables(words: list[dict], tables: list[DetectedTable]) -> str:
    """Rebuild the page text from words that were not placed in a table."""
    used = set().union(*(table.word_ids for table in tables)) if tables else set()
    kept = [word for word in words if id(word) not in used]
    lines = _cluster_lines(kept)
    parts: list[str] = []
    previous = None
    for line in lines:
        if previous is not None and line["yc"] - previous > 12:
            parts.append("")
        text = _line_text(line["words"])
        if text:
            parts.append(text)
        previous = line["yc"]
    return "\n".join(parts).strip()
