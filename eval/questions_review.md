# Independent review of eval/questions_draft.jsonl

Reviewer: a separate AI agent (read-only mode), working only from the PDFs via `pdftotext -layout` and `eval/metrics.py`. Saved here by Claude Code on 2026-11-13, because the reviewer couldn't write files. Page numbers are PDF pages, starting at 1.

## Result

- Pass: 18 (h01, h02, h03, h05, h06, h07, h08, h09, h10, h13, h14, h15, a01, a04, a05, a08, a14, a15)
- Needs change: 7 (h04, a02, a03, a06, a07, a09, a10)
- Questionable: 5 (h11, h12, a11, a12, a13)

## Needs change

| id | Problem | Fix |
|---|---|---|
| h04 | The row for $12.52 on p.11 has no label; pp.18, 83 and 136 say "full-year dividend of $12.52" in words | gold_pages → [11, 18, 83, 136] |
| a02 | EV Equity appears both before capital returns (84,384) and after (79,678) | Question says "closing … after shareholder dividends and share buy-backs"; add EV table pages |
| a03 | Note 5 / Note 7 give total OPAT as 7,171 (including non-controlling interests); 7,136 is the shareholders' share | Question says "attributable to shareholders"; add pages |
| a06 | p.2 only says the rounded "US$345 billion" | gold_pages → [42, 43, 156, 196] |
| a07 | No unit in the question; an answer of "HK$1.4408" would be scored wrong | Question says "(in Hong Kong cents)"; add pages, including the Directors' Report (p.94) and Note 13 (p.214) |
| a09 | p.70 gives both 25,981 (standard basis) and 27,524 (including four medical subsidiaries) | Question names the excluded entities |
| a10 | p.121 says "estimated remuneration payable" US$34.5m; Note 10 (p.207) says US$34m, which is outside the 0.5% tolerance | Question uses the p.121 wording |

## Questionable

| id | Problem | Fix |
|---|---|---|
| h11 | Only the signature page is marked | Add pp.6, 31, 33, 43 (p.33 gives her name as "CHAN Yiting, Bonnie") |
| h12 | Only p.103 is marked | Add pp.113, 138, 142 |
| a11 | The chairman changed during 2025 (Edmund Tse until 30 Sep; Sir Mark Edward Tucker from 1 Oct, p.95); the reference answer includes his full title | Ask "at 31 December 2025"; answer is the name only; add pp.13, 80, 95, 348 |
| a12 | Only the photo caption page is marked | Add pp.21, 80, 95, 303, 348 |
| a13 | pp.17 and 47 state the same policy | gold_pages → [17, 25, 47] |

## Extra pages for questions that passed

h01 +[12, 48, 75, 83, 153] · h02 +[12, 75, 81, 83, 146, 166] · h03 +[12, 83, 166] · h05 +[48] · h06 +[24, 46, 53, 83] · h07 +[18, 55] · h10 +[17, 32, 138] · a01 +[10, 16, 27, 30, 320, 325, 328, 329] · a04 +[39, 40, 41, 42, 158, 191, 213] · a05 +[195, 197, 201] · a08 +[16, 34, 196]

Reason: `hit_at_k` and `citation_page_precision` count a retrieved page as a miss when it states the answer but isn't in `gold_pages`.

The full reviewer report, with source quotes for every change, is in the conversation where this file was saved. The key quotes are copied into each question's `notes`.
