"""
src/ingest/chunk_spo.py — Session A2
Structure-aware chunking for the SPO regulations: split on § section boundaries
so each chunk holds a COMPLETE legal rule, instead of blind fixed-size windows
that slice rules in half. Oversized sections are sub-split to stay within the
embedding model's input limit.
"""
from __future__ import annotations

import re

MAX_WORDS = 350  # sections longer than this get sub-split


def chunk_spo_text(text: str) -> list[str]:
    """Split SPO text into section-aligned chunks."""
    # Drop table-of-contents lines (dotted leaders) and repeating footer/page lines
    # The processed text has flowed newlines, so work on the whole string.
    # Strip table-of-contents lines (dotted leaders) and repeating footer/page noise.
    full = re.sub(r"\.{3,}\s*\d*", " ", text)            # ToC dotted leaders (+ page num) -> space          # ToC dotted leaders
    full = re.sub(r"(?m)^SPO MA-TA.*$", "", full)         # footer
    full = re.sub(r"(?m)^Seite \d+ von \d+.*$", "", full)  # page numbers

    # Real section headers = "§ <num> <Title>", where the word after the number
    # is a capitalised title — NOT "Abs" (which marks an inline cross-reference
    # like "§ 17 Abs. 2"). This is what separates headers from references.
    header_re = re.compile(r"§\s*\d+[a-z]?\s+(?!Abs\b)[A-ZÄÖÜ][a-zäöüß]+")
    starts = [m.start() for m in header_re.finditer(full)]

    sections = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(full)
        body = full[start:end].strip()
        if len(body.split()) >= 20:   # keep real sections, skip short fragments
            sections.append(body)

    # Sub-split any section that is too long for the embedding model
        # Sub-split each section into its numbered clauses (1), (2), (3), ... so each
    # chunk is a small, focused rule — sharper embeddings, better retrieval.
    # The § header is prefixed onto every clause so it keeps its section context.
    chunks = []
    for sec in sections:
        m = re.match(r"(§\s*\d+[a-z]?\s+[^(§]{0,60})", sec)
        head = m.group(1).strip() if m else sec[:40]

        parts = re.split(r"(?=\(\d+\)\s)", sec)   # split at "(1) ", "(2) ", ...
        clause_chunks = []
        for p in parts:
            p = p.strip()
            if len(p.split()) < 5:
                continue
            # prefix the section header so a bare clause still carries its context
            clause_chunks.append(p if p.startswith(head[:12]) else f"{head}: {p}")

        # if a clause is still very long, fall back to word-window splitting on it
        for c in (clause_chunks or [sec]):
            words = c.split()
            if len(words) <= MAX_WORDS:
                chunks.append(c)
            else:
                for j in range(0, len(words), MAX_WORDS):
                    chunks.append(" ".join(words[j:j + MAX_WORDS]))
    return chunks