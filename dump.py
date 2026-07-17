#!/usr/bin/env python3
"""Dump a Simple English Wikipedia parquet to TSV for `slugs --index`.

Each output line is "pageid<TAB>title<TAB>text"; text is truncated to the given
character budget and stripped of tabs/newlines so one article is one line.
This is the only non-C step, and only when (re)building the index from parquet.

Usage: python3 dump.py <parquet> [text_chars]
"""
from __future__ import annotations

import sys

import pyarrow.parquet as pq


def main() -> int:
    status = 1
    if len(sys.argv) < 2:
        sys.stderr.write("usage: dump.py <parquet> [text_chars]\n")
    else:
        path = sys.argv[1]
        text_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
        table = pq.read_table(path, columns=["id", "title", "text"])
        ids = table.column("id").to_pylist()
        titles = table.column("title").to_pylist()
        texts = table.column("text").to_pylist()
        write = sys.stdout.write
        for pid, title, text in zip(ids, titles, texts):
            body = (text or "").strip()
            if body:
                name = (title or "").replace("\t", " ").replace("\n", " ")
                body = body[:text_chars].replace("\t", " ")
                body = body.replace("\n", " ").replace("\r", " ")
                write(f"{pid}\t{name}\t{body}\n")
        status = 0
    return status


if __name__ == "__main__":
    raise SystemExit(main())
