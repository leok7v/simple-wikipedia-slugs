#!/usr/bin/env python3
"""Build the `slugs --index` input from a Simple English Wikipedia parquet.

Each article becomes one "pageid<TAB>title<TAB>body" line. The body is the text
we embed, chosen to be an article's tightest on-topic summary:

  * short/medium articles (<= LONG chars): the first ~cap characters, cut on a
    sentence boundary -- for the ~85% of articles under ~cap chars this is the
    whole article, and for the rest it is the lead plus the start of the body.

  * long articles (> LONG chars): title-guided *extractive* selection. The
    article is split into paragraphs; the title and every paragraph are
    embedded with our own MiniLM (the `llmembed` helper, i.e. llm.c built with
    -DLLM_MAIN); each paragraph is scored by cosine similarity to the title;
    the highest-scoring paragraphs -- in original order -- are kept up to ~cap
    chars. This drops the tangents a blind first-cap head would include and
    keeps the on-topic core, which fixes the "diffuse centroid" that a long
    article's mean-pooled embedding otherwise suffers from.

The only dependencies are pyarrow (to read the parquet) and numpy (bundled with
it); the embedder is our own C encoder, so no third-party ML stack is needed.
This is the one non-C step, and only when (re)building the index from parquet.

Usage: python3 extract.py <parquet> <llmembed> <model.gguf> [cap] [shards]
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

LONG = int(os.environ.get("SLUGS_LONG", "8000"))   # extractive above this len
EMBED_DIM = 384


def sentence_cap(body: str, cap: int) -> str:
    if len(body) > cap:
        cut = body[:cap]
        ends = [m.end() for m in re.finditer(r"[.!?]\s", cut)]
        body = cut[: ends[-1]].strip() if ends else cut
    return body


def lead(text: str, cap: int) -> str:
    return sentence_cap(re.sub(r"\s+", " ", text).strip(), cap)


def split_units(text: str) -> list[str]:
    """Coherent units for selection: paragraphs, falling back to sentences when
    an article has too few paragraph breaks to choose from."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p) > 1]
    if len(paras) < 4:
        flat = re.sub(r"\s+", " ", text).strip()
        paras = [p for p in re.split(r"(?<=[.!?])\s+", flat) if len(p) > 1]
    return paras


def embed_lines(lines: list[str], llmembed: str, model: str,
                shards: int) -> np.ndarray:
    """Embed each line with `llmembed --stdin`, sharded across `shards`
    processes, and return an (len(lines), EMBED_DIM) row-aligned array. Every
    line must be non-empty: llmembed drops empty input lines, which would
    misalign the rows, so that is asserted per shard."""
    assert all(lines), "empty line would misalign llmembed output"
    scratch = tempfile.mkdtemp(prefix="slugs_embed_")
    try:
        n = len(lines)
        chunk = (n + shards - 1) // shards
        procs = []
        for s in range(shards):
            lo, hi = s * chunk, min((s + 1) * chunk, n)
            if lo >= hi:
                break
            inp = os.path.join(scratch, f"in.{s}")
            out = os.path.join(scratch, f"out.{s}")
            with open(inp, "w") as f:
                f.write("\n".join(lines[lo:hi]) + "\n")
            fout = open(out, "w")
            proc = subprocess.Popen([llmembed, model, "--stdin"],
                                    stdin=open(inp), stdout=fout)
            procs.append((proc, fout, out, hi - lo))
        rows = []
        for i, (proc, fout, out, want) in enumerate(procs):
            proc.wait()
            fout.close()
            got = 0
            with open(out) as f:
                for line in f:
                    rows.append(np.array(line.split(), dtype=np.float32))
                    got += 1
            sys.stderr.write("  shard %d/%d done (%d rows)\n"
                             % (i + 1, len(procs), got))
            sys.stderr.flush()
            assert got == want, f"shard {i} misaligned: {got} != {want}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    emb = np.vstack(rows)
    assert emb.shape == (n, EMBED_DIM), f"embed shape {emb.shape}"
    return emb


def extract(units: list[str], tvec: np.ndarray, uvec: np.ndarray,
            cap: int) -> str:
    """Keep the paragraphs most similar to the title, in original order, up to
    cap chars. tvec is the title embedding; uvec the paragraph embeddings."""
    order = np.argsort(-(uvec @ tvec))
    chosen: list[int] = []
    total = 0
    for j in order:
        length = len(units[j]) + 1
        if total + length > cap and chosen:
            break
        chosen.append(int(j))
        total += length
    chosen.sort()
    return sentence_cap(" ".join(units[c] for c in chosen), cap)


def main() -> int:
    path, llmembed, model = sys.argv[1], sys.argv[2], sys.argv[3]
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    shards = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    table = pq.read_table(path, columns=["id", "title", "text"])
    ids = table.column("id").to_pylist()
    titles = table.column("title").to_pylist()
    texts = table.column("text").to_pylist()

    # Work-list for the long tail: per long article, one title line followed by
    # its paragraph units; remember each article's row range in the embedding.
    lines: list[str] = []
    plan: dict[int, tuple[int, list[str]]] = {}
    for i, (title, text) in enumerate(zip(titles, texts)):
        if text and len(text) > LONG:
            units = split_units(text)
            if len(units) >= 4:
                plan[i] = (len(lines), units)
                lines.append((title or "").strip() or "article")
                lines.extend(units)
    sys.stderr.write("embedding %d lines for %d long articles (%d shards)...\n"
                     % (len(lines), len(plan), shards))
    sys.stderr.flush()
    emb = embed_lines(lines, llmembed, model, shards) if lines else None

    out = []
    for i, (pid, title, text) in enumerate(zip(ids, titles, texts)):
        if not text:
            continue
        if i in plan:
            base, units = plan[i]
            body = extract(units, emb[base],
                           emb[base + 1: base + 1 + len(units)], cap)
        else:
            body = lead(text, cap)
        if body:
            name = (title or "").replace("\t", " ").replace("\n", " ")
            out.append(f"{pid}\t{name}\t{body}\n")
    sys.stdout.write("".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
