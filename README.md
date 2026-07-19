---
license: apache-2.0
language:
  - en
base_model: sentence-transformers/all-MiniLM-L6-v2
base_model_relation: quantized
pipeline_tag: sentence-similarity
tags:
  - gguf
  - sentence-transformers
  - feature-extraction
  - sentence-similarity
  - semantic-search
  - embeddings
  - wikipedia
  - simple-english-wikipedia
---

# simple-wikipedia-slugs

`minilm.gguf` is all-MiniLM-L6-v2 quantized to Q4_0 and augmented, in one file,
with a compact semantic index over all 279,678 Simple English Wikipedia
articles: ask a question, get back the matching article ids. The rest of the
repo is a tiny pure-C reference tool that builds the index and queries the
model. Model weights are Apache-2.0; the Wikipedia-derived index is CC BY-SA
4.0.

A very small semantic index over the whole Simple English Wikipedia. Each
article is compressed to a single 256-bit fingerprint. A natural-language
query ("What is dark matter?") is turned into the same kind of fingerprint
and matched against every article by Hamming distance. The result is a short
list of article ids and titles that you then fetch, in full, from Wikipedia
online.

It is pure C with no dependencies: the text encoder (all-MiniLM-L6-v2, a
6-layer BERT) is written from scratch in llm.c, so there is no llama.cpp even
though it reads a GGUF model, and nothing to install.

    query --> MiniLM embedding (384d) --> PCA (256d) --> sign bits (32 bytes)
                            (llm.c)                            |
                                            Hamming scan over all articles
                                                             |
                                        top-K (pageid, title) --> fetch online


## Files

    slugs.c                             index builder + search (the CLI)
    llm.c / llm.h                       pure-C MiniLM encoder: GGUF loader,
                                        WordPiece tokenizer, 6-layer forward
                                        pass, threaded batch embedding
    core.c / core.h                     vendored containers: growable arrays,
                                        chars buffer, hash map, OOM malloc
    extract.py                          parquet -> index input: per-article body
                                        selection (extractive for long articles)
    minilm.gguf                         the encoder model (Q4_0) + baked index
    simple-wikipedia-20260301.parquet   source corpus, only used to (re)build
                                        the index: 279,678 articles
    pca.bin                             PCA mean + 256 components (384-d)
    keys.bin                            one 32-byte key per article
    slugs.tbl                           "pageid<TAB>title" per article, aligned
                                        with keys

To query you need only: the binary, `minilm.gguf`, and the three index files
(`pca.bin`, `keys.bin`, `slugs.tbl`). The parquet is the raw source the index
is built from and is not needed at query time.


## The model

`minilm.gguf` is [sentence-transformers/all-MiniLM-L6-v2][st] quantized to
Q4_0. Thanks and credit to:

  - the Sentence-Transformers team for all-MiniLM-L6-v2 (Apache-2.0),
  - [second-state][ss] for the GGUF conversion we started from,
  - llama.cpp's `llama-quantize` for the f16 -> Q4_0 step.

llm.c only *reads* the GGUF (weights + tokenizer vocab); llama.cpp is not a
build or runtime dependency.

[st]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[ss]: https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF


## The data

`simple-wikipedia-20260301.parquet` is the 2026-03-01 snapshot of Simple
English Wikipedia (279,678 articles; columns id, url, title, text), taken from
[omarkamali/wikipedia-monthly][wm] (config `20260301.simple`), which
repackages the official Wikimedia dumps into clean monthly parquet. Thanks to
Omar Kamali for maintaining it, and to the Wikipedia editors whose text this
is. Article text is CC BY-SA 4.0.

To update to a newer month, download the corresponding config from that
dataset and rebuild the index.

[wm]: https://huggingface.co/datasets/omarkamali/wikipedia-monthly


## How it works

Select. Each article is reduced to a compact, on-topic body (extract.py). A
short or medium article keeps its first ~1200 characters (cut on a sentence
boundary) -- for most articles that is the whole thing. A long article
(> 8000 chars) instead gets title-guided *extractive* selection: it is split
into paragraphs, the title and every paragraph are embedded, and the
paragraphs most similar to the title are kept (in original order). This drops
the tangents that otherwise blur a long article's mean-pooled vector, so
famous long topics (Einstein, World War II) still land on their own article.

Encode. That body is embedded by all-MiniLM-L6-v2 into a 384-dimensional unit
vector. llm.c parses the GGUF, dequantizes the Q4_0 / Q8_0 / F16 weights to
f32, tokenizes with the BERT WordPiece scheme, runs the 6 encoder blocks,
mean-pools and L2-normalizes. It matches llama.cpp's embeddings to cosine ~1.0
on clean English text.

Reduce. A PCA fitted over the corpus projects 384 dimensions down to 256,
keeping the directions of largest variance.

Sign-quantize. The sign of each of the 256 projected values becomes one bit.
256 bits = 32 bytes: the whole article, as a key.

Search. A query is encoded the same way. Its 32-byte key is compared to every
article's key with Hamming distance (popcount of xor) -- a few million byte
compares, done in well under a millisecond in plain C. The nearest keys are
the most semantically similar articles.

Why signs work. For two unit vectors with angle t between them, a random
hyperplane separates them with probability t/pi. Each PCA component is such a
hyperplane, so over 256 of them the expected Hamming distance is 256*t/pi and
    cos(t) ~= cos(pi * d / 256)
A distance near 0 means near-identical direction; 128 means orthogonal
(unrelated). The tool flags matches past distance 100 as low confidence.


## Building

Querying is pure C; you need only a C compiler:

    make slugs        # cc *.c -> slugs; that's the whole build

The dot-product kernel behind every matmul auto-vectorizes under
`-O3 -mcpu=native`. Embedding is multi-threaded (one worker per CPU; set
`LLM_THREADS` to override).

Building the index from the parquet is the only step that needs anything more
(python3 + pyarrow, to read the parquet). It still uses no third-party ML
stack: the long-article extraction embeds paragraphs with `llmembed`, which is
just llm.c built with -DLLM_MAIN.

    make              # build slugs + llmembed, then the index (slow, one-time)
    make index        # just the index artifacts
    make pack         # build the index and embed it into minilm.gguf
    make clean        # remove binaries + index artifacts

`make index` embeds all 279,678 articles once (plus the paragraphs of the
~5,500 long articles for extraction); the keys are tiny and searching them is
instant.


## Asking a question

    ./slugs "What is dark matter?"

prints the top matches, for example:

    (example output is filled in after the full index is built)

Each result is an article you fetch yourself from Simple English Wikipedia by
page id. For example, to retrieve "Astronomy" (id 48) to feed an LLM:

  As plain wikitext, minimum markup (no XML):

    https://simple.wikipedia.org/?curid=48&action=raw

  As JSON with the markup stripped to plain text:

    https://simple.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&pageids=48

  In a browser (rendered HTML page):

    https://simple.wikipedia.org/?curid=48

Flags:

    --top:N            number of results (default 5)
    --verbose          more detail
    --verbosity:N      0 silent .. 9 max (default 1)

A distance histogram for a query (how separable it is from the corpus):

    ./slugs --stats "What is dark matter?"


## Reading the distance

Each result carries a Hamming distance `d` (0 = identical direction, 128 =
unrelated). It is the only confidence signal -- the search has no "no match"
mode, it always returns the nearest keys, so read `d` to judge them:

    d < 60      strong match (usually the exact article)
    60 - 76     relevant / semantically close
    77 - 82     weak; possibly the best of a bad lot
    > 82        no confident match -- flagged [low]

Gibberish, or a concept absent from Simple English Wikipedia, has no cluster:
the whole corpus sits near d = 120-128 (orthogonal) and the "nearest" article
(d ~ 85-95) is just the closest of 279k near-random points -- an unrelated
short article, sometimes only a spelling echo of the query. Full-sentence
questions embed real meaning and match semantically; a single rare word embeds
mostly its subword spelling and tends to match look-alikes, not sense. Example
`--stats` output for the gibberish query "fkjsdfhshkn":

    nearest distance: 90 (of 256 bits)
      88.. 95            19  ( 0.01%)
      96..103           362  ( 0.13%)
     104..111  ##      5340  ( 1.91%)
     112..119  #####   33924  (12.13%)   <- corpus piled up near orthogonal
     120..127  ####### 91537  (32.73%)

Nothing sits below d = 88: no real neighbour exists.


## Query with phrases, not single words

The distance is a nearest-neighbour signal, so a query always returns
something; the *meaning* comes from context. A full question ("What is dark
matter?") embeds enough meaning to match semantically. A single rare or
out-of-corpus word embeds mostly its own spelling -- its subword (WordPiece)
tokens dominate the vector -- so it retrieves titles that merely *look* like
it, not ones about it. Real words with no Simple-Wikipedia article are matched
by surface form:

    borborygmus  ->  "Bo..." towns   (Boligee, Bordj Bou Arreridj)
    susurrus     ->  "Su..." places  (Sucre, Sukkur, Sufers)
    tsundoku     ->  Japanese names  (Maki Kaji, Nobuo Kawaguchi)

So embedding search generalises where there is context (phrases, questions)
and degrades to spelling-match for isolated rare terms.

When wiring this in as an LLM / MCP tool, instruct the calling model to:

  - phrase the query as a sentence or question, not a bare keyword;
  - treat a `[low]` result (d > 82) as "no good match" and answer from its own
    knowledge rather than the returned articles.


## One self-contained file

The three index artifacts (`pca.bin`, `keys.bin`, `slugs.tbl`) are small, so
`slugs --pack minilm.gguf` (or `make pack`) appends them to the model as a
trailer. GGUF readers ignore trailing bytes, so the file stays a valid GGUF
(llama.cpp still loads it), while `slugs` reads the embedded index straight
from the mmap. After packing, a single file -- `minilm.gguf` -- is a complete
"query Simple English Wikipedia" engine:

    make pack
    rm -f pca.bin keys.bin slugs.tbl  # not needed; index is in the gguf
    ./slugs "What is dark matter?"        # answers from the embedded index

Queries prefer the embedded index and fall back to the on-disk files if the
model has no trailer. That makes it trivial to wire into an LLM tool / MCP
server: ship the one gguf, take a question, run the search, and for each hit
return the article text via

    https://simple.wikipedia.org/?curid=<id>&action=raw   (wikitext)
    .../w/api.php?action=query&prop=extracts&explaintext=1&format=json&pageids=<id>   (JSON)

so the LLM answers from the fetched, always-current article text.


## Notes

  - The index stores only pageid and title, never article text. Text is
    fetched live from Wikipedia, so it is always current and carries the
    real markup.
  - The tokenizer lowercases ASCII and folds Latin accents to their base
    letter, BERT-style (NFD + combining-mark drop: "Zürich" -> "zurich",
    "Marić" -> "maric"); it matches llama.cpp's MiniLM embeddings to cosine
    ~1.0 on English and accented-Latin text. Non-Latin scripts (CJK, Greek)
    still fall back to [UNK].
  - Rebuilding for a newer dump: replace the parquet, `make clean`, `make`.
  - SLUGS_MODEL overrides the model path (default minilm.gguf).


## Citations

The corpus -- Wikipedia Monthly:

    @misc{wikipedia_kamali_2025,
        author = { Omar Kamali and Omneity Labs },
        title = { Wikipedia Monthly },
        year = { 2025 },
        url = { https://huggingface.co/datasets/omarkamali/wikipedia-monthly },
        doi = { 10.57967/hf/6575 },
        publisher = { Hugging Face }
    }

The encoder -- Sentence-BERT / all-MiniLM-L6-v2 (https://www.sbert.net/):

    @inproceedings{reimers-2019-sentence-bert,
        title     = "Sentence-BERT: Sentence Embeddings using
                     Siamese BERT-Networks",
        author    = "Reimers, Nils and Gurevych, Iryna",
        booktitle = "Proceedings of the 2019 Conference on Empirical Methods in
                     Natural Language Processing",
        month     = "11",
        year      = "2019",
        publisher = "Association for Computational Linguistics",
        url       = "https://arxiv.org/abs/1908.10084"
    }
