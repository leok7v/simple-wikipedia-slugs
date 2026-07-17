# simple-wikipedia-slugs

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
    minilm.gguf                         the encoder model (~20 MB, Q4_0)
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

Encode. Each article's text (its lead, up to ~510 tokens) is embedded by
all-MiniLM-L6-v2 into a 384-dimensional unit vector. llm.c parses the GGUF,
dequantizes the Q4_0 / Q8_0 / F16 weights to f32, tokenizes with the BERT
WordPiece scheme, runs the 6 encoder blocks, mean-pools and L2-normalizes. It
matches llama.cpp's embeddings to cosine ~1.0 on clean English text.

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
(python3 + pyarrow, to read the parquet):

    make              # build the binary, then build the index (slow, one-time)
    make index        # just the index artifacts
    make pack         # build the index and embed it into minilm.gguf
    make clean        # remove binary + index artifacts

`make index` embeds all 279,678 articles once; the keys are tiny and searching
them is instant.


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
  - The tokenizer is English-pragmatic (ASCII lowercase, no Unicode NFD
    accent-stripping); it matches llama.cpp's MiniLM embeddings to cosine
    ~1.0 on clean English text, diverging slightly on accented / CJK input.
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
