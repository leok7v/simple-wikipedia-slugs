# Simple English Wikipedia compact semantic index.
#
# The encoder is our own llm.c (pure C, auto-vectorized). Zero external
# libraries: just libc, libm and pthreads -- no llama.cpp, no BLAS/LAPACK (the
# PCA fit uses a hand-rolled Jacobi eigendecomposition).
#
# Querying is pure C (model + index). Rebuilding the index from the parquet is
# the only step that needs python3 + pyarrow: extract.py turns each article
# into its indexed body (see that file), and for long articles it calls the
# `llmembed` helper (llm.c built with -DLLM_MAIN) for title-guided extraction.
#
# Default target builds `slugs`, then the index artifacts (keys.bin, slugs.tbl,
# pca.bin). Re-running `make` is cheap: existing artifacts are not rebuilt.

UNAME_M := $(shell uname -m)

CC       ?= cc
PYTHON   ?= python3
PARQUET  := simple-wikipedia-20260301.parquet
MODEL    := minilm.gguf
# Character budget per indexed body (cut on a sentence boundary), and the
# number of embedding shards for the long-article extraction pass.
CAP      ?= 1200
SHARDS   ?= 30

CFLAGS := -O3 -std=c11 -Wall -Wextra -pthread
LIBS   := -lm -pthread

# Best SIMD for the local CPU (arm uses -mcpu, x86 uses -march).
ifneq (,$(filter arm64 aarch64,$(UNAME_M)))
	CFLAGS += -mcpu=native
else
	CFLAGS += -march=native
endif

SRC := slugs.c llm.c core.c

all: slugs index

slugs: $(SRC) llm.h core.h
	$(CC) $(CFLAGS) $(SRC) -o slugs $(LIBS)

# The single-text embedder used by extract.py for long-article extraction.
llmembed: llm.c core.c llm.h core.h
	$(CC) $(CFLAGS) -DLLM_MAIN llm.c core.c -o llmembed $(LIBS)

# Build the index artifacts from the parquet. keys.bin stands in for the trio
# (slugs.tbl and pca.bin are written by the same run).
index: keys.bin

keys.bin slugs.tbl pca.bin: slugs llmembed $(PARQUET) $(MODEL) extract.py
	$(PYTHON) extract.py $(PARQUET) ./llmembed $(MODEL) $(CAP) $(SHARDS) \
	    | ./slugs --index

# Bake the index into the model so it is a single self-contained file: after
# this, querying needs only `slugs` + `$(MODEL)` (no separate index files).
pack: index
	./slugs --pack $(MODEL)

clean:
	rm -f slugs llmembed keys.bin slugs.tbl pca.bin

.PHONY: all index pack clean
