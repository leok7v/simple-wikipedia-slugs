# Simple English Wikipedia compact semantic index.
#
# Zero llama.cpp: the encoder is our own llm.c (pure C + optional SIMD). The
# only external libraries are the system BLAS/LAPACK used for the PCA fit
# (Accelerate on macOS, OpenBLAS + LAPACK on Linux).
#
# Default target builds the `slugs` binary from *.c, then builds the index
# artifacts (keys.bin, slugs.tbl, pca.bin) from the bundled parquet. Re-running
# `make` is cheap: existing artifacts are not rebuilt.
#
# Zero external libraries: just libc, libm and pthreads. The PCA fit uses a
# hand-rolled Jacobi eigendecomposition, so there is no BLAS/LAPACK dependency.
# Querying is pure C (model + index); rebuilding the index from the parquet is
# the only step that needs python3 + pyarrow.

UNAME_M := $(shell uname -m)

CC       ?= cc
PYTHON   ?= python3
PARQUET  := simple-wikipedia-20260301.parquet
MODEL    := minilm.gguf
# Bytes of article text fed to the encoder (~510 tokens); the lead/intro.
TEXT_CHARS ?= 2500

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

# Build the index artifacts from the parquet. keys.bin stands in for the trio
# (slugs.tbl and pca.bin are written by the same run).
index: keys.bin

keys.bin slugs.tbl pca.bin: slugs $(PARQUET) $(MODEL) dump.py
	$(PYTHON) dump.py $(PARQUET) $(TEXT_CHARS) | ./slugs --index

# Bake the index into the model so it is a single self-contained file: after
# this, querying needs only `slugs` + `$(MODEL)` (no separate index files).
pack: index
	./slugs --pack $(MODEL)

clean:
	rm -f slugs keys.bin slugs.tbl pca.bin

.PHONY: all index pack clean
