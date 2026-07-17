/* Minimal MiniLM (BERT) sentence-embedding encoder, pure C.
 *
 * Loads an all-MiniLM-L6-v2 GGUF (F16 or Q4_0), tokenizes text with a BERT
 * WordPiece tokenizer, runs the 6-layer encoder, mean-pools and L2-normalizes,
 * and returns the sentence embedding. Embeddings only -- no generation, no KV
 * cache, no llama.cpp. See llm.c for the intent block.
 */
#pragma once
#ifndef LLM_H
#define LLM_H

#include <stddef.h>

typedef struct llm llm_t;

/* Load a GGUF model. Returns NULL on failure. */
llm_t * llm_load(const char * gguf_path);

/* Release the model and all its buffers. */
void llm_free(llm_t * m);

/* Embedding dimension (384 for all-MiniLM-L6-v2). */
int llm_dim(const llm_t * m);

/* Encode text into a unit-normalized embedding of llm_dim() floats.
 * Returns the number of tokens encoded (> 0), or 0 on failure. */
int llm_embed(llm_t * m, const char * text, float * out);

/* Embed n texts in parallel into out[i * dim] (dim must be llm_dim()). Uses a
 * pthread pool of n_threads workers (n_threads <= 0 -> one per CPU, or the
 * LLM_THREADS env override). The model is shared read-only across threads.
 * Rows whose text fails to encode are zero-filled. */
void llm_embed_batch(llm_t * m, const char * const * texts, int n, int dim,
                     float * out, int n_threads);

/* Extra data blobs embedded in the GGUF (see llm_pack). Returns a pointer into
 * the model's mmap for the named blob and sets *size, or NULL if absent. */
const void * llm_extra(const llm_t * m, const char * name, size_t * size);

/* Append named blobs to a GGUF file as a trailer, replacing any prior trailer,
 * so the file stays a valid GGUF and also carries the blobs (read back via
 * llm_extra after llm_load). Returns 1 on success. */
int llm_pack(const char * gguf_path, const char * const * names,
             const void * const * blobs, const size_t * sizes, int n);

#endif
