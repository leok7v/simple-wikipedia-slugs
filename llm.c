/* Minimal MiniLM (BERT) sentence-embedding encoder, pure C.
 *
 * Turns text into a 384-d unit vector via all-MiniLM-L6-v2. The model is small
 * and fixed, so the whole runtime is ~one file: parse the GGUF, dequantize the
 * Q4_0/Q8_0/F16 weights to f32 at load, tokenize with BERT WordPiece, run 6
 * encoder blocks, mean-pool, L2-normalize.
 *
 * The dot product behind every linear layer is a plain restrict loop left for
 * the compiler to auto-vectorize under -O3 -mcpu=native.
 *
 * Storage uses core.h containers (chars / i32 array / map): token lists and
 * word buffers grow on demand, the vocab is a hash map, and allocations go
 * through core_oom (fatal on failure).
 *
 * Not implemented (not needed for embeddings): generation, sampling, KV cache,
 * causal masking. BERT attention here is full/bidirectional.
 *
 * Tokenizer scope: English-pragmatic. Lowercases ASCII, splits on ASCII
 * whitespace/punctuation, greedy WordPiece against the GGUF vocab. No Unicode
 * NFD accent-stripping, so accented / CJK input diverges from the reference
 * tokenizer (rare in Simple English; falls back to [UNK]).
 *
 * Style: snake_case, star-attached pointers, single status flag, no early
 * exits.
 */

#define _GNU_SOURCE   /* pread/getline: glibc hides POSIX under -std=c11 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "core.h"
#include "llm.h"

#define GGML_TYPE_F32       0
#define GGML_TYPE_F16       1
#define GGML_TYPE_Q4_0      2
#define GGML_TYPE_Q8_0      8
#define QK4_0              32
#define Q4_0_BYTES         18        /* fp16 scale (2) + 16 packed nibbles */
#define QK8_0              32
#define Q8_0_BYTES         34        /* fp16 scale (2) + 32 int8 quants */
#define BERT_MAX_WORD     100  /* BERT max_input_chars_per_word */
#define GGUF_MAX_DIMS       4        /* GGUF tensor rank limit */
#define GGUF_MAGIC 0x46554747u       /* "GGUF" little-endian */

/* Optional trailer appended after the GGUF tensor data (see llm_pack): named
 * blobs, then their descriptor entries, then a fixed tail with the magic last.
 * GGUF readers ignore trailing bytes, so the file stays a valid GGUF. */
#define SLGX_MAGIC 0x58474c53u       /* "SLGX" little-endian */
#define SLGX_MAX            8

typedef struct { char name[16]; uint64_t off; uint64_t size; } slgx_entry_t;
typedef struct {
    uint64_t base_size; uint32_t count; uint32_t magic;
} slgx_tail_t;

define_array(int, i32);

// # Half precision

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            exp = 127u - 15u + 1u;
            while ((mant & 0x400u) == 0u) { mant <<= 1; exp -= 1u; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* Dequantize n elements of the given ggml type into dst (f32). */
static void dequant(int type, const void * src, int64_t n, float * dst) {
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t)n * sizeof(float));
    } else if (type == GGML_TYPE_F16) {
        const uint16_t * h = (const uint16_t *)src;
        for (int64_t i = 0; i < n; i++) { dst[i] = half_to_float(h[i]); }
    } else if (type == GGML_TYPE_Q4_0) {
        const uint8_t * p = (const uint8_t *)src;
        int64_t nb = n / QK4_0;
        assert(n % QK4_0 == 0);
        for (int64_t b = 0; b < nb; b++) {
            uint16_t dh;
            memcpy(&dh, p, 2);
            float d = half_to_float(dh);
            const uint8_t * qs = p + 2;
            float * o = dst + b * QK4_0;
            for (int j = 0; j < QK4_0 / 2; j++) {
                o[j]             = (float)((int)(qs[j] & 0x0Fu) - 8) * d;
                o[j + QK4_0 / 2] = (float)((int)(qs[j] >> 4)    - 8) * d;
            }
            p += Q4_0_BYTES;
        }
    } else if (type == GGML_TYPE_Q8_0) {
        const uint8_t * p = (const uint8_t *)src;
        int64_t nb = n / QK8_0;
        assert(n % QK8_0 == 0);
        for (int64_t b = 0; b < nb; b++) {
            uint16_t dh;
            memcpy(&dh, p, 2);
            float d = half_to_float(dh);
            const int8_t * qs = (const int8_t *)(p + 2);
            float * o = dst + b * QK8_0;
            for (int j = 0; j < QK8_0; j++) { o[j] = (float)qs[j] * d; }
            p += Q8_0_BYTES;
        }
    } else {
        assert(0 && "unsupported ggml tensor type");
    }
}

// # GGUF parsing

typedef struct {
    const char * name;
    int          name_len;
    int          type;
    int          n_dims;
    int64_t      ne[GGUF_MAX_DIMS];
    uint64_t     offset;
} tensor_info_t;

typedef struct { const uint8_t * p; } cur_t;

static uint32_t rd_u32(cur_t * c) {
    uint32_t v;
    memcpy(&v, c->p, 4);
    c->p += 4;
    return v;
}

static uint64_t rd_u64(cur_t * c) {
    uint64_t v;
    memcpy(&v, c->p, 8);
    c->p += 8;
    return v;
}

static float rd_f32(cur_t * c) {
    float v;
    memcpy(&v, c->p, 4);
    c->p += 4;
    return v;
}

/* Read a GGUF string: uint64 length then raw bytes (not null-terminated). */
static const char * rd_str(cur_t * c, int * len) {
    uint64_t n = rd_u64(c);
    const char * s = (const char *)c->p;
    c->p += n;
    *len = (int)n;
    return s;
}

static size_t gguf_scalar_size(uint32_t type) {
    size_t s;
    switch (type) {
        case 0: case 1: case 7:    s = 1; break;  /* u8 i8 bool */
        case 2: case 3:            s = 2; break;  /* u16 i16 */
        case 4: case 5: case 6:    s = 4; break;  /* u32 i32 f32 */
        case 10: case 11: case 12: s = 8; break;  /* u64 i64 f64 */
        default:                   s = 0; break;
    }
    return s;
}

/* Advance past one metadata value of the given type (arrays recurse). */
static void skip_value(cur_t * c, uint32_t type) {
    if (type == 8) {
        int len;
        (void)rd_str(c, &len);
    } else if (type == 9) {
        uint32_t elem_type = rd_u32(c);
        uint64_t count = rd_u64(c);
        for (uint64_t i = 0; i < count; i++) { skip_value(c, elem_type); }
    } else {
        c->p += gguf_scalar_size(type);
    }
}

// # Model

typedef struct {
    float * wq; float * bq;
    float * wk; float * bk;
    float * wv; float * bv;
    float * wo; float * bo;
    float * attn_norm_w; float * attn_norm_b;
    float * w_up;   float * b_up;
    float * w_down; float * b_down;
    float * out_norm_w; float * out_norm_b;
} layer_t;

struct llm {
    int      fd;
    void *   map;
    size_t   map_size;
    int      n_layer;
    int      n_embd;
    int      n_head;
    int      n_ff;
    int      n_ctx;
    float    ln_eps;
    int      cls_id;
    int      sep_id;
    int      unk_id;
    float *  tok_emb;
    float *  pos_emb;
    float *  type_emb;
    float *  emb_norm_w;
    float *  emb_norm_b;
    layer_t * layers;
    int          vocab_count;
    struct map   vocab;            /* token bytes -> int id */
    tensor_info_t * tensors;
    int             tensor_count;
    const uint8_t * data;
    float **        owned;         /* dequantized weight buffers */
    int             owned_count;
    int             extra_count;   /* blobs embedded in the GGUF trailer */
    struct {
        char name[16]; const uint8_t * ptr; size_t size;
    } extra[SLGX_MAX];
};

static int vocab_id(struct llm * m, const char * s, int len) {
    struct chars key = { (char *)s, (size_t)len, 0 };
    int * v = map_get(&m->vocab, &key);
    return (v != NULL) ? *v : -1;
}

// # Tokenizer

static int is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\f' || c == '\v';
}

static int is_punct(unsigned char c) {
    int r = 0;
    if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
        (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
        r = 1;
    }
    return r;
}

static char lower_ascii(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
}

/* llama.cpp WPM: prepend U+2581 ("phantom space") to the word, then greedy
 * longest-match from the front over the prefixed bytes. Word-initial pieces
 * carry the prefix in the vocab; continuations are bare. word1 is a reused
 * buffer. Appends ids to pieces; returns 1 if the whole word matched. */
static int wordpiece(struct llm * m, const char * w, int wlen,
                     struct chars * word1, struct i32 * pieces) {
    word1->count = 0;
    chars_puts(word1, "\xe2\x96\x81");   /* U+2581 */
    chars_put(word1, w, (size_t)wlen);
    int n = (int)word1->count;
    size_t base = pieces->count;
    int i = 0;
    while (i < n) {
        int j = n;
        int found = -1;
        while (j > i && found == -1) {
            found = vocab_id(m, word1->data + i, j - i);
            if (found == -1) { j -= 1; }
        }
        if (found == -1) {
            pieces->count = base;   /* discard partial pieces */
            i = n + 1;              /* sentinel: word is out-of-vocab */
        } else {
            i32_put(pieces, found);
            i = j;
        }
    }
    return i == n;
}

/* Append a word's tokens to ids (its WordPiece pieces, or one [UNK]). */
static void add_word(struct llm * m, const char * w, int wlen,
                     struct chars * word1, struct i32 * pieces,
                     struct i32 * ids) {
    pieces->count = 0;
    int known = wordpiece(m, w, wlen, word1, pieces);
    if (known) {
        for (size_t p = 0; p < pieces->count; p++) {
            i32_put(ids, pieces->data[p]);
        }
    } else {
        i32_put(ids, m->unk_id);
    }
}

/* Consume one whitespace/punctuation-delimited word from text[*i], lowercasing
 * ASCII into word. Returns the word length; advances *i past it. */
static int scan_word(const char * text, int len, int * i,
                     struct chars * word) {
    word->count = 0;
    while (*i < len && !is_space((unsigned char)text[*i]) &&
           !is_punct((unsigned char)text[*i])) {
        char c = lower_ascii((unsigned char)text[*i]);
        chars_put(word, &c, 1);
        *i += 1;
    }
    return (int)word->count;
}

/* Tokenize text into ids: [CLS] word-pieces... [SEP], truncated to n_ctx. */
static void tokenize(struct llm * m, const char * text, struct i32 * ids) {
    i32_put(ids, m->cls_id);
    int len = (int)strlen(text);
    int i = 0;
    struct chars word = {0};
    struct chars word1 = {0};
    struct i32   pieces = {0};
    while (i < len && (int)ids->count < m->n_ctx - 1) {
        unsigned char c = (unsigned char)text[i];
        if (is_space(c)) {
            i += 1;
        } else if (is_punct(c)) {
            char pc = (char)c;
            add_word(m, &pc, 1, &word1, &pieces, ids);
            i += 1;
        } else {
            int wl = scan_word(text, len, &i, &word);
            add_word(m, word.data, wl, &word1, &pieces, ids);
        }
    }
    i32_put(ids, m->sep_id);
    if ((int)ids->count > m->n_ctx) { ids->count = (size_t)m->n_ctx; }
    chars_free(&word);
    chars_free(&word1);
    i32_free(&pieces);
}

// # Math kernels

static float dotf(const float * restrict a, const float * restrict b, int n) {
    float s = 0.0f;
#pragma clang loop vectorize(enable)
    for (int i = 0; i < n; i++) { s += a[i] * b[i]; }
    return s;
}

/* y[out] = W[out][in] . x[in] + b[out] */
static void linear(const float * w, const float * b, const float * x,
                   int in, int out, float * y) {
    for (int o = 0; o < out; o++) {
        float bias = (b != NULL) ? b[o] : 0.0f;
        y[o] = dotf(w + (size_t)o * in, x, in) + bias;
    }
}

static void layer_norm(float * x, const float * w, const float * b,
                       int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) { mean += x[i]; }
    mean /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) { x[i] = (x[i] - mean) * inv * w[i] + b[i]; }
}

static void gelu_inplace(float * x, int n) {
    const float k = 0.7978845608028654f;   /* sqrt(2/pi) */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
    }
}

static void softmax(float * s, int n) {
    float mx = s[0];
    for (int i = 1; i < n; i++) { if (s[i] > mx) { mx = s[i]; } }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        s[i] = expf(s[i] - mx);
        sum += s[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) { s[i] *= inv; }
}

// # Forward pass

/* One query's attention over all T keys/values within a single head. k and v
 * are already offset to this head; out is this head's context slice. */
static void attend_query(const float * qi, const float * k, const float * v,
                         int T, int hd, int n_embd, float scale,
                         float * scores, float * out) {
    for (int j = 0; j < T; j++) {
        scores[j] = dotf(qi, k + (size_t)j * n_embd, hd) * scale;
    }
    softmax(scores, T);
    for (int d = 0; d < hd; d++) { out[d] = 0.0f; }
    for (int j = 0; j < T; j++) {
        float wj = scores[j];
        const float * vj = v + (size_t)j * n_embd;
        for (int d = 0; d < hd; d++) { out[d] += wj * vj[d]; }
    }
}

typedef struct {
    float * q;
    float * k;
    float * v;
    float * ctx;
    float * scores;
    float * tmp;
    float * ff;
} scratch_t;

static void self_attention(const layer_t * L, const float * x, int T,
                           int n_embd, int n_head, scratch_t * s) {
    int hd = n_embd / n_head;
    float scale = 1.0f / sqrtf((float)hd);
    for (int t = 0; t < T; t++) {
        const float * xt = x + (size_t)t * n_embd;
        linear(L->wq, L->bq, xt, n_embd, n_embd, s->q + (size_t)t * n_embd);
        linear(L->wk, L->bk, xt, n_embd, n_embd, s->k + (size_t)t * n_embd);
        linear(L->wv, L->bv, xt, n_embd, n_embd, s->v + (size_t)t * n_embd);
    }
    for (int h = 0; h < n_head; h++) {
        int off = h * hd;
        for (int i = 0; i < T; i++) {
            const float * qi = s->q + (size_t)i * n_embd + off;
            attend_query(qi, s->k + off, s->v + off, T, hd, n_embd, scale,
                         s->scores, s->ctx + (size_t)i * n_embd + off);
        }
    }
}

/* BERT post-LN block: x = LN(x + Attn(x)); x = LN(x + FFN(x)). */
static void encoder_layer(const layer_t * L, float * x, int T, int n_embd,
                          int n_head, int n_ff, float eps, scratch_t * s) {
    self_attention(L, x, T, n_embd, n_head, s);
    for (int t = 0; t < T; t++) {
        float * xt = x + (size_t)t * n_embd;
        float * cx = s->ctx + (size_t)t * n_embd;
        linear(L->wo, L->bo, cx, n_embd, n_embd, s->tmp);
        for (int d = 0; d < n_embd; d++) { xt[d] += s->tmp[d]; }
        layer_norm(xt, L->attn_norm_w, L->attn_norm_b, n_embd, eps);
        linear(L->w_up, L->b_up, xt, n_embd, n_ff, s->ff);
        gelu_inplace(s->ff, n_ff);
        linear(L->w_down, L->b_down, s->ff, n_ff, n_embd, s->tmp);
        for (int d = 0; d < n_embd; d++) { xt[d] += s->tmp[d]; }
        layer_norm(xt, L->out_norm_w, L->out_norm_b, n_embd, eps);
    }
}

static void embed_tokens(const struct llm * m, const int * ids, int T,
                         float * x) {
    for (int t = 0; t < T; t++) {
        const float * te = m->tok_emb + (size_t)ids[t] * m->n_embd;
        const float * pe = m->pos_emb + (size_t)t * m->n_embd;
        float * xt = x + (size_t)t * m->n_embd;
        for (int d = 0; d < m->n_embd; d++) {
            xt[d] = te[d] + pe[d] + m->type_emb[d];
        }
        layer_norm(xt, m->emb_norm_w, m->emb_norm_b, m->n_embd, m->ln_eps);
    }
}

static void mean_pool_norm(const float * x, int T, int n, float * out) {
    for (int d = 0; d < n; d++) { out[d] = 0.0f; }
    for (int t = 0; t < T; t++) {
        const float * xt = x + (size_t)t * n;
        for (int d = 0; d < n; d++) { out[d] += xt[d]; }
    }
    float inv_t = 1.0f / (float)T;
    for (int d = 0; d < n; d++) { out[d] *= inv_t; }
    float ss = 0.0f;
    for (int d = 0; d < n; d++) { ss += out[d] * out[d]; }
    float inv = (ss > 0.0f) ? 1.0f / sqrtf(ss) : 0.0f;
    for (int d = 0; d < n; d++) { out[d] *= inv; }
}

static void scratch_alloc(scratch_t * s, int T, int n_embd, int n_ff) {
    size_t te = (size_t)T * n_embd;
    s->q      = core_oom(malloc(te * sizeof(float)));
    s->k      = core_oom(malloc(te * sizeof(float)));
    s->v      = core_oom(malloc(te * sizeof(float)));
    s->ctx    = core_oom(malloc(te * sizeof(float)));
    s->scores = core_oom(malloc((size_t)T * sizeof(float)));
    s->tmp    = core_oom(malloc((size_t)n_embd * sizeof(float)));
    s->ff     = core_oom(malloc((size_t)n_ff * sizeof(float)));
}

static void scratch_free(scratch_t * s) {
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->ctx);
    free(s->scores);
    free(s->tmp);
    free(s->ff);
}

int llm_embed(struct llm * m, const char * text, float * out) {
    struct i32 ids = {0};
    tokenize(m, text, &ids);
    int T = (int)ids.count;
    if (T > 0) {
        float * x = core_oom(malloc((size_t)T * m->n_embd * sizeof(float)));
        scratch_t s;
        scratch_alloc(&s, T, m->n_embd, m->n_ff);
        embed_tokens(m, ids.data, T, x);
        for (int l = 0; l < m->n_layer; l++) {
            encoder_layer(&m->layers[l], x, T, m->n_embd, m->n_head, m->n_ff,
                          m->ln_eps, &s);
        }
        mean_pool_norm(x, T, m->n_embd, out);
        scratch_free(&s);
        free(x);
    }
    i32_free(&ids);
    return T;
}

int llm_dim(const struct llm * m) {
    return m->n_embd;
}

// # Parallel batch embedding

/* Each worker handles a strided slice (i = start, start+stride, ...) so long
 * and short articles interleave across threads for balance. llm_embed is
 * thread-safe on a shared model: weights and the vocab map are read-only and
 * all scratch is per-call. */
typedef struct {
    struct llm *         m;
    const char * const * texts;
    int                  n;
    int                  dim;
    float *              out;
    int                  start;
    int                  stride;
} batch_job_t;

static void * batch_worker(void * arg) {
    batch_job_t * j = (batch_job_t *)arg;
    for (int i = j->start; i < j->n; i += j->stride) {
        float * row = j->out + (size_t)i * j->dim;
        int t = llm_embed(j->m, j->texts[i], row);
        if (t <= 0) {
            for (int d = 0; d < j->dim; d++) { row[d] = 0.0f; }
        }
    }
    return NULL;
}

static int resolve_threads(int requested, int n) {
    int nt = requested;
    if (nt <= 0) {
        const char * env = getenv("LLM_THREADS");
        if (env != NULL) {
            nt = atoi(env);
        } else {
            long c = sysconf(_SC_NPROCESSORS_ONLN);
            nt = (c > 0) ? (int)c : 1;
        }
    }
    if (nt < 1) { nt = 1; }
    if (nt > n) { nt = (n > 0) ? n : 1; }
    return nt;
}

void llm_embed_batch(struct llm * m, const char * const * texts, int n,
                     int dim, float * out, int n_threads) {
    int nt = resolve_threads(n_threads, n);
    pthread_t *   th   = core_oom(malloc((size_t)nt * sizeof(pthread_t)));
    batch_job_t * jobs = core_oom(malloc((size_t)nt * sizeof(batch_job_t)));
    for (int t = 0; t < nt; t++) {
        jobs[t] = (batch_job_t){ m, texts, n, dim, out, t, nt };
        pthread_create(&th[t], NULL, batch_worker, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) { pthread_join(th[t], NULL); }
    free(th);
    free(jobs);
}

// # Weight loading

static int find_tensor(const struct llm * m, const char * name) {
    int idx = -1;
    int len = (int)strlen(name);
    for (int i = 0; i < m->tensor_count && idx == -1; i++) {
        if (m->tensors[i].name_len == len &&
            memcmp(m->tensors[i].name, name, (size_t)len) == 0) {
            idx = i;
        }
    }
    return idx;
}

/* Dequantize a named tensor to a fresh f32 buffer, tracked for free. */
static float * weight(struct llm * m, const char * name) {
    float * out = NULL;
    int idx = find_tensor(m, name);
    if (idx >= 0) {
        const tensor_info_t * t = &m->tensors[idx];
        int64_t n = 1;
        for (int d = 0; d < t->n_dims; d++) { n *= t->ne[d]; }
        out = core_oom(malloc((size_t)n * sizeof(float)));
        dequant(t->type, m->data + t->offset, n, out);
        m->owned[m->owned_count] = out;
        m->owned_count += 1;
    }
    return out;
}

static float * weight_layer(struct llm * m, int l, const char * suffix) {
    struct chars name = {0};
    chars_printf(&name, "blk.%d.%s", l, suffix);
    float * w = weight(m, name.data);
    chars_free(&name);
    return w;
}

static void weights_load(struct llm * m) {
    m->tok_emb    = weight(m, "token_embd.weight");
    m->pos_emb    = weight(m, "position_embd.weight");
    m->type_emb   = weight(m, "token_types.weight");
    m->emb_norm_w = weight(m, "token_embd_norm.weight");
    m->emb_norm_b = weight(m, "token_embd_norm.bias");
    m->layers = core_oom(calloc((size_t)m->n_layer, sizeof(layer_t)));
    for (int l = 0; l < m->n_layer; l++) {
        layer_t * L = &m->layers[l];
        L->wq = weight_layer(m, l, "attn_q.weight");
        L->bq = weight_layer(m, l, "attn_q.bias");
        L->wk = weight_layer(m, l, "attn_k.weight");
        L->bk = weight_layer(m, l, "attn_k.bias");
        L->wv = weight_layer(m, l, "attn_v.weight");
        L->bv = weight_layer(m, l, "attn_v.bias");
        L->wo = weight_layer(m, l, "attn_output.weight");
        L->bo = weight_layer(m, l, "attn_output.bias");
        L->attn_norm_w = weight_layer(m, l, "attn_output_norm.weight");
        L->attn_norm_b = weight_layer(m, l, "attn_output_norm.bias");
        L->w_up   = weight_layer(m, l, "ffn_up.weight");
        L->b_up   = weight_layer(m, l, "ffn_up.bias");
        L->w_down = weight_layer(m, l, "ffn_down.weight");
        L->b_down = weight_layer(m, l, "ffn_down.bias");
        L->out_norm_w = weight_layer(m, l, "layer_output_norm.weight");
        L->out_norm_b = weight_layer(m, l, "layer_output_norm.bias");
    }
}

// # GGUF header + KV pass

static void read_tokens(struct llm * m, cur_t * c) {
    uint32_t elem_type = rd_u32(c);
    uint64_t count = rd_u64(c);
    assert(elem_type == 8);
    map_init(&m->vocab, MAP_KEY_CHARS, sizeof(struct chars),
             sizeof(int), NULL);
    for (uint64_t i = 0; i < count; i++) {
        int len;
        const char * s = rd_str(c, &len);
        struct chars key = { (char *)s, (size_t)len, 0 };
        int id = (int)i;
        map_put(&m->vocab, &key, &id);
    }
    m->vocab_count = (int)count;
}

static int key_is(const char * k, int klen, const char * name) {
    return klen == (int)strlen(name) && memcmp(k, name, (size_t)klen) == 0;
}

/* Parse the metadata KV block, capturing arch params + tokenizer tokens.
 * Leaves the cursor at the start of the tensor-info table. */
static void parse_kv(struct llm * m, cur_t * c, uint64_t kv_count) {
    for (uint64_t i = 0; i < kv_count; i++) {
        int klen;
        const char * key = rd_str(c, &klen);
        uint32_t vt = rd_u32(c);
        if (key_is(key, klen, "bert.block_count")) {
            m->n_layer = (int)rd_u32(c);
        } else if (key_is(key, klen, "bert.embedding_length")) {
            m->n_embd = (int)rd_u32(c);
        } else if (key_is(key, klen, "bert.feed_forward_length")) {
            m->n_ff = (int)rd_u32(c);
        } else if (key_is(key, klen, "bert.attention.head_count")) {
            m->n_head = (int)rd_u32(c);
        } else if (key_is(key, klen, "bert.context_length")) {
            m->n_ctx = (int)rd_u32(c);
        } else if (key_is(key, klen, "bert.attention.layer_norm_epsilon")) {
            m->ln_eps = rd_f32(c);
        } else if (key_is(key, klen, "tokenizer.ggml.cls_token_id")) {
            m->cls_id = (int)rd_u32(c);
        } else if (key_is(key, klen, "tokenizer.ggml.seperator_token_id")) {
            m->sep_id = (int)rd_u32(c);
        } else if (key_is(key, klen, "tokenizer.ggml.unknown_token_id")) {
            m->unk_id = (int)rd_u32(c);
        } else if (key_is(key, klen, "tokenizer.ggml.tokens")) {
            read_tokens(m, c);
        } else {
            skip_value(c, vt);
        }
    }
}

/* Read the tensor-info table and locate the aligned data section. */
static void parse_tensors(struct llm * m, cur_t * c, uint64_t tensor_count) {
    m->tensor_count = (int)tensor_count;
    m->tensors = core_oom(malloc(tensor_count * sizeof(tensor_info_t)));
    for (uint64_t i = 0; i < tensor_count; i++) {
        tensor_info_t * t = &m->tensors[i];
        t->name = rd_str(c, &t->name_len);
        t->n_dims = (int)rd_u32(c);
        assert(t->n_dims <= GGUF_MAX_DIMS);
        for (int d = 0; d < t->n_dims; d++) { t->ne[d] = (int64_t)rd_u64(c); }
        t->type = (int)rd_u32(c);
        t->offset = rd_u64(c);
    }
    size_t pos = (size_t)(c->p - (const uint8_t *)m->map);
    size_t align = 32;
    size_t aligned = (pos + align - 1) / align * align;
    m->data = (const uint8_t *)m->map + aligned;
}

// # Load / free

static void * map_file(const char * path, int * fd, size_t * size) {
    void * map = MAP_FAILED;
    *fd = open(path, O_RDONLY);
    if (*fd >= 0) {
        struct stat st;
        if (fstat(*fd, &st) == 0 && st.st_size > 0) {
            *size = (size_t)st.st_size;
            map = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
        }
    }
    return (map == MAP_FAILED) ? NULL : map;
}

/* Discover a blob trailer (see llm_pack); record pointers into the mmap. */
static void parse_embedded(struct llm * m) {
    if (m->map_size >= sizeof(slgx_tail_t)) {
        const uint8_t * base = (const uint8_t *)m->map;
        slgx_tail_t tail;
        memcpy(&tail, base + m->map_size - sizeof(tail), sizeof(tail));
        size_t ents = (size_t)tail.count * sizeof(slgx_entry_t) + sizeof(tail);
        if (tail.magic == SLGX_MAGIC && tail.count <= SLGX_MAX &&
            ents <= m->map_size) {
            const uint8_t * ep = base + m->map_size - ents;
            for (uint32_t i = 0; i < tail.count; i++) {
                slgx_entry_t e;
                memcpy(&e, ep + (size_t)i * sizeof(e), sizeof(e));
                if (e.off + e.size <= m->map_size) {
                    memcpy(m->extra[m->extra_count].name, e.name, 16);
                    m->extra[m->extra_count].ptr  = base + e.off;
                    m->extra[m->extra_count].size = (size_t)e.size;
                    m->extra_count += 1;
                }
            }
        }
    }
}

const void * llm_extra(const struct llm * m, const char * name,
                       size_t * size) {
    const void * out = NULL;
    for (int i = 0; i < m->extra_count && out == NULL; i++) {
        if (strncmp(m->extra[i].name, name, 16) == 0) {
            out = m->extra[i].ptr;
            if (size != NULL) { *size = m->extra[i].size; }
        }
    }
    return out;
}

/* Append blobs as a trailer, replacing any existing one. Idempotent. */
int llm_pack(const char * path, const char * const * names,
             const void * const * blobs, const size_t * sizes, int n) {
    int status = 0;
    int fd = open(path, O_RDWR);
    if (fd >= 0 && n <= SLGX_MAX) {
        struct stat st;
        if (fstat(fd, &st) == 0) {
            uint64_t base = (uint64_t)st.st_size;
            slgx_tail_t old;
            if (st.st_size >= (off_t)sizeof(old) &&
                pread(fd, &old, sizeof(old), st.st_size - (off_t)sizeof(old))
                    == (ssize_t)sizeof(old) && old.magic == SLGX_MAGIC) {
                base = old.base_size;
            }
            slgx_entry_t * ents =
                core_oom(calloc((size_t)n, sizeof(slgx_entry_t)));
            uint64_t off = base;
            int wrote = (ftruncate(fd, (off_t)base) == 0 &&
                         lseek(fd, (off_t)base, SEEK_SET) == (off_t)base);
            for (int i = 0; i < n && wrote; i++) {
                strncpy(ents[i].name, names[i], sizeof(ents[i].name) - 1);
                ents[i].off  = off;
                ents[i].size = sizes[i];
                wrote = (write(fd, blobs[i], sizes[i]) == (ssize_t)sizes[i]);
                off += sizes[i];
            }
            size_t eb = (size_t)n * sizeof(slgx_entry_t);
            slgx_tail_t tail = { base, (uint32_t)n, SLGX_MAGIC };
            if (wrote && write(fd, ents, eb) == (ssize_t)eb &&
                write(fd, &tail, sizeof(tail)) == (ssize_t)sizeof(tail)) {
                status = 1;
            }
            free(ents);
        }
    }
    if (fd >= 0) { close(fd); }
    return status;
}

struct llm * llm_load(const char * path) {
    struct llm * m = core_oom(calloc(1, sizeof(*m)));
    m->fd = -1;
    m->map = map_file(path, &m->fd, &m->map_size);
    if (m->map != NULL) {
        cur_t c = { (const uint8_t *)m->map };
        uint32_t magic = rd_u32(&c);
        uint32_t version = rd_u32(&c);
        uint64_t tensor_count = rd_u64(&c);
        uint64_t kv_count = rd_u64(&c);
        assert(magic == GGUF_MAGIC && version == 3u);
        parse_kv(m, &c, kv_count);
        m->owned = core_oom(malloc(tensor_count * sizeof(float *)));
        parse_tensors(m, &c, tensor_count);
        weights_load(m);
        parse_embedded(m);
    }
    if (m->map == NULL || m->tok_emb == NULL || m->layers == NULL ||
        m->layers[m->n_layer - 1].out_norm_b == NULL) {
        llm_free(m);
        m = NULL;
    }
    return m;
}

void llm_free(struct llm * m) {
    if (m != NULL) {
        if (m->owned != NULL) {
            for (int i = 0; i < m->owned_count; i++) { free(m->owned[i]); }
            free(m->owned);
        }
        free(m->layers);
        free(m->tensors);
        map_free(&m->vocab);
        if (m->map != NULL) { munmap(m->map, m->map_size); }
        if (m->fd >= 0) { close(m->fd); }
        free(m);
    }
}

// # Standalone test main

#ifdef LLM_MAIN
static void print_embedding(struct llm * m, const char * text) {
    float * emb = core_oom(malloc((size_t)llm_dim(m) * sizeof(float)));
    llm_embed(m, text, emb);
    for (int i = 0; i < llm_dim(m); i++) {
        printf("%s%.7f", (i > 0) ? " " : "", emb[i]);
    }
    printf("\n");
    free(emb);
}

/* usage: llm model.gguf "text"      -> one embedding
 *        llm model.gguf --stdin     -> one embedding per input line */
int main(int argc, char ** argv) {
    int status = 1;
    if (argc >= 3) {
        struct llm * m = llm_load(argv[1]);
        if (m != NULL) {
            if (strcmp(argv[2], "--stdin") == 0) {
                struct chars line = {0};
                int ch = 0;
                while (ch != EOF) {
                    line.count = 0;
                    while ((ch = getchar()) != EOF && ch != '\n') {
                        char b = (char)ch;
                        chars_put(&line, &b, 1);
                    }
                    if (line.count > 0) { print_embedding(m, line.data); }
                }
                chars_free(&line);
            } else {
                print_embedding(m, argv[2]);
            }
            llm_free(m);
            status = 0;
        } else {
            fprintf(stderr, "failed to load %s\n", argv[1]);
        }
    } else {
        fprintf(stderr, "usage: %s model.gguf \"text\" | --stdin\n", argv[0]);
    }
    return status;
}
#endif
