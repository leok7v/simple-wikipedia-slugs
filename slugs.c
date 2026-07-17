/* Simple English Wikipedia semantic slug engine.
 *
 * A compact fingerprint-per-article index over the whole Simple English
 * Wikipedia. Each article becomes one 256-bit key (sign-quantized PCA of an
 * all-MiniLM-L6-v2 embedding, computed by llm.c). A query is encoded the same
 * way and Hamming-searched against every key. The top matches
 * are printed as (pageid, title, url) so the caller can fetch the real article
 * text online.
 *
 * Modes:
 *   --index   : read "pageid<TAB>title<TAB>text" lines from stdin, embed each
 *               text via MiniLM, fit PCA 384->256, sign-quantize to 32 bytes,
 *               write keys.bin + slugs.tbl + pca.bin.
 *   --stats <q...> : encode a query and print its Hamming-distance histogram
 *               over the whole index (a confidence probe for the README).
 *   <query...>: encode, project, sign-pack, Hamming-search keys.bin, print
 *               the top-K articles.
 *
 * On-disk formats:
 *   pca.bin   : pca_params_t, mean[384] then components[256*384] row-major.
 *   keys.bin  : flat array of n 32-byte keys (sign-quantized PCA-256).
 *   slugs.tbl : [u64 count][u32 offsets[count+1]][u8 strings[*]]
 *               record i is strings[offsets[i]..offsets[i+1]), holding
 *               "pageid<TAB>title" (lossless UTF-8), aligned with keys.
 *
 * Style: snake_case, mandatory braces, single status flag, no early exits.
 */

#define _GNU_SOURCE   /* getline (glibc hides POSIX extensions under -std=c11) */

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "core.h"
#include "llm.h"

#define EMBED_DIM            384
#define PCA_DIM             256
#define KEY_BYTES            32           /* PCA_DIM / 8 */
#define DEFAULT_TOP_K         5
#define LOW_CONFIDENCE_DIST 100           /* ~ orthogonal for 256-bit keys */
#define PCA_SAMPLE_MAX   100000   /* cap PCA fit sample for speed/memory */
#define MODEL_PATH "minilm.gguf"
#define KEYS_PATH "keys.bin"
#define TBL_PATH  "slugs.tbl"
#define PCA_PATH  "pca.bin"

define_array(uint32_t, u32s);

typedef struct {
    float mean[EMBED_DIM];
    float components[PCA_DIM * EMBED_DIM]; /* row-major */
} pca_params_t;

static int g_verbosity = 1;

#define VLOG(level, ...) do { \
    if (g_verbosity >= (level)) { fprintf(stderr, __VA_ARGS__); } \
} while (0)

static const char * model_path(void) {
    const char *p = getenv("SLUGS_MODEL");
    return (p != NULL) ? p : MODEL_PATH;
}

// # Slug table (mmap reader)

typedef struct {
    int             fd;
    uint8_t        *map;
    size_t          map_size;
    uint64_t        count;
    const uint32_t *offsets;   /* offsets[count+1] */
    const uint8_t  *strings;
} slugs_tbl_t;

static int slugs_tbl_open(slugs_tbl_t *s, const char *path) {
    int status = 0;
    memset(s, 0, sizeof(*s));
    s->fd = open(path, O_RDONLY);
    if (s->fd >= 0) {
        struct stat st;
        if (fstat(s->fd, &st) == 0 && st.st_size >= 8) {
            s->map_size = (size_t)st.st_size;
            s->map = mmap(NULL, s->map_size, PROT_READ, MAP_PRIVATE, s->fd, 0);
            if (s->map != MAP_FAILED) {
                memcpy(&s->count, s->map, 8);
                size_t off_bytes = (size_t)(s->count + 1) * 4;
                if (8 + off_bytes <= s->map_size) {
                    s->offsets = (const uint32_t *)(s->map + 8);
                    s->strings = s->map + 8 + off_bytes;
                    status = 1;
                }
            }
        }
        if (status == 0) { close(s->fd); s->fd = -1; }
    }
    return status;
}

static void slugs_tbl_close(slugs_tbl_t *s) {
    if (s->map != NULL && s->map != MAP_FAILED) {
        munmap(s->map, s->map_size);
    }
    if (s->fd >= 0) { close(s->fd); }
    memset(s, 0, sizeof(*s));
}

static const uint8_t *slugs_tbl_get(const slugs_tbl_t *s, uint64_t i,
                                    size_t *out_len) {
    *out_len = (size_t)(s->offsets[i + 1] - s->offsets[i]);
    return s->strings + s->offsets[i];
}

/* Wrap an in-memory slug table (e.g. embedded in the model). Owns nothing;
 * slugs_tbl_close is a no-op on it (fd = -1, map = NULL). */
static int slugs_tbl_from_mem(slugs_tbl_t *s, const uint8_t *data,
                              size_t size) {
    int status = 0;
    memset(s, 0, sizeof(*s));
    s->fd = -1;
    if (size >= 8) {
        memcpy(&s->count, data, 8);
        size_t off_bytes = (size_t)(s->count + 1) * 4;
        if (8 + off_bytes <= size) {
            s->offsets = (const uint32_t *)(data + 8);
            s->strings = data + 8 + off_bytes;
            status = 1;
        }
    }
    return status;
}

// # Slug table writer (core containers)

typedef struct {
    struct chars strings;
    struct u32s  offsets;   /* offsets[0]=0, one per record end; count = n+1 */
} slug_writer_t;

static void slug_writer_init(slug_writer_t *w) {
    w->strings = (struct chars){0};
    w->offsets = (struct u32s){0};
    u32s_put(&w->offsets, 0);
}

static void slug_writer_append(slug_writer_t *w, const char *slug,
                               size_t len) {
    chars_put(&w->strings, slug, len);
    u32s_put(&w->offsets, (uint32_t)w->strings.count);
}

static int slug_writer_flush(slug_writer_t *w, const char *path) {
    int status = 0;
    FILE *fp = fopen(path, "wb");
    if (fp != NULL) {
        size_t no = w->offsets.count, ns = w->strings.count;
        uint64_t cnt = (uint64_t)(no - 1);
        if (fwrite(&cnt, 8, 1, fp) == 1 &&
            fwrite(w->offsets.data, sizeof(uint32_t), no, fp) == no &&
            fwrite(w->strings.data, 1, ns, fp) == ns)
        {
            status = 1;
        }
        fclose(fp);
    }
    return status;
}

static void slug_writer_free(slug_writer_t *w) {
    chars_free(&w->strings);
    u32s_free(&w->offsets);
}

// # PCA

/* Auto-vectorized dot product. */
static float sdot(const float * restrict a, const float * restrict b, int n) {
    float s = 0.0f;
#pragma clang loop vectorize(enable)
    for (int i = 0; i < n; i++) { s += a[i] * b[i]; }
    return s;
}

/* Symmetric eigendecomposition by cyclic Jacobi rotations. On entry a is an
 * n x n row-major symmetric matrix (destroyed); on return a's diagonal holds
 * the eigenvalues and v (n x n row-major) holds the eigenvectors as columns
 * (eigenvector k is v[0..n-1][k]). n is 384 and this runs once per index
 * build, so a handful of O(n^3) sweeps is cheap. */
static void jacobi_eigen(float *a, int n, float *v) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) { v[i * n + j] = (i == j) ? 1.0f : 0.0f; }
    }
    int sweep = 0;
    double off = 1.0;
    while (sweep < 80 && off > 1e-12) {
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                float apq = a[p * n + q];
                if (fabsf(apq) > 1e-20f) {
                    float theta = (a[q * n + q] - a[p * n + p]) / (2.0f * apq);
                    float sgn = (theta >= 0.0f) ? 1.0f : -1.0f;
                    float h = fabsf(theta) + sqrtf(theta * theta + 1.0f);
                    float t = sgn / h;
                    float c = 1.0f / sqrtf(t * t + 1.0f);
                    float s = t * c;
                    for (int k = 0; k < n; k++) {
                        float akp = a[k * n + p], akq = a[k * n + q];
                        a[k * n + p] = c * akp - s * akq;
                        a[k * n + q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++) {
                        float apk = a[p * n + k], aqk = a[q * n + k];
                        a[p * n + k] = c * apk - s * aqk;
                        a[q * n + k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++) {
                        float vkp = v[k * n + p], vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        double on_e = 0.0, off_e = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double e = (double)a[i * n + j] * a[i * n + j];
                if (i == j) { on_e += e; } else { off_e += e; }
            }
        }
        off = (on_e > 0.0) ? off_e / on_e : 0.0;
        sweep += 1;
    }
}

/* Fit PCA on n EMBED_DIM vectors; keep the top PCA_DIM components. */
static int fit_pca(const float *X, int n, pca_params_t *params) {
    for (int j = 0; j < EMBED_DIM; j++) { params->mean[j] = 0.0f; }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            params->mean[j] += X[i * EMBED_DIM + j];
        }
    }
    float inv_n = 1.0f / (float)n;
    for (int j = 0; j < EMBED_DIM; j++) { params->mean[j] *= inv_n; }
    float *Xc = core_oom(malloc((size_t)n * EMBED_DIM * sizeof(float)));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            Xc[i * EMBED_DIM + j] = X[i * EMBED_DIM + j] - params->mean[j];
        }
    }
    /* cov = Xc^T Xc / (n-1), accumulated by outer products; the inner j loop
     * is contiguous and vectorizes. */
    float *cov =
        core_oom(calloc((size_t)EMBED_DIM * EMBED_DIM, sizeof(float)));
    for (int k = 0; k < n; k++) {
        const float *x = Xc + (size_t)k * EMBED_DIM;
        for (int i = 0; i < EMBED_DIM; i++) {
            float a = x[i];
            float *crow = cov + (size_t)i * EMBED_DIM;
            for (int j = 0; j < EMBED_DIM; j++) { crow[j] += a * x[j]; }
        }
    }
    free(Xc);
    float inv_d = 1.0f / (float)((n > 1) ? (n - 1) : 1);
    size_t ncov = (size_t)EMBED_DIM * EMBED_DIM;
    for (size_t i = 0; i < ncov; i++) { cov[i] *= inv_d; }
    float *V = core_oom(malloc((size_t)EMBED_DIM * EMBED_DIM * sizeof(float)));
    jacobi_eigen(cov, EMBED_DIM, V);
    float evals[EMBED_DIM];
    int   order[EMBED_DIM];
    for (int i = 0; i < EMBED_DIM; i++) {
        evals[i] = cov[i * EMBED_DIM + i];
        order[i] = i;
    }
    /* selection sort: order[] indexes eigenvalues descending. */
    for (int i = 0; i < EMBED_DIM; i++) {
        int mx = i;
        for (int j = i + 1; j < EMBED_DIM; j++) {
            if (evals[order[j]] > evals[order[mx]]) { mx = j; }
        }
        int tmp = order[i]; order[i] = order[mx]; order[mx] = tmp;
    }
    double sum_all = 0.0, sum_top = 0.0;
    for (int i = 0; i < EMBED_DIM; i++) {
        sum_all += (evals[i] > 0 ? evals[i] : 0);
    }
    for (int i = 0; i < PCA_DIM; i++) {
        sum_top += (evals[order[i]] > 0 ? evals[order[i]] : 0);
    }
    VLOG(2, "[pca] top-%d eigenvalues %.4g..%.4g, capture %.1f%% variance\n",
         PCA_DIM, evals[order[PCA_DIM - 1]], evals[order[0]],
         sum_all > 0 ? sum_top / sum_all * 100 : 0.0);
    for (int i = 0; i < PCA_DIM; i++) {
        int c = order[i];
        for (int j = 0; j < EMBED_DIM; j++) {
            params->components[i * EMBED_DIM + j] = V[j * EMBED_DIM + c];
        }
    }
    free(V);
    free(cov);
    return 1;
}

static void pca_apply(const pca_params_t *p, const float *emb, float *out_y) {
    float centered[EMBED_DIM];
    for (int j = 0; j < EMBED_DIM; j++) { centered[j] = emb[j] - p->mean[j]; }
    for (int i = 0; i < PCA_DIM; i++) {
        const float *row = p->components + (size_t)i * EMBED_DIM;
        out_y[i] = sdot(row, centered, EMBED_DIM);
    }
}

static void pack_signs(const float *y, uint8_t *key) {
    for (int b = 0; b < KEY_BYTES; b++) { key[b] = 0; }
    for (int b = 0; b < PCA_DIM; b++) {
        if (y[b] > 0.0f) { key[b / 8] |= (uint8_t)(1u << (7 - (b % 8))); }
    }
}

static inline int hamming32(const uint8_t *a, const uint8_t *b) {
    uint64_t x0, x1, x2, x3;
    memcpy(&x0, a,      8); memcpy(&x1, a + 8,  8);
    memcpy(&x2, a + 16, 8); memcpy(&x3, a + 24, 8);
    uint64_t y0, y1, y2, y3;
    memcpy(&y0, b,      8); memcpy(&y1, b + 8,  8);
    memcpy(&y2, b + 16, 8); memcpy(&y3, b + 24, 8);
    return __builtin_popcountll(x0 ^ y0) + __builtin_popcountll(x1 ^ y1)
         + __builtin_popcountll(x2 ^ y2) + __builtin_popcountll(x3 ^ y3);
}

// # --index

/* Fit PCA (on a capped random sample of vecs), sign-quantize all n rows, and
 * write keys.bin + slugs.tbl + pca.bin. */
static int write_index(const float *vecs, size_t n, struct text *recs) {
    int status = 0;
    int fit_n = (n < PCA_SAMPLE_MAX) ? (int)n : PCA_SAMPLE_MAX;
    const float *sample = vecs;
    float *owned = NULL;
    if ((size_t)fit_n < n) {
        owned = core_oom(malloc((size_t)fit_n * EMBED_DIM * sizeof(float)));
        uint64_t s = 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < fit_n; i++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            size_t idx = (size_t)((s >> 33) % n);
            memcpy(&owned[i * EMBED_DIM], &vecs[idx * EMBED_DIM],
                   EMBED_DIM * sizeof(float));
        }
        sample = owned;
        VLOG(1, "fitting PCA on %d sampled vectors...\n", fit_n);
    } else {
        VLOG(1, "fitting PCA on all %d vectors...\n", fit_n);
    }
    pca_params_t *params = core_oom(malloc(sizeof(pca_params_t)));
    int fitted = fit_pca(sample, fit_n, params);
    free(owned);
    if (fitted) {
        FILE *pf = fopen(PCA_PATH, "wb");
        if (pf != NULL) {
            fwrite(params, sizeof(pca_params_t), 1, pf);
            fclose(pf);
        }
        FILE *kf = fopen(KEYS_PATH, "wb");
        slug_writer_t sw;
        slug_writer_init(&sw);
        if (kf != NULL) {
            for (size_t i = 0; i < n; i++) {
                float y[PCA_DIM];
                pca_apply(params, &vecs[i * EMBED_DIM], y);
                uint8_t key[KEY_BYTES];
                pack_signs(y, key);
                fwrite(key, 1, KEY_BYTES, kf);
                slug_writer_append(&sw, recs->data[i], strlen(recs->data[i]));
            }
            fclose(kf);
            if (slug_writer_flush(&sw, TBL_PATH)) {
                VLOG(1, "wrote %s + %s + %s (%zu articles, %d-bit keys)\n",
                     KEYS_PATH, TBL_PATH, PCA_PATH, n, PCA_DIM);
                status = 1;
            }
        }
        slug_writer_free(&sw);
    }
    free(params);
    return status;
}

static int index_mode(void) {
    int status = 0;
    struct llm *m = llm_load(model_path());
    if (m != NULL) {
        assert(llm_dim(m) == EMBED_DIM);
        struct text recs  = {0};   /* "pageid\ttitle" per row */
        struct text texts = {0};   /* article text per row */
        char   *line = NULL;
        size_t  lcap = 0;
        ssize_t llen;
        VLOG(1, "reading 'pageid<TAB>title<TAB>text' from stdin...\n");
        while ((llen = getline(&line, &lcap, stdin)) != -1) {
            if (llen > 0 && line[llen - 1] == '\n') { line[--llen] = '\0'; }
            char *tab1 = strchr(line, '\t');
            char *tab2 = (tab1 != NULL) ? strchr(tab1 + 1, '\t') : NULL;
            if (tab1 != NULL && tab2 != NULL) {
                *tab2 = '\0';   /* now "pageid\ttitle"; text follows */
                text_puts(&recs, line);
                text_puts(&texts, tab2 + 1);
            }
        }
        free(line);
        int n = (int)recs.count;
        if (n >= 2) {
            float *vecs =
                core_oom(malloc((size_t)n * EMBED_DIM * sizeof(float)));
            VLOG(1, "embedding %d articles with a thread pool...\n", n);
            llm_embed_batch(m, (const char *const *)texts.data, n, EMBED_DIM,
                            vecs, 0);
            VLOG(1, "embedded %d articles\n", n);
            status = write_index(vecs, (size_t)n, &recs);
            free(vecs);
        } else {
            VLOG(0, "need >=2 rows to fit PCA; got %d\n", n);
        }
        text_free(&recs);
        text_free(&texts);
        llm_free(m);
    } else {
        VLOG(0, "could not load model %s\n", model_path());
    }
    return status;
}

// # Query helpers

static void gather_query(int argc, char **argv, int start,
                         struct chars *query) {
    for (int i = start; i < argc; i++) {
        if (i > start) { chars_puts(query, " "); }
        chars_puts(query, argv[i]);
    }
}

/* Encode a query to a 256-bit key. */
static int encode_query_key(struct llm *m, const char *query,
                            const pca_params_t *params, uint8_t *q_key) {
    float emb[EMBED_DIM];
    int T = llm_embed(m, query, emb);
    if (T > 0) {
        float y[PCA_DIM];
        pca_apply(params, emb, y);
        pack_signs(y, q_key);
    }
    return T > 0;
}

static int load_pca(pca_params_t *params) {
    int ok = 0;
    FILE *fp = fopen(PCA_PATH, "rb");
    if (fp != NULL) {
        ok = (fread(params, sizeof(*params), 1, fp) == 1);
        fclose(fp);
    }
    return ok;
}

/* Read a whole file into a fresh buffer. Returns NULL if absent/empty. */
static void *read_file(const char *path, size_t *size) {
    void *buf = NULL;
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            buf = core_oom(malloc((size_t)st.st_size));
            if (read(fd, buf, (size_t)st.st_size) == (ssize_t)st.st_size) {
                *size = (size_t)st.st_size;
            } else {
                free(buf);
                buf = NULL;
            }
        }
        close(fd);
    }
    return buf;
}

// # --pack : embed the index into the model

static int pack_mode(const char *gguf) {
    int status = 0;
    size_t ps = 0, ks = 0, ts = 0;
    void *pca  = read_file(PCA_PATH, &ps);
    void *keys = read_file(KEYS_PATH, &ks);
    void *tbl  = read_file(TBL_PATH, &ts);
    if (pca != NULL && keys != NULL && tbl != NULL) {
        const char *names[3] = { "pca", "keys", "slugs" };
        const void *blobs[3] = { pca, keys, tbl };
        size_t      sizes[3] = { ps, ks, ts };
        if (llm_pack(gguf, names, blobs, sizes, 3)) {
            VLOG(1, "packed pca+keys+slugs into %s (self-contained)\n", gguf);
            status = 1;
        } else {
            VLOG(0, "could not write trailer to %s\n", gguf);
        }
    } else {
        VLOG(0, "missing %s/%s/%s; run --index first\n",
             PCA_PATH, KEYS_PATH, TBL_PATH);
    }
    free(pca);
    free(keys);
    free(tbl);
    return status;
}

// # Index source: embedded in the model, or the on-disk files

typedef struct {
    pca_params_t   params;
    const uint8_t *keys;
    size_t         n_keys;
    slugs_tbl_t    tbl;
    const uint8_t *kmap;    /* file-mmap of keys.bin, NULL when embedded */
    size_t         kmapsz;
    int            kfd;     /* keys.bin fd, -1 when embedded */
} index_t;

/* Prefer the model's embedded index (self-contained gguf); else fall back to
 * the on-disk pca.bin / keys.bin / slugs.tbl. Returns 1 on success. */
static int resolve_index(struct llm *m, index_t *idx) {
    int status = 0;
    memset(idx, 0, sizeof(*idx));
    idx->kfd = -1;
    size_t ps = 0, ks = 0, ts = 0;
    const void *pca  = llm_extra(m, "pca", &ps);
    const void *keys = llm_extra(m, "keys", &ks);
    const void *tbl  = llm_extra(m, "slugs", &ts);
    if (pca != NULL && keys != NULL && tbl != NULL &&
        ps == sizeof(pca_params_t) && ks >= KEY_BYTES) {
        memcpy(&idx->params, pca, sizeof(pca_params_t));
        idx->keys   = keys;
        idx->n_keys = ks / KEY_BYTES;
        status = slugs_tbl_from_mem(&idx->tbl, tbl, ts);
    } else if (load_pca(&idx->params)) {
        idx->kfd = open(KEYS_PATH, O_RDONLY);
        struct stat kst;
        if (idx->kfd >= 0 && fstat(idx->kfd, &kst) == 0 &&
            kst.st_size >= KEY_BYTES) {
            idx->kmap = mmap(NULL, kst.st_size, PROT_READ, MAP_PRIVATE,
                             idx->kfd, 0);
            if (idx->kmap != MAP_FAILED &&
                slugs_tbl_open(&idx->tbl, TBL_PATH)) {
                idx->keys   = idx->kmap;
                idx->kmapsz = (size_t)kst.st_size;
                idx->n_keys = (size_t)kst.st_size / KEY_BYTES;
                status = 1;
            }
        }
    }
    return status;
}

static void free_index(index_t *idx) {
    slugs_tbl_close(&idx->tbl);
    if (idx->kmap != NULL && idx->kmap != MAP_FAILED) {
        munmap((void *)idx->kmap, idx->kmapsz);
    }
    if (idx->kfd >= 0) { close(idx->kfd); }
}

// # <query> : top-K

typedef struct { int dist; size_t idx; } hit_t;

static void print_hit(const slugs_tbl_t *st, int rank, const hit_t *h) {
    size_t slen;
    const uint8_t *rec = slugs_tbl_get(st, h->idx, &slen);
    size_t tabpos = 0;
    while (tabpos < slen && rec[tabpos] != '\t') { tabpos++; }
    int idlen    = (int)tabpos;
    int titlepos = (int)((tabpos < slen) ? tabpos + 1 : slen);
    int titlelen = (int)slen - titlepos;
    const char *flag = (h->dist > LOW_CONFIDENCE_DIST) ? "  [low]" : "";
    printf("  %d. d=%-3d  id=%-8.*s  %.*s%s\n",
           rank, h->dist, idlen, (const char *)rec,
           titlelen, (const char *)rec + titlepos, flag);
    printf("       https://simple.wikipedia.org/?curid=%.*s\n",
           idlen, (const char *)rec);
}

static int query_mode(int argc, char **argv, int top_k) {
    int status = 0;
    struct chars query = {0};
    gather_query(argc, argv, 1, &query);
    struct llm *m = (query.count > 0) ? llm_load(model_path()) : NULL;
    index_t idx;
    if (m != NULL && resolve_index(m, &idx)) {
        uint8_t q_key[KEY_BYTES];
        if (encode_query_key(m, query.data, &idx.params, q_key)) {
            hit_t *top = core_oom(malloc((size_t)top_k * sizeof(hit_t)));
            for (int i = 0; i < top_k; i++) {
                top[i].dist = PCA_DIM + 1;
                top[i].idx = 0;
            }
            for (size_t i = 0; i < idx.n_keys; i++) {
                int d = hamming32(q_key, idx.keys + i * KEY_BYTES);
                if (d < top[top_k - 1].dist) {
                    int p = top_k - 1;
                    while (p > 0 && top[p - 1].dist > d) {
                        top[p] = top[p - 1];
                        p--;
                    }
                    top[p].dist = d;
                    top[p].idx  = i;
                }
            }
            printf("query: %s\n", query.data);
            printf("top %d of %zu (256-bit hamming):\n", top_k, idx.n_keys);
            for (int r = 0; r < top_k; r++) {
                print_hit(&idx.tbl, r + 1, &top[r]);
            }
            free(top);
            status = 1;
        }
        free_index(&idx);
    } else {
        VLOG(0, "no index or model; build with --index then --pack\n");
    }
    if (m != NULL) { llm_free(m); }
    chars_free(&query);
    return status;
}

// # --stats <query> : hamming distribution

static void hbar(int width) {
    for (int i = 0; i < width; i++) { fputs("#", stdout); }
}

static int stats_mode(int argc, char **argv) {
    int status = 0;
    struct chars query = {0};
    gather_query(argc, argv, 2, &query);
    struct llm *m = (query.count > 0) ? llm_load(model_path()) : NULL;
    index_t idx;
    if (m != NULL && resolve_index(m, &idx)) {
        uint8_t q_key[KEY_BYTES];
        if (encode_query_key(m, query.data, &idx.params, q_key)) {
            long bins[33] = {0};   /* bin i = [i*8,(i+1)*8); 256/8 = 32 */
            int min_d = PCA_DIM + 1;
            for (size_t i = 0; i < idx.n_keys; i++) {
                int d = hamming32(q_key, idx.keys + i * KEY_BYTES);
                int bin = d / 8;
                if (bin > 32) { bin = 32; }
                bins[bin]++;
                if (d < min_d) { min_d = d; }
            }
            long maxbin = 0;
            for (int i = 0; i < 33; i++) {
                if (bins[i] > maxbin) { maxbin = bins[i]; }
            }
            printf("query: %s\n", query.data);
            printf("nearest distance: %d (of %d bits)\n\n", min_d, PCA_DIM);
            printf("distance distribution over %zu articles:\n", idx.n_keys);
            for (int i = 0; i < 33; i++) {
                int skip = (bins[i] == 0 && i * 8 > min_d + 40 && i < 24);
                if (!skip) {
                    int width = (maxbin > 0)
                        ? (int)(50.0 * bins[i] / (double)maxbin) : 0;
                    int lo = i * 8;
                    int hi = (i == 32) ? PCA_DIM : ((i + 1) * 8 - 1);
                    double pct = 100.0 * bins[i] / (double)idx.n_keys;
                    printf("  %3d..%3d  ", lo, hi);
                    hbar(width);
                    printf("  %8ld  (%5.2f%%)\n", bins[i], pct);
                }
            }
            status = 1;
        }
        free_index(&idx);
    } else {
        VLOG(0, "no index or model; build with --index then --pack\n");
    }
    if (m != NULL) { llm_free(m); }
    chars_free(&query);
    return status;
}

// # Main

int main(int argc, char **argv) {
    int status = 0;
    int top_k = DEFAULT_TOP_K;
    /* Strip our own flags out of argv before dispatching to a mode. */
    int kept = 1;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--verbosity:", 12) == 0) {
            int v = atoi(argv[i] + 12);
            g_verbosity = (v < 0) ? 0 : v;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            g_verbosity = 3;
        } else if (strncmp(argv[i], "--top:", 6) == 0) {
            int t = atoi(argv[i] + 6);
            top_k = (t < 1) ? 1 : t;
        } else {
            argv[kept++] = argv[i];
        }
    }
    argv[kept] = NULL;
    int eff_argc = kept;
    if (eff_argc < 2) {
        fprintf(stderr,
                "usage:\n"
                "  %s --index              index pageid<TAB>title<TAB>text\n"
                "  %s --pack <model.gguf>  embed the index into the model\n"
                "  %s --stats <query...>   hamming distance histogram\n"
                "  %s <query...>           search; print top-K articles\n"
                "flags: --top:N  --verbose  --verbosity:N\n",
                argv[0], argv[0], argv[0], argv[0]);
        status = 1;
    } else {
        int ok = 0;
        if (strcmp(argv[1], "--index") == 0) {
            ok = index_mode();
        } else if (strcmp(argv[1], "--pack") == 0) {
            ok = (eff_argc >= 3) ? pack_mode(argv[2]) : 0;
        } else if (strcmp(argv[1], "--stats") == 0) {
            ok = stats_mode(eff_argc, argv);
        } else {
            ok = query_mode(eff_argc, argv, top_k);
        }
        status = ok ? 0 : 1;
    }
    return status;
}
