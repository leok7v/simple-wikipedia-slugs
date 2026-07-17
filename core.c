// Fundamental collections -- implementation. See inc/core.h.
// Standalone C; depends only on the C standard library.

#include "core.h"

#include <stdio.h>
#include <string.h>

// ============================================================================
// allocation
// ============================================================================

// Treat OOM as fatal; abort() so we don't run atexit handlers on a
// half-built program state.
void * core_oom(void * p) {
    if (!p) { fprintf(stderr, "OOM"); abort(); }
    return p;
}

// ============================================================================
// chars
// ============================================================================

// Uses any_grow with sizeof(s->data[0]) == 1 (char). An earlier shape
// delegated through a (void*) cast which violates strict aliasing -- gcc
// -O2 then silently skips the write-back to s->data, leaving the buffer
// NULL. Keep the typed any_grow.
void chars_grow(struct chars * s, size_t need) {
    any_grow(s, need);
}

void chars_put(struct chars * s, const char * d, size_t count) {
    chars_grow(s, s->count + count + 1);
    if (s->data) {
        memcpy(s->data + s->count, d, count);
        s->count += count;
        s->data[s->count] = '\0';
    }
}

void chars_free(struct chars * s) {
    free(s->data);
    s->data = NULL;
    s->count = 0;
    s->capacity = 0;
}

void chars_puts(struct chars * s, const char * a) {
    chars_put(s, a, strlen(a));
}

void chars_vprintf(struct chars * s, const char * format, va_list vl) {
    va_list cp;
    va_copy(cp, vl);
    int r = vsnprintf(NULL, 0, format, cp);
    va_end(cp);
    if (r > 0) {
        size_t n = (size_t)r;
        chars_grow(s, s->count + n + 1);
        vsnprintf(s->data + s->count, n + 1, format, vl);
        s->count += n;
    }
}

void chars_printf(struct chars * s, const char * format, ...) {
    va_list vl;
    va_start(vl, format);
    chars_vprintf(s, format, vl);
    va_end(vl);
}

// ============================================================================
// text
// ============================================================================

void text_grow(struct text * t, size_t need) {
    any_grow(t, need);
}

void text_put(struct text * t, const char * s, size_t n) {
    text_grow(t, t->count + 1);
    char * copy = core_oom(malloc(n + 1));
    memcpy(copy, s, n);
    copy[n] = '\0';
    t->data[t->count++] = copy;
}

void text_puts(struct text * t, const char * s) {
    text_put(t, s, strlen(s));
}

void text_free(struct text * t) {
    for (size_t i = 0; i < t->count; i++) { free(t->data[i]); }
    free(t->data);
    t->data = NULL;
    t->count = 0;
    t->capacity = 0;
}

// ============================================================================
// maps
// ============================================================================

static inline uint64_t map_hash_i(int64_t k) {
    uint64_t x = (uint64_t)k;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline uint64_t map_hash_s(const struct chars * k) {
    uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a offset
    for (size_t i = 0; i < k->count; i++) {
        h ^= (uint8_t)k->data[i];
        h *= 0x100000001b3ULL; // FNV prime
    }
    return h;
}

static inline uint64_t map_hash(const struct map * m, const void * k) {
    uint64_t h = 0;
    if (m->key_kind == MAP_KEY_INT) {
        h = map_hash_i(*(const int64_t *)k);
    } else {
        h = map_hash_s((const struct chars *)k);
    }
    return h;
}

static inline bool map_eq(const struct map * m, const void * a,
                          const void * b) {
    bool r = false;
    if (m->key_kind == MAP_KEY_INT) {
        r = *(const int64_t *)a == *(const int64_t *)b;
    } else {
        const struct chars * x = a;
        const struct chars * y = b;
        r = x->count == y->count
            && memcmp(x->data, y->data, x->count) == 0;
    }
    return r;
}

static inline void * map_k(struct map * m, size_t i) {
    return (char *)m->keys + i * m->key_size;
}

static inline void * map_v(struct map * m, size_t i) {
    return (char *)m->values + i * m->value_size;
}

static inline void map_key_copy(struct map * m, void * dst,
                                const void * src) {
    if (m->key_kind == MAP_KEY_INT) {
        memcpy(dst, src, sizeof(int64_t));
    } else {
        const struct chars * s = src;
        struct chars * d = dst;
        *d = (struct chars){0};
        chars_put(d, s->data, s->count);
    }
}

static inline void map_key_free(struct map * m, void * k) {
    if (m->key_kind == MAP_KEY_CHARS) {
        chars_free(k);
    }
}

void map_init(struct map * m, enum map_key kk,
              size_t ks, size_t vs, void (*vf)(void *)) {
    m->states     = NULL;
    m->keys       = NULL;
    m->values     = NULL;
    m->count      = 0;
    m->capacity   = 0;
    m->key_size   = ks;
    m->value_size = vs;
    m->key_kind   = kk;
    m->value_free = vf;
}

// Lazy initialization for the typed accessors: a zero-initialized map has
// value_size == 0, which is the unambiguous "not yet configured" marker
// (a real value type is always > 0 bytes). Sets key kind/size and value
// size on first use; leaves value_free as the caller left it (NULL for a
// {0} map). A no-op once configured, so it composes with map_init.
static void map_lazy_init(struct map * m, enum map_key kk,
                          size_t ks, size_t vs) {
    if (m->value_size == 0) {
        m->key_kind   = kk;
        m->key_size   = ks;
        m->value_size = vs;
    }
}

static void map_grow(struct map * m, size_t new_cap) {
    uint8_t * ns = core_oom(calloc(new_cap, 1));
    void *    nk = core_oom(malloc(new_cap * m->key_size));
    void *    nv = core_oom(malloc(new_cap * m->value_size));
    size_t mask = new_cap - 1;
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->states[i] == MAP_LIVE) {
            void * ok = map_k(m, i);
            size_t j = (size_t)map_hash(m, ok) & mask;
            while (ns[j] != MAP_EMPTY) { j = (j + 1) & mask; }
            ns[j] = MAP_LIVE;
            memcpy((char *)nk + j * m->key_size, ok, m->key_size);
            memcpy((char *)nv + j * m->value_size,
                   map_v(m, i), m->value_size);
        }
    }
    free(m->states);
    free(m->keys);
    free(m->values);
    m->states   = ns;
    m->keys     = nk;
    m->values   = nv;
    m->capacity = new_cap;
}

void * map_put(struct map * m, const void * k, const void * v) {
    assert(m->value_size > 0 &&
           "map_put: call map_init() or a typed accessor (map_puti/map_puts)");
    void * r = NULL;
    if (m->capacity == 0) {
        map_grow(m, 16);
    } else if ((m->count + 1) * 4 > m->capacity * 3) {
        map_grow(m, m->capacity * 2);
    }
    size_t mask = m->capacity - 1;
    size_t j    = (size_t)map_hash(m, k) & mask;
    size_t tomb = (size_t)-1;
    while (m->states[j] != MAP_EMPTY
           && !(m->states[j] == MAP_LIVE && map_eq(m, map_k(m, j), k))) {
        if (m->states[j] == MAP_TOMBSTONE && tomb == (size_t)-1) {
            tomb = j;
        }
        j = (j + 1) & mask;
    }
    if (m->states[j] == MAP_LIVE) {
        void * vs = map_v(m, j);
        if (m->value_free) { m->value_free(vs); }
        memcpy(vs, v, m->value_size);
        r = vs;
    } else {
        if (tomb != (size_t)-1) { j = tomb; }
        m->states[j] = MAP_LIVE;
        map_key_copy(m, map_k(m, j), k);
        memcpy(map_v(m, j), v, m->value_size);
        m->count++;
        r = map_v(m, j);
    }
    return r;
}

void * map_get(struct map * m, const void * k) {
    void * r = NULL;
    if (m->capacity > 0) {
        size_t mask = m->capacity - 1;
        size_t j    = (size_t)map_hash(m, k) & mask;
        while (m->states[j] != MAP_EMPTY
               && !(m->states[j] == MAP_LIVE && map_eq(m, map_k(m, j), k))) {
            j = (j + 1) & mask;
        }
        if (m->states[j] == MAP_LIVE) {
            r = map_v(m, j);
        }
    }
    return r;
}

void map_remove(struct map * m, const void * k) {
    if (m->capacity > 0) {
        size_t mask = m->capacity - 1;
        size_t j    = (size_t)map_hash(m, k) & mask;
        while (m->states[j] != MAP_EMPTY
               && !(m->states[j] == MAP_LIVE && map_eq(m, map_k(m, j), k))) {
            j = (j + 1) & mask;
        }
        if (m->states[j] == MAP_LIVE) {
            if (m->value_free) { m->value_free(map_v(m, j)); }
            map_key_free(m, map_k(m, j));
            m->states[j] = MAP_TOMBSTONE;
            m->count--;
        }
    }
}

void map_free(struct map * m) {
    if (m->capacity > 0) {
        for (size_t i = 0; i < m->capacity; i++) {
            if (m->states[i] == MAP_LIVE) {
                if (m->value_free) { m->value_free(map_v(m, i)); }
                map_key_free(m, map_k(m, i));
            }
        }
        free(m->states);
        free(m->keys);
        free(m->values);
    }
    m->states   = NULL;
    m->keys     = NULL;
    m->values   = NULL;
    m->count    = 0;
    m->capacity = 0;
}

void map_for_each(struct map * m,
                  void (*fn)(const void * k, void * v, void * ctx),
                  void * ctx) {
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->states[i] == MAP_LIVE) {
            fn(map_k(m, i), map_v(m, i), ctx);
        }
    }
}

void * map_put_int(struct map * m, int64_t k, const void * v, size_t vs) {
    map_lazy_init(m, MAP_KEY_INT, sizeof(int64_t), vs);
    return map_put(m, &k, v);
}

void * map_get_int(struct map * m, int64_t k) {
    return map_get(m, &k);
}

void map_remove_int(struct map * m, int64_t k) {
    map_remove(m, &k);
}

void * map_put_str(struct map * m, const char * k, const void * v, size_t vs) {
    map_lazy_init(m, MAP_KEY_CHARS, sizeof(struct chars), vs);
    struct chars tmp = {0};
    tmp.data  = (char *)(uintptr_t)k;
    tmp.count = strlen(k);
    return map_put(m, &tmp, v);
}

void * map_get_str(struct map * m, const char * k) {
    struct chars tmp = {0};
    tmp.data  = (char *)(uintptr_t)k;
    tmp.count = strlen(k);
    return map_get(m, &tmp);
}

void map_remove_str(struct map * m, const char * k) {
    struct chars tmp = {0};
    tmp.data  = (char *)(uintptr_t)k;
    tmp.count = strlen(k);
    map_remove(m, &tmp);
}

void chars_free_v(void * s) {
    chars_free((struct chars *)s);
}

// ============================================================================
// rng
// ============================================================================

void rng_seed(struct rng * r, uint64_t s) {
    r->state = s ? s : 0xdeadbeefcafef00dULL;
}

uint64_t rng_next(struct rng * r) {
    uint64_t x = r->state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    r->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

float rng_uniform(struct rng * r) {
    return (float)((rng_next(r) >> 40) & 0xFFFFFF) / (float)(1u << 24);
}
