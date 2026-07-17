// Fundamental collections: growable arrays, a NUL-terminated byte buffer
// (chars), an owned-string vector (text), an open-addressed hash map, and
// a small PRNG.
//
// Standalone C: depends only on the C standard library and compiles
// cleanly on Windows (MSVC /std:c17), macOS (clang) and Linux (gcc/clang).
// No dependency on the rt_* runtime.

#ifndef CORE_H
#define CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#define CORE_PRINTF_ATTR(fmt, va) __attribute__((format(printf, fmt, va)))
#else
#define CORE_PRINTF_ATTR(fmt, va)
#endif

// Treat allocation failure as fatal (aborts). Returns `p` unchanged when
// non-NULL so it can wrap malloc/realloc inline. Implemented in core.c.
void * core_oom(void * p);

// ============================================================================
// arrays -- generic {data, count, capacity} growth
// ============================================================================

// Generic grow for any {data, count, capacity}-shaped struct. Element size
// comes from sizeof(a->data[0]) -- that walks through the typed pointer so
// we never need a (T) parameter or a (void*) cast, which would re-introduce
// the strict-aliasing pitfall gcc -O2 exploits.
//
// Pre-conditions (asserted in debug, compiled out under -DNDEBUG):
//   - if data == NULL : count == 0 && capacity == 0  (cleanly empty)
//   - if data != NULL : capacity > 0 && count <= capacity
// Catches the easy misuse -- a stack container without an explicit {0}
// initializer, or a malloc without memset -- which would otherwise read
// garbage count/capacity.
//
// Post-condition: data != NULL && capacity >= need (core_oom aborts on
// malloc/realloc failure, so we never return with data == NULL).
#define any_grow(a, need) do { \
    assert((a)->data \
           ? ((a)->capacity > 0 && (a)->count <= (a)->capacity) \
           : ((a)->capacity == 0 && (a)->count == 0)); \
    if (!(a)->data) { \
        (a)->capacity = (need); \
        (a)->data = core_oom(malloc((need) * sizeof((a)->data[0]))); \
    } else if ((need) > (a)->capacity) { \
        (a)->capacity = (need) * 2; \
        (a)->data = core_oom(realloc((a)->data, \
                    (a)->capacity * sizeof((a)->data[0]))); \
    } \
    assert((a)->data != NULL && (a)->capacity >= (need)); \
} while (0)

#define define_array(T, name) \
struct name { T * data; size_t count; size_t capacity; }; \
static inline void name##_grow(struct name * a, size_t need) { \
    any_grow(a, need); \
} \
static inline void name##_put(struct name * a, T v) { \
    name##_grow(a, a->count + 1); \
    a->data[a->count++] = v; \
} \
static inline void name##_free(struct name * a) { \
    free(a->data); \
    a->data = NULL; \
    a->count = 0; \
    a->capacity = 0; \
} \
struct name##_swallow_semicolon

// ============================================================================
// chars -- grown-on-demand, always NUL-terminated byte buffer
// ============================================================================

struct chars {
    char * data;
    size_t count;
    size_t capacity;
};

void chars_grow(struct chars * s, size_t need);
void chars_put(struct chars * s, const char * d, size_t count);
void chars_free(struct chars * s);
void chars_puts(struct chars * s, const char * a);
void chars_vprintf(struct chars * s, const char * format, va_list vl);
void chars_printf(struct chars * s, const char * format, ...)
    CORE_PRINTF_ATTR(2, 3);

// ============================================================================
// text -- append-only vector of owned C strings
// ============================================================================

struct text {
    char ** data;
    size_t  count;
    size_t  capacity;
};

void text_grow(struct text * t, size_t need);
void text_put(struct text * t, const char * s, size_t n);
void text_puts(struct text * t, const char * s);
void text_free(struct text * t);

// ============================================================================
// maps -- open-addressed hash map (int64 or struct chars keys)
// ============================================================================

enum map_key { MAP_KEY_INT, MAP_KEY_CHARS };

#define MAP_EMPTY     0
#define MAP_LIVE      1
#define MAP_TOMBSTONE 2

struct map {
    uint8_t *    states;
    void *       keys;
    void *       values;
    size_t       count;
    size_t       capacity;
    size_t       key_size;
    size_t       value_size;
    enum map_key key_kind;
    void         (*value_free)(void *);
};

// Optional explicit setup. Needed only when storing values that own
// resources (pass `value_free`) or when using the generic map_put/map_get
// with a custom key kind. For the common case a zero-initialized map plus
// the typed accessors below is enough -- see map_puti/map_puts.
void map_init(struct map * m, enum map_key kk, size_t ks, size_t vs,
              void (*vf)(void *));

// Generic key/value access. Requires the map to be initialized (via
// map_init or a prior typed put): asserts value_size > 0.
void * map_put(struct map * m, const void * k, const void * v);
void * map_get(struct map * m, const void * k);
void   map_remove(struct map * m, const void * k);
void   map_free(struct map * m);
void   map_for_each(struct map * m,
                    void (*fn)(const void * k, void * v, void * ctx),
                    void * ctx);

// Typed accessors. These work on a zero-initialized map -- `struct map m =
// {0};` -- with no map_init() call: the first put captures the key kind /
// size and the value size (taken via sizeof at the call site, so `v` must
// be a typed pointer to the value, e.g. &my_int). value_free defaults to
// NULL; if stored values own resources set `m.value_free` before the first
// put (e.g. m.value_free = chars_free_v).
void * map_put_int(struct map * m, int64_t k, const void * v, size_t vs);
void * map_get_int(struct map * m, int64_t k);
void   map_remove_int(struct map * m, int64_t k);
void * map_put_str(struct map * m, const char * k, const void * v, size_t vs);
void * map_get_str(struct map * m, const char * k);
void   map_remove_str(struct map * m, const char * k);

#define map_puti(m, k, v)    map_put_int((m), (int64_t)(k), (v), sizeof(*(v)))
#define map_geti(m, k)       map_get_int((m), (int64_t)(k))
#define map_removei(m, k)    map_remove_int((m), (int64_t)(k))
#define map_puts(m, k, v)    map_put_str((m), (k), (v), sizeof(*(v)))
#define map_gets(m, k)       map_get_str((m), (k))
#define map_removes(m, k)    map_remove_str((m), (k))

// Convenience value_free for maps whose values are `struct chars`.
void chars_free_v(void * s);

// ============================================================================
// rng -- xorshift64*; good enough for token sampling
// ============================================================================

struct rng { uint64_t state; };

void     rng_seed(struct rng * r, uint64_t s);
uint64_t rng_next(struct rng * r);
float    rng_uniform(struct rng * r);

#ifdef __cplusplus
}
#endif
#endif // CORE_H
