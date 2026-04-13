/**
 * Isolated in-memory store for this feature only.
 * No shared state with host platform.
 * TTL: 2 hours. Auto-prune on every write.
 */
const TTL_MS = 2 * 60 * 60 * 1000;
const _store = new Map();

function prune() {
  const now = Date.now();
  for (const [id, val] of _store.entries()) {
    if (now - val.created_at > TTL_MS) _store.delete(id);
  }
}

module.exports = {
  set(id, value) { prune(); _store.set(id, { ...value, created_at: Date.now() }); },
  get(id)        { return _store.get(id) || null; },
  delete(id)     { _store.delete(id); },
  // Cache lookup by topic (avoid re-generating identical topics)
  findByTopic(topic) {
    for (const val of _store.values()) {
      if (val.meta?.topic?.toLowerCase() === topic.toLowerCase()) return val;
    }
    return null;
  },
};
