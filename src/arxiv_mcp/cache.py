"""TTL-based LRU cache with expiry for paper metadata and text."""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheEntry:
    """A single cached value with its expiry timestamp."""

    value: Any
    expires_at: float


class TTLCache:
    """A simple TTL + LRU cache with a max size.

    When the cache is full, the least recently used entry is evicted.
    Expired entries are cleaned up on access (lazy eviction).

    Thread-safe: all operations are guarded by a lock, since sync
    functions running via asyncio.to_thread share the global caches.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: float = 3600.0):
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value if it exists and hasn't expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            if time.monotonic() > entry.expires_at:
                # Expired — remove and treat as miss
                del self._store[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        """Store a value with the configured TTL."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = CacheEntry(
                    value=value,
                    expires_at=time.monotonic() + self._ttl_seconds,
                )
            else:
                # Evict LRU if at capacity
                if len(self._store) >= self._max_size:
                    self._store.popitem(last=False)
                self._store[key] = CacheEntry(
                    value=value,
                    expires_at=time.monotonic() + self._ttl_seconds,
                )

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total * 100:.1f}%" if total > 0 else "N/A",
            }


# Paper metadata: small, accessed frequently, 1-hour TTL
metadata_cache = TTLCache(max_size=500, ttl_seconds=3600)

# Extracted text: large, accessed less often, 24-hour TTL
text_cache = TTLCache(max_size=100, ttl_seconds=86400)
