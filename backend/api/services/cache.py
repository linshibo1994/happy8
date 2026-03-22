import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheItem:
    value: Any
    expire_at: Optional[float]


class CacheManager:
    def __init__(self, default_ttl_seconds: int = 300):
        self.default_ttl_seconds = default_ttl_seconds
        self._store: Dict[str, CacheItem] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _is_expired(self, item: CacheItem) -> bool:
        return item.expire_at is not None and item.expire_at < time.time()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None

            if self._is_expired(item):
                self._store.pop(key, None)
                self._misses += 1
                return None

            self._hits += 1
            return item.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        expire_at = None if ttl_seconds <= 0 else time.time() + ttl_seconds
        with self._lock:
            self._store[key] = CacheItem(value=value, expire_at=expire_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._hits = 0
            self._misses = 0
            return count

    def cleanup(self) -> int:
        removed = 0
        with self._lock:
            for key in list(self._store.keys()):
                if self._is_expired(self._store[key]):
                    self._store.pop(key, None)
                    removed += 1
        return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests else 0.0
            return {
                "size": len(self._store),
                "default_ttl_seconds": self.default_ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 6),
                "total_requests": total_requests,
            }


cache_manager = CacheManager()

