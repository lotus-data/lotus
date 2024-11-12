from collections import OrderedDict
from typing import Any

import lotus


class Cache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if not lotus.settings.enable_cache:
            return None

        if key in self.cache:
            lotus.logger.debug(f"Cache hit for {key}")

        return self.cache.get(key)

    def insert(self, key: str, value: Any):
        if not lotus.settings.enable_cache:
            return

        self.cache[key] = value

        # LRU eviction
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def reset(self, max_size: int | None = None):
        self.cache.clear()
        if max_size is not None:
            self.max_size = max_size
