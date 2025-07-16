"""A simple in-memory cache for caching arbitrary objects."""

from typing import Any, Dict, Optional

class InMemoryCache:
    """
    A simple in-memory cache implementation using a Python dictionary.
    """
    def __init__(self):
        """Initializes the InMemoryCache."""
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The cached item, or None if the key is not found.
        """
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Adds or updates an item in the cache.

        Args:
            key: The key of the item to set.
            value: The value of the item to set.
        """
        self._cache[key] = value

    def clear(self) -> None:
        """Clears all items from the cache."""
        self._cache.clear()
