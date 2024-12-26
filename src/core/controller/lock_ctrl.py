import threading
from collections import defaultdict as dt


class ResourceGuard:
    """Handles synchronization mechanisms for individual entities."""

    def __init__(self):
        self.sync_locks = dt(threading.Lock)

    def fetch_lock(self, entity_id: str) -> threading.Lock:
        """Fetch the synchronization object corresponding to a given entity."""
        return self.sync_locks[entity_id]

    def release_lock(self, entity_id: str):
        """Optionally discard the synchronization object when no longer necessary."""
        if entity_id in self.sync_locks:
            del self.sync_locks[entity_id]
