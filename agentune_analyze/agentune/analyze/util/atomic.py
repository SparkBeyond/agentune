from threading import Lock


class AtomicInt:
    def __init__(self, value: int = 0):
        self.__value = value
        self.__lock = Lock()

    # Get and put are already atomic under GIL semantics, supposedly: https://docs.python.org/3/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe
    # But using GIL to ensure threadsafety (concurrently with locking operations) scares me, so I take the lock anyway.

    def get(self) -> int: 
        with self.__lock:
            return self.__value 
    
    def put(self, value: int) -> None:
        with self.__lock:
            self.__value = value

    def inc_and_get(self, diff: int = 1) -> int:
        with self.__lock:
            self.__value += diff
            return self.__value

    def setmax(self, max_value: int) -> None:
        with self.__lock:
            self.__value = max(self.__value, max_value)

    def setmin(self, min_value: int) -> None:
        with self.__lock:
            self.__value = min(self.__value, min_value)
