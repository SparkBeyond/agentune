from threading import Lock


class AtomicInt:
    def __init__(self, value: int = 0):
        self.__value = value
        self.__lock = Lock()

    # Get and put are already atomic under GIL semantics, supposedly: https://docs.python.org/3/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe
    # At least, if your native int is really int sized!
    # But using GIL to ensure threadsafety scares me.

    def get(self) -> int: 
        with self.__lock:
            return self.__value 
    
    def put(self, value: int) -> None:
        with self.__lock:
            self.__value = value

    def inc_and_get(self) -> int:
        with self.__lock:
            self.__value += 1
            return self.__value
    
