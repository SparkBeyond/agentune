import logging
import queue
import random
import threading
import time
from datetime import timedelta
from queue import Queue
from threading import Event

import pytest
from attrs import field, frozen

from agentune.analyze.util.atomic import AtomicInt
from agentune.analyze.util.pool import PoolClosedError, PoolTimeoutError, ThreadsafePool

_logger = logging.getLogger(__name__)


@frozen
class Resource:
    close_count: AtomicInt = field(init=False, factory=AtomicInt)

    def close(self) -> None:
        self.close_count.inc_and_get()


def test_pool_no_blocking() -> None:
    pool = ThreadsafePool(size=1, factory=lambda: Resource(), acquire_timeout=timedelta())

    with pool.acquire() as resource1: pass
    with pool.acquire() as resource2:
        assert resource2 is resource1, 'Same instance is acquired'
        with pytest.raises(PoolTimeoutError):
            with pool.acquire(): pass

    assert resource1.close_count.get() == 0

    pool.close()
    assert resource1.close_count.get() == 1

    pool.close()
    assert resource1.close_count.get() == 1, 'Not closed again'

    with pytest.raises(PoolClosedError):
        with pool.acquire(): pass


def test_pool_closing_resource_on_exception() -> None:
    pool = ThreadsafePool(size=2, factory=lambda: Resource(), acquire_timeout=timedelta())

    with pytest.raises(ValueError, match='foo'):
        with pool.acquire() as resource:
            assert resource.close_count.get() == 0
            raise ValueError('foo')
    assert resource.close_count.get() == 1
    with pool.acquire() as resource2:
        assert resource2 is not resource, 'A new resource was created'

def test_pool_blocking() -> None:
    pool = ThreadsafePool(size=1, factory=lambda: Resource(), acquire_timeout=timedelta(milliseconds=1))

    def in_thread() -> None:
        with pool.acquire(timeout=timedelta(seconds=1)): pass

    with pool.acquire():
        with pytest.raises(PoolTimeoutError): # Timeout after a short block
            with pool.acquire(): pass

        thread = threading.Thread(target=in_thread, daemon=True)
        thread2 = threading.Thread(target=in_thread, daemon=True)
        thread.start()
        thread2.start()
        time.sleep(0.1)

        assert thread.is_alive(), 'Thread is blocked'
        assert thread2.is_alive(), 'Thread is blocked'

    time.sleep(0.1)
    assert not thread.is_alive() and not thread2.is_alive(), 'Both threads acquired the resource'

def test_pool_closing() -> None:
    pool = ThreadsafePool(size=1, factory=lambda: Resource(), acquire_timeout=timedelta(milliseconds=1))

    saw_error = Event()
    def in_thread_expect_pool_closed() -> None:
        try:
            with pool.acquire(timeout=timedelta(seconds=1)): pass
        except PoolClosedError:
            saw_error.set()

    with pool.acquire() as resource:
        thread = threading.Thread(target=in_thread_expect_pool_closed, daemon=True)
        thread.start()
        pool.close()
        time.sleep(0.1)
        assert not thread.is_alive(), 'Thread is not blocked'
        assert saw_error.is_set(), 'Thread got an error'

        assert resource.close_count.get() == 0, 'Resource not closed while out of the pool'
    assert resource.close_count.get() == 1, 'Resource closed once returned to the shut-down pool'
    with pytest.raises(PoolClosedError):
        with pool.acquire(): pass


def test_multithreaded_pressure() -> None:
    pool = ThreadsafePool(size=5, factory=lambda: Resource(), acquire_timeout=timedelta(seconds=10))

    resources: Queue[Resource] = Queue()
    failed: Queue[Resource] = Queue()

    def drain(q: Queue[Resource]) -> list[Resource]:
        ret: list[Resource] = []
        try:
            while True:
                ret.append(q.get_nowait())
        except queue.Empty:
            return ret

    rnd = random.Random(42)
    def in_thread() -> None:
        try:
            with pool.acquire() as resource:
                resources.put(resource)
                time.sleep(0.01)
                if rnd.random() < 0.5:
                    failed.put(resource)
                    raise ValueError # Pool won't give out this resource again # noqa: TRY301
        except ValueError:
            pass # The thread itself doesn't need to fail (and it creates noise in the pytest output)

    threads = [threading.Thread(target=in_thread, daemon=True) for i in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    all_resources = drain(resources)
    failed_resources = drain(failed)

    assert len(all_resources) == 100, 'Every thread got a resource'
    assert len(set(all_resources)) < 100, 'Some resources were reused'
    assert len(failed_resources) < 100, 'Not all resources were failed'
    assert len(set(failed_resources)) == len(failed_resources)
    assert set(failed_resources).issubset(set(all_resources))


def test_error_in_close() -> None:
    class BadResource:
        def close(self) -> None:
            raise RuntimeError('Close failed')
    
    with ThreadsafePool(BadResource, size=1, acquire_timeout=timedelta(seconds=1)) as pool:
        with pool.acquire() as resource:
            assert isinstance(resource, BadResource)
        with pool.acquire() as resource2:
            assert resource2 is resource # No reason to close it

        with pytest.raises(ValueError, match='foo'), pool.acquire() as resource3:
            assert resource3 is resource2
            raise ValueError('foo')

        with pool.acquire() as resource4:
            assert resource4 is not resource, 'Got a new one'
            pool.close() # Make sure nothing is raised
