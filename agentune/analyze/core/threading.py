"""Utilities for safely sending values between threads."""

from abc import ABC, abstractmethod
from typing import Self


class CopyToThread(ABC):
    @abstractmethod
    def copy_to_thread(self) -> Self: 
        """Make a shallow copy of this immutable object that can be used in another thread.
        
        To send the same object to multiple threads, create a copy for each.
        The copy does not have to be created on the thread that will use it, but it cannot be used by other threads first.

        Mutable objects are not expected to implement this protocol; if they do, they should either
        make a deep copy or document what precisely they are doing.
        """
        ...
