from functools import wraps

import time
import sys
import threading
from typing import Callable, ParamSpec, TypeVar

INPUT = ParamSpec("INPUT")
OUTPUT = TypeVar("OUTPUT")


class RateLimitException(Exception):
    def __init__(self, message: str, remaining_period: float):
        super().__init__(message)
        self.remaining_period = remaining_period


class RateLimiter:
    def __init__(self, calls: int = 15, period: float = 900, clock: Callable[[], float] | None = None):
        self.clamped_calls = max(1, min(sys.maxsize, calls))
        self.period = period
        self.clock = clock if clock is not None else time.perf_counter

        self.last_reset = self.clock()
        self.num_calls = 0

        self.lock = threading.RLock()

    def __call__(self, function: Callable[INPUT, OUTPUT]) -> Callable[INPUT, OUTPUT]:
        @wraps(function)
        def wrapper(*args: INPUT.args, **kwargs: INPUT.kwargs) -> OUTPUT:
            with self.lock:
                remaining_period = self.__remaining_period()

                if remaining_period <= 0:
                    self.num_calls = 0
                    self.last_reset = self.clock()

                self.num_calls += 1

                if self.num_calls > self.clamped_calls:
                    raise RateLimitException("too many calls", remaining_period)

            return function(*args, **kwargs)

        return wrapper

    def __remaining_period(self):
        elapsed = self.clock() - self.last_reset
        return self.period - elapsed


def sleep_and_retry(function: Callable[INPUT, OUTPUT]) -> Callable[INPUT, OUTPUT]:
    @wraps(function)
    def wrapper(*args: INPUT.args, **kwargs: INPUT.kwargs) -> OUTPUT:
        while True:
            try:
                return function(*args, **kwargs)
            except RateLimitException as exception:
                time.sleep(exception.remaining_period)

    return wrapper
