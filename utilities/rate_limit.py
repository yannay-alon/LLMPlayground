import asyncio
from functools import wraps

import time
import sys
import threading
from typing import Callable, ParamSpec, TypeVar, Hashable, Generic

INPUT = ParamSpec("INPUT")
OUTPUT = TypeVar("OUTPUT")


class RateLimitException(Exception):
    def __init__(self, message: str, remaining_period_in_seconds: float):
        super().__init__(message)
        self.remaining_period_in_seconds = remaining_period_in_seconds


class RateLimiter:
    _instances: dict[Hashable, "RateLimiter"] = {}

    def __init__(
            self,
            calls: int = 15,
            period_in_seconds: float = 900,
            clock: Callable[[], float] | None = None,
            sleep_on_rate_limit: bool = False,
            *,
            shared_limiter_name: Hashable | None = None,
    ):
        if getattr(self, "_initialized", False):
            return
        self.clamped_calls = max(1, min(sys.maxsize, calls))
        self.period_in_seconds = period_in_seconds
        self.clock = clock if clock is not None else time.perf_counter
        self.sleep_on_rate_limit = sleep_on_rate_limit

        self.last_reset = self.clock()
        self.num_calls = 0

        self.lock = threading.RLock()
        self.async_lock = asyncio.Lock()

        self._initialized = True

    def __new__(cls, *args, shared_limiter_name: Hashable | None = None, **kwargs):
        if shared_limiter_name is None:
            return super().__new__(cls)

        if shared_limiter_name not in cls._instances:
            cls._instances[shared_limiter_name] = super().__new__(cls)
        return cls._instances[shared_limiter_name]

    def __call__(self, function: Callable[INPUT, OUTPUT]) -> Callable[INPUT, OUTPUT]:
        @wraps(function)
        def sync_wrapper(*args: INPUT.args, **kwargs: INPUT.kwargs) -> OUTPUT:
            with self:
                return function(*args, **kwargs)

        @wraps(function)
        async def async_wrapper(*args: INPUT.args, **kwargs: INPUT.kwargs) -> OUTPUT:
            async with self:
                return await function(*args, **kwargs)

        if asyncio.iscoroutinefunction(function):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper

        return wrapper

    def __enter__(self):
        while True:
            try:
                with self.lock:
                    self.__add_call()
                break
            except RateLimitException as exception:
                if not self.sleep_on_rate_limit:
                    raise
                time.sleep(exception.remaining_period_in_seconds)
        return self

    def __exit__(self, *args, **kwargs):
        return

    async def __aenter__(self):
        while True:
            try:
                async with self.async_lock:
                    self.__add_call()
                break
            except RateLimitException as exception:
                if not self.sleep_on_rate_limit:
                    raise
                await asyncio.sleep(exception.remaining_period_in_seconds)
        return self

    async def __aexit__(self, *args, **kwargs):
        return

    def __remaining_period_in_seconds(self):
        elapsed = self.clock() - self.last_reset
        return self.period_in_seconds - elapsed

    def __add_call(self):
        remaining_period_in_seconds = self.__remaining_period_in_seconds()

        if remaining_period_in_seconds <= 0:
            self.num_calls = 0
            self.last_reset = self.clock()

        if self.clamped_calls <= self.num_calls:
            raise RateLimitException("too many calls", remaining_period_in_seconds)
        self.num_calls += 1
