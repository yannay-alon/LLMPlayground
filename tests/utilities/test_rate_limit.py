import asyncio
import threading
import time

import pytest

from utilities.rate_limit import RateLimiter, RateLimitException


class TestRateLimiter:
    @pytest.fixture
    def rate_limiter(self) -> RateLimiter:
        return RateLimiter(calls=3, period_in_seconds=1)

    @pytest.fixture
    def shared_rate_limiter(self) -> RateLimiter:
        return RateLimiter(calls=3, period_in_seconds=1, shared_limiter_name="shared")

    @pytest.fixture
    def empty_shared_rate_limiter(self) -> RateLimiter:
        return RateLimiter(shared_limiter_name="shared")

    @pytest.fixture
    def sleeping_rate_limiter(self) -> RateLimiter:
        return RateLimiter(calls=3, period_in_seconds=1, sleep_on_rate_limit=True)

    @pytest.fixture
    def sleeping_shared_rate_limiter(self) -> RateLimiter:
        return RateLimiter(calls=3, period_in_seconds=1, shared_limiter_name="sleeping", sleep_on_rate_limit=True)

    @pytest.fixture
    def sleeping_empty_shared_rate_limiter(self) -> RateLimiter:
        return RateLimiter(shared_limiter_name="sleeping")

    def test_sync_rate_limit_within_limit(self, rate_limiter: RateLimiter):
        @rate_limiter
        def test_function():
            return "Executed"

        for _ in range(3):
            assert test_function() == "Executed"

    def test_sync_rate_limit_exceeds_limit(self, rate_limiter: RateLimiter):
        @rate_limiter
        def test_function():
            return "Executed"

        for _ in range(3):
            assert test_function() == "Executed"

        with pytest.raises(RateLimitException):
            test_function()

    @pytest.mark.asyncio
    async def test_async_rate_limit_within_limit(self, rate_limiter: RateLimiter):
        @rate_limiter
        async def test_function():
            return "Executed"

        for _ in range(3):
            assert await test_function() == "Executed"

    @pytest.mark.asyncio
    async def test_async_rate_limit_exceeds_limit(self, rate_limiter: RateLimiter):
        @rate_limiter
        async def test_function():
            return "Executed"

        for _ in range(3):
            assert await test_function() == "Executed"

        with pytest.raises(RateLimitException):
            await test_function()

    def test_shared_limiter(self, shared_rate_limiter: RateLimiter, empty_shared_rate_limiter: RateLimiter):
        assert shared_rate_limiter is empty_shared_rate_limiter
        assert empty_shared_rate_limiter.clamped_calls == 3
        assert empty_shared_rate_limiter.period_in_seconds == 1

        @shared_rate_limiter
        def test_function_1():
            return "Function 1 executed"

        @empty_shared_rate_limiter
        def test_function_2():
            return "Function 2 executed"

        assert test_function_1() == "Function 1 executed"
        assert test_function_2() == "Function 2 executed"
        assert test_function_1() == "Function 1 executed"

        with pytest.raises(RateLimitException):
            test_function_2()

    def test_limit_reset_after_period(self, rate_limiter: RateLimiter):
        @rate_limiter
        def test_function():
            return "Executed"

        for _ in range(3):
            assert test_function() == "Executed"

        with pytest.raises(RateLimitException):
            test_function()

        time.sleep(1.1)
        assert test_function() == "Executed"

    def test_sleep_on_rate_limit(self, sleeping_rate_limiter: RateLimiter):
        @sleeping_rate_limiter
        def test_function():
            return "Executed"

        for _ in range(3):
            assert test_function() == "Executed"

        start_time = time.perf_counter()
        assert test_function() == "Executed"
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        assert run_duration >= 0.99

    def test_sleep_on_shared_limiter(
            self,
            sleeping_shared_rate_limiter: RateLimiter,
            sleeping_empty_shared_rate_limiter: RateLimiter
    ):
        @sleeping_shared_rate_limiter
        def test_function_1():
            return "Function 1 executed"

        @sleeping_empty_shared_rate_limiter
        def test_function_2():
            return "Function 2 executed"

        assert test_function_1() == "Function 1 executed"
        assert test_function_2() == "Function 2 executed"
        assert test_function_2() == "Function 2 executed"

        start_time = time.perf_counter()
        test_function_1()
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        assert run_duration >= 0.99

    def test_concurrent_access(self, rate_limiter: RateLimiter):
        successful_calls = 0
        failed_calls = 0
        lock = threading.Lock()
        thread_count = 10

        @rate_limiter
        def test_function():
            return "Executed"

        def worker():
            nonlocal successful_calls, failed_calls
            try:
                test_function()
                with lock:
                    successful_calls += 1
            except RateLimitException:
                with lock:
                    failed_calls += 1

        threads = []
        for _ in range(thread_count):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        assert successful_calls == 3
        assert failed_calls == 7

    @pytest.mark.asyncio
    async def test_async_shared_limiter(self, shared_rate_limiter: RateLimiter, empty_shared_rate_limiter: RateLimiter):
        @shared_rate_limiter
        async def async_function_1():
            return "Async 1"

        @empty_shared_rate_limiter
        async def async_function_2():
            return "Async 2"

        for _ in range(3):
            assert await async_function_1() == "Async 1"

        with pytest.raises(RateLimitException):
            await async_function_2()

        await asyncio.sleep(1.1)
        assert await async_function_1() == "Async 1"

    @pytest.mark.asyncio
    async def test_mixed_thread_and_async_calls(self, rate_limiter: RateLimiter):
        successful_calls = 0
        failed_calls = 0
        lock = threading.Lock()

        # Create both sync and async functions that share the same limiter
        @rate_limiter
        def sync_function():
            return "Sync"

        @rate_limiter
        async def async_function():
            return "Async"

        # Helper function to run async function in a thread
        def thread_run_async():
            nonlocal successful_calls, failed_calls
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(async_function())
                with lock:
                    successful_calls += 1
            except RateLimitException:
                with lock:
                    failed_calls += 1
            finally:
                loop.close()

        # Create and start threads that will run the async function
        threads = []
        for _ in range(2):
            thread = threading.Thread(target=thread_run_async)
            thread.start()
            threads.append(thread)

        # Run sync function in main thread
        try:
            sync_function()
            successful_calls += 1
        except RateLimitException:
            failed_calls += 1

        # Run async function in current event loop
        try:
            await async_function()
            successful_calls += 1
        except RateLimitException:
            failed_calls += 1

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify total calls match our rate limit
        assert successful_calls == 3  # Based on fixture settings (calls=3)
        assert failed_calls == 1  # We made 4 attempts total (2 thread + 1 sync + 1 async)

        # Wait for the period to expire
        await asyncio.sleep(1.1)

        # Verify we can make calls again after the period expires
        successful_calls = 0
        failed_calls = 0

        # Try one of each type after reset
        try:
            sync_function()
            successful_calls += 1
        except RateLimitException:
            failed_calls += 1

        try:
            await async_function()
            successful_calls += 1
        except RateLimitException:
            failed_calls += 1

        assert successful_calls == 2
        assert failed_calls == 0
