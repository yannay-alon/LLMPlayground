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
        def function():
            return "Executed"

        for _ in range(3):
            assert function() == "Executed"

    def test_sync_rate_limit_exceeds_limit(self, rate_limiter: RateLimiter):
        @rate_limiter
        def function():
            return "Executed"

        for _ in range(3):
            assert function() == "Executed"

        with pytest.raises(RateLimitException):
            function()

    @pytest.mark.asyncio
    async def test_async_rate_limit_within_limit(self, rate_limiter: RateLimiter):
        @rate_limiter
        async def function():
            return "Executed"

        for _ in range(3):
            assert await function() == "Executed"

    @pytest.mark.asyncio
    async def test_async_rate_limit_exceeds_limit(self, rate_limiter: RateLimiter):
        @rate_limiter
        async def function():
            return "Executed"

        for _ in range(3):
            assert await function() == "Executed"

        with pytest.raises(RateLimitException):
            await function()

    def test_shared_limiter(self, shared_rate_limiter: RateLimiter, empty_shared_rate_limiter: RateLimiter):
        assert shared_rate_limiter is empty_shared_rate_limiter
        assert empty_shared_rate_limiter.clamped_calls == 3
        assert empty_shared_rate_limiter.period_in_seconds == 1

        @shared_rate_limiter
        def function_1():
            return "Function 1 executed"

        @empty_shared_rate_limiter
        def function_2():
            return "Function 2 executed"

        assert function_1() == "Function 1 executed"
        assert function_2() == "Function 2 executed"
        assert function_1() == "Function 1 executed"

        with pytest.raises(RateLimitException):
            function_2()

    def test_limit_reset_after_period(self, rate_limiter: RateLimiter):
        @rate_limiter
        def function():
            return "Executed"

        for _ in range(3):
            assert function() == "Executed"

        with pytest.raises(RateLimitException):
            function()

        time.sleep(1.1)
        assert function() == "Executed"

    def test_sleep_on_rate_limit(self, sleeping_rate_limiter: RateLimiter):
        @sleeping_rate_limiter
        def function():
            return "Executed"

        for _ in range(3):
            assert function() == "Executed"

        start_time = time.perf_counter()
        assert function() == "Executed"
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        assert run_duration >= 0.99

    def test_sleep_on_shared_limiter(
            self,
            sleeping_shared_rate_limiter: RateLimiter,
            sleeping_empty_shared_rate_limiter: RateLimiter
    ):
        @sleeping_shared_rate_limiter
        def function_1():
            return "Function 1 executed"

        @sleeping_empty_shared_rate_limiter
        def function_2():
            return "Function 2 executed"

        assert function_1() == "Function 1 executed"
        assert function_2() == "Function 2 executed"
        assert function_2() == "Function 2 executed"

        start_time = time.perf_counter()
        function_1()
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        assert run_duration >= 0.99

    def test_concurrent_access(self, rate_limiter: RateLimiter):
        successful_calls = 0
        failed_calls = 0
        lock = threading.Lock()
        thread_count = 10

        @rate_limiter
        def function():
            return "Executed"

        def worker():
            nonlocal successful_calls, failed_calls
            try:
                function()
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


        @rate_limiter
        def sync_function():
            return "Sync"

        @rate_limiter
        async def async_function():
            return "Async"


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


        threads = []
        for _ in range(2):
            thread = threading.Thread(target=thread_run_async)
            thread.start()
            threads.append(thread)


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


        for thread in threads:
            thread.join()


        assert successful_calls == 3
        assert failed_calls == 1


        await asyncio.sleep(1.1)


        successful_calls = 0
        failed_calls = 0


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

    def test_sync_context_manager(self, rate_limiter: RateLimiter):
        for _ in range(3):
            with rate_limiter:
                pass

        with pytest.raises(RateLimitException):
            with rate_limiter:
                pass

        time.sleep(1.1)
        with rate_limiter:
            pass

    def test_sync_context_manager_shared(self, shared_rate_limiter: RateLimiter, empty_shared_rate_limiter: RateLimiter):
        with shared_rate_limiter:
            pass
        with empty_shared_rate_limiter:
            pass
        with shared_rate_limiter:
            pass

        with pytest.raises(RateLimitException):
            with empty_shared_rate_limiter:
                pass

    def test_sync_context_manager_with_sleep(self, sleeping_rate_limiter: RateLimiter):
        for _ in range(3):
            with sleeping_rate_limiter:
                pass

        start_time = time.perf_counter()
        with sleeping_rate_limiter:
            pass
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        assert run_duration >= 0.99

    @pytest.mark.asyncio
    async def test_async_context_manager(self, rate_limiter: RateLimiter):

        for _ in range(3):
            async with rate_limiter:
                pass


        with pytest.raises(RateLimitException):
            async with rate_limiter:
                pass


        await asyncio.sleep(1.1)
        async with rate_limiter:
            pass

    @pytest.mark.asyncio
    async def test_async_context_manager_shared(self, shared_rate_limiter: RateLimiter, empty_shared_rate_limiter: RateLimiter):

        async with shared_rate_limiter:
            pass
        async with empty_shared_rate_limiter:
            pass
        async with shared_rate_limiter:
            pass

        with pytest.raises(RateLimitException):
            async with empty_shared_rate_limiter:
                pass

    @pytest.mark.asyncio
    async def test_async_context_manager_with_sleep(self, sleeping_rate_limiter: RateLimiter):

        for _ in range(3):
            async with sleeping_rate_limiter:
                pass


        start_time = time.perf_counter()
        async with sleeping_rate_limiter:
            pass
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        assert run_duration >= 0.99

    @pytest.mark.asyncio
    async def test_mixed_context_manager_calls(self, rate_limiter: RateLimiter):
        successful_calls = 0
        failed_calls = 0
        lock = threading.Lock()


        def thread_run_async():
            nonlocal successful_calls, failed_calls
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(async_context_manager_task())
                with lock:
                    successful_calls += 1
            except RateLimitException:
                with lock:
                    failed_calls += 1
            finally:
                loop.close()

        async def async_context_manager_task():
            async with rate_limiter:
                return "Async"


        threads = []
        for _ in range(2):
            thread = threading.Thread(target=thread_run_async)
            thread.start()
            threads.append(thread)


        try:
            with rate_limiter:
                pass
            successful_calls += 1
        except RateLimitException:
            failed_calls += 1


        try:
            async with rate_limiter:
                pass
            successful_calls += 1
        except RateLimitException:
            failed_calls += 1


        for thread in threads:
            thread.join()

        assert successful_calls == 3
        assert failed_calls == 1