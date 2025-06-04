import asyncio
import time

import pytest

from utilities.concurrency import (
    SpeculativeError,
    speculative_execution,
    background_execution
)


class TestSpeculativeExecution:
    def test_speculative_execution_matching_output(self):
        def predicate():
            return 2

        def outcome(x):
            return f"Result: {x}"

        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[1, 2, 3]
        )
        expected = "Result: 2"
        assert result == expected

    def test_speculative_execution_with_args(self):
        def predicate(x, y):
            return x + y

        def outcome(x):
            return x * 2

        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[3, 5, 7],
            x=2,
            y=3
        )
        expected = 10  # (2 + 3) matches 5, then 5 * 2 = 10
        assert result == expected

    def test_speculative_execution_no_matching_outcome_inputs(self):
        def predicate():
            return 4

        def outcome(x):
            return f"Result: {x}"

        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[1, 2, 3],
            use_predicate_output=True
        )
        expected = "Result: 4"
        assert result == expected

    def test_speculative_execution_no_match_raises(self):
        def predicate():
            return 4

        def outcome(x):
            return f"Result: {x}"

        with pytest.raises(SpeculativeError) as exc_info:
            speculative_execution(
                predicate=predicate,
                outcome=outcome,
                outcome_inputs=[1, 2, 3],
                use_predicate_output=False
            )
        expected_predicate_output = 4
        assert exc_info.value.predicate_output == expected_predicate_output

    def test_speculative_execution_empty_outcome_inputs(self):
        def predicate():
            return 1

        def outcome(x):
            return x * 2

        # Empty outcome inputs should still work with use_predicate_output=True
        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[],
            use_predicate_output=True
        )
        expected = 2  # Predicate returns 1, outcome should return 1 * 2
        assert result == expected

    def test_speculative_execution_none_predicate_output(self):
        def predicate():
            return None

        def outcome(x):
            return f"Result: {x}"

        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[None, 1, 2]
        )
        expected = "Result: None"
        assert result == expected

    def test_speculative_execution_exception_in_predicate(self):
        def predicate():
            raise ValueError("Predicate error")

        def outcome(x):
            return f"Result: {x}"

        with pytest.raises(ValueError, match="Predicate error"):
            speculative_execution(
                predicate=predicate,
                outcome=outcome,
                outcome_inputs=[1, 2, 3]
            )

    def test_speculative_execution_exception_in_matching_outcome(self):
        def predicate():
            return 2

        def outcome(x):
            if x == 2:
                raise ValueError("Outcome error")
            return f"Result: {x}"

        with pytest.raises(ValueError, match="Outcome error"):
            speculative_execution(
                predicate=predicate,
                outcome=outcome,
                outcome_inputs=[1, 2, 3]
            )

    def test_speculative_execution_exception_in_non_matching_outcome(self):
        def predicate():
            return 2

        def outcome(x):
            if x == 1:
                raise ValueError("Should not affect result")
            return f"Result: {x}"

        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[1, 2, 3]
        )
        expected = "Result: 2"
        assert result == expected


class TestBackgroundExecution:
    def test_background_execution(self):
        execution_completed = False

        async def async_task():
            nonlocal execution_completed
            await asyncio.sleep(0.1)
            execution_completed = True

        background_execution(async_task)
        time.sleep(0.2)
        assert execution_completed

    def test_speculative_execution_cancels_unneeded_tasks(self):
        def predicate():
            time.sleep(0.1)  # Small delay to ensure outcomes start
            return 2

        def outcome(x):
            if x != 2:
                time.sleep(1)  # Long delay for non-matching outcomes
            return f"Result: {x}"

        start_time = time.perf_counter()
        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[1, 2, 3]
        )
        elapsed_time = time.perf_counter() - start_time

        assert result == "Result: 2"
        assert elapsed_time < 0.5  # Should complete quickly due to cancellation

    def test_speculative_execution_concurrent_execution(self):
        completed_order = []

        def predicate():
            time.sleep(0.1)
            completed_order.append("predicate")
            return 2

        def outcome(x):
            time.sleep(0.1)
            completed_order.append(f"outcome_{x}")
            return f"Result: {x}"

        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[1, 2, 3]
        )

        assert result == "Result: 2"
        # Verify that some outcomes started before predicate completed
        assert len(completed_order) > 1

    def test_background_execution_with_exception(self):
        async def failing_task():
            raise ValueError("Test error")

        # Should not raise an exception
        background_execution(failing_task)

    @pytest.mark.asyncio
    async def test_background_execution_completes(self):
        result = None

        async def async_task():
            nonlocal result
            await asyncio.sleep(0.1)
            result = "completed"

        background_execution(async_task)
        await asyncio.sleep(0.2)
        expected = "completed"
        assert result == expected

    @pytest.mark.asyncio
    async def test_background_execution_multiple_concurrent(self):
        results = set()

        async def async_task(value):
            await asyncio.sleep(0.1)
            results.add(value)

        for task_index in range(3):
            background_execution(async_task, task_index)
        await asyncio.sleep(0.2)

        expected_results = {0, 1, 2}
        assert results == expected_results

    def test_background_execution_non_coroutine(self):
        def non_coroutine_function():
            pass

        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            background_execution(non_coroutine_function)


    def test_speculative_execution_custom_equality(self):
        class CustomEqual:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return isinstance(other, CustomEqual) and self.value == other.value

        def predicate() -> CustomEqual:
            return CustomEqual(2)

        def outcome(x: CustomEqual) -> str:
            return f"Result: {x.value}"

        result = speculative_execution(
            predicate=predicate,
            outcome=outcome,
            outcome_inputs=[CustomEqual(1), CustomEqual(2), CustomEqual(3)]
        )
        expected = "Result: 2"
        assert result == expected


    @pytest.mark.asyncio
    async def test_background_execution_in_running_loop(self):
        result = []

        async def async_task():
            await asyncio.sleep(0.1)
            result.append("completed")

        background_execution(async_task)
        await asyncio.sleep(0.2)
        expected = ["completed"]
        assert result == expected


    def test_background_execution_without_running_loop(self):
        result = []

        async def async_task():
            await asyncio.sleep(0.1)
            result.append("completed")

        background_execution(async_task)
        time.sleep(0.3)
        expected = ["completed"]
        assert result == expected


    @pytest.mark.asyncio
    async def test_multiple_background_executions_in_running_loop(self):
        results = set()

        async def async_task(task_id: int):
            await asyncio.sleep(0.1)
            results.add(task_id)

        for task_index in range(3):
            background_execution(async_task, task_index)

        await asyncio.sleep(0.3)
        expected_results = {0, 1, 2}
        assert set(results) == expected_results
