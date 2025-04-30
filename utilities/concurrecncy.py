import concurrent.futures
from typing import Callable, TypeVar, ParamSpec, Sequence, Generic, Protocol, Coroutine


class SupportsEquality(Protocol):
    def __eq__(self, other: object) -> bool:
        pass


PREDICATE_INPUT = ParamSpec("PREDICATE_INPUT")
PREDICATE_OUTPUT = TypeVar("PREDICATE_OUTPUT", bound=SupportsEquality)
OUTCOME_OUTPUT = TypeVar("OUTCOME_OUTPUT")


class SpeculativeError(KeyError, Generic[PREDICATE_OUTPUT]):
    def __init__(self, message: str, predicate_output: PREDICATE_OUTPUT):
        super().__init__(message)
        self.predicate_output = predicate_output


def speculative_execution(
        predicate: Callable[PREDICATE_INPUT, PREDICATE_OUTPUT],
        outcome: Callable[[PREDICATE_OUTPUT], OUTCOME_OUTPUT],
        outcome_inputs: Sequence[PREDICATE_OUTPUT],
        *predicate_args: PREDICATE_INPUT.args,
        max_workers: int | None = None,
        use_predicate_output: bool = True,
        **predicate_kwargs: PREDICATE_INPUT.kwargs,
) -> OUTCOME_OUTPUT:
    """
    Execute a predicate and multiple outcomes concurrently. Will only wait for the outcome that matches the predicate output.
    Note some outcomes may still be running when the function returns (so make sure no unintentional side effects will occur).

    Some optimization considerations:
    - The outcome inputs will be executed according to their order, their order should be according to their likelihood of being the predicate output.
    - The predicate function is executed first. After it is finished, outcomes that do not match the output and did not start yet will be canceled.

    :param predicate: The predicate function to be executed.
    :param outcome: The outcome function to be executed.
    :param outcome_inputs: A sequence of inputs for the outcome function (to be executed concurrently).
    :param predicate_args: The arguments to be passed to the predicate function.
    :param max_workers: An optional maximum number of worker threads to use for concurrent execution.
    :param use_predicate_output: Only used if the predicate output is not in the outcome inputs.
    If True, the predicate output will be used as input for the outcome function.
    If False, a SpeculativeException will be raised with the predicate output in the exception's field.
    :param predicate_kwargs: The keyword arguments to be passed to the predicate function.
    :return: The result of the outcome function corresponding to the predicate output.
    """
    if max_workers is None:
        max_workers = len(outcome_inputs) + 1
    else:
        max_workers = min(max_workers, len(outcome_inputs) + 1)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    try:
        predicate_future = executor.submit(predicate, *predicate_args, **predicate_kwargs)

        outcome_outputs = [
            executor.submit(outcome, outcome_input)
            for outcome_input in outcome_inputs
        ]

        predicate_output = predicate_future.result()

        matching_outcome_future = None
        for outcome_input, outcome_future in zip(outcome_inputs, outcome_outputs):
            if predicate_output == outcome_input:
                matching_outcome_future = outcome_future
            else:
                outcome_future.cancel()

        if matching_outcome_future is None:
            if use_predicate_output:
                return outcome(predicate_output)
            else:
                raise SpeculativeError(
                    f"Predicate output {predicate_output} not found in outcome inputs.",
                    predicate_output,
                )
        else:
            return matching_outcome_future.result()
    finally:
        executor.shutdown(wait=False)


COROUTINE_INPUT = ParamSpec("COROUTINE_INPUT")
COROUTINE_OUTPUT = TypeVar("COROUTINE_OUTPUT", bound=Coroutine)


def background_execution(
        coroutine: Callable[COROUTINE_INPUT, COROUTINE_OUTPUT],
        *args: COROUTINE_INPUT.args,
        **kwargs: COROUTINE_INPUT.kwargs,
) -> None:
    """
    Run a coroutine in the background in a separate thread.
    This function does not wait for the coroutine to finish.

    :param coroutine: The coroutine function to be executed.
    :param args: The arguments to be passed to the coroutine function.
    :param kwargs: The keyword arguments to be passed to the coroutine function.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        executor.submit(coroutine, *args, **kwargs)
    finally:
        executor.shutdown(wait=False)
