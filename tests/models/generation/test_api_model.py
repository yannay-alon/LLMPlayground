from typing import Any, Iterable, AsyncIterable

import pytest

from components.documents import Document
from components.messages import UserMessage, SystemMessage, BaseMessage
from components.responses import Completion
from components.responses.choice import ParsedType
from components.tools import Tool
from models.generation.remote.api_model import APIModel


class TestAPIModel:
    @pytest.fixture
    def simple_completion(self, ) -> Completion[ParsedType]:
        return Completion(
            choices=[],
            usage=None
        )

    @pytest.fixture
    def concrete_api_model(self, simple_completion) -> type[APIModel]:
        class ConcreteAPIModel(APIModel):
            def _invoke(
                    self,
                    messages: list[BaseMessage],
                    stream: bool,
                    tools: dict[str, Tool] | None,
                    documents: list[Document] | None,
                    response_format: type[ParsedType] | None,
                    max_tokens: int | None,
                    temperature: float
            ) -> Completion[ParsedType] | Iterable[Completion[ParsedType]]:
                return simple_completion

            async def _async_invoke(
                    self,
                    messages: list[BaseMessage],
                    stream: bool,
                    tools: dict[str, Tool] | None,
                    documents: list[Document] | None,
                    response_format: type[ParsedType] | None,
                    max_tokens: int | None,
                    temperature: float
            ) -> Completion[ParsedType] | AsyncIterable[Completion[ParsedType]]:
                return simple_completion

        return ConcreteAPIModel

    def test_model_initialization(self, mocker: Any, concrete_api_model) -> None:
        model = concrete_api_model("test-model", "test-key", "test-url")

        assert model.model_name == "test-model"
        assert model.api_key == "test-key"
        assert model.base_url == "test-url"

    def test_temperature_validation(self, concrete_api_model) -> None:
        model = concrete_api_model("test-model")

        model.temperature = 0.5
        assert model.temperature == 0.5

        with pytest.raises(ValueError, match="Temperature must be positive!"):
            model.temperature = -1

        model.temperature = None
        assert model.temperature is None

    def test_max_tokens_validation(self, concrete_api_model) -> None:
        model = concrete_api_model("test-model")

        model.max_tokens = 100
        assert model.max_tokens == 100

        with pytest.raises(ValueError, match="Max tokens must be positive!"):
            model.max_tokens = 0

        model.max_tokens = None
        assert model.max_tokens is None

    def test_tool_registration(self, concrete_api_model) -> None:
        model = concrete_api_model("test-model")

        @Tool
        def test_tool(x: int) -> int:
            return x

        model.register_tool(test_tool)
        assert "test_tool" in model.tools

        with pytest.raises(AssertionError):
            model.register_tool(test_tool)  # Duplicate registration

    def test_tool_removal(self, concrete_api_model) -> None:
        model = concrete_api_model("test-model")

        @Tool
        def test_tool(x: int) -> int:
            return x

        model.register_tool(test_tool)
        assert "test_tool" in model.tools

        model.remove_tools("test_tool")
        assert "test_tool" not in model.tools

        with pytest.raises(ValueError):
            model.remove_tools("non_existent_tool")

    def test_invoke(self, simple_completion, concrete_api_model) -> None:
        model = concrete_api_model("test-model")
        messages = [
            UserMessage(content="Hello"),
            SystemMessage(content="System message")
        ]

        response = model.invoke(messages)
        assert response == simple_completion

    @pytest.mark.asyncio
    async def test_async_invoke(self, simple_completion, concrete_api_model) -> None:
        model = concrete_api_model("test-model")
        messages = [
            UserMessage(content="Hello"),
            SystemMessage(content="System message")
        ]

        async_response = await model.async_invoke(messages)
        assert async_response == simple_completion
