from typing import AsyncIterator

import httpx
import pytest
from openai import Client, AsyncClient
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChoiceChunk

from components.messages import UserMessage
from components.responses import Completion
from components.tools import Tool
from models.generation.remote.openai_model import OpenAIModel


class TestOpenAIModel:
    @pytest.fixture
    def mock_client(self, mocker) -> Client:
        client = mocker.Mock()
        client.http_client = httpx.Client()
        mocker.patch("models.generation.openai_model.Client", return_value=client)
        return client

    @pytest.fixture
    def mock_async_client(self, mocker) -> AsyncClient:
        client = mocker.AsyncMock()
        client.http_client = httpx.AsyncClient()
        mocker.patch("models.generation.openai_model.AsyncClient", return_value=client)
        return client

    @pytest.fixture
    def tool_example(self) -> Tool:
        @Tool
        def tool_example(x: int, optional_arg: str = "default") -> int:
            """Test tool description"""
            return x

        return tool_example

    @pytest.fixture
    def model(self, mock_client: Client, mock_async_client: AsyncClient) -> OpenAIModel:
        return OpenAIModel(
            model_name="test-model",
            api_key="test-key",
            base_url="https://test.openai.com/v1"
        )

    def test_model_initialization(self, mock_client: Client, mock_async_client: AsyncClient):
        model = OpenAIModel("test-model", "test-key", "test-url")
        assert model.model_name == "test-model"
        assert model.api_key == "test-key"
        assert model.base_url == "test-url"
        assert model.strict_mode == True
        assert isinstance(model.client.http_client, httpx.Client)
        assert isinstance(model.async_client.http_client, httpx.AsyncClient)

    def test_invoke_non_streaming(self, model, mock_client, mocker):
        messages = [UserMessage(content="Test")]
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock(
            message=mocker.Mock(content="Test response", tool_calls=None),
            finish_reason="stop"
        )]
        mock_response.usage = mocker.Mock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create.return_value = mock_response

        result = model.invoke(messages)

        assert isinstance(result, Completion)
        assert len(result.choices) == 1
        assert result.choices[0].content == "Test response"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_async_invoke_streaming(self, model, mock_async_client, mocker):
        messages = [UserMessage(content="Test")]

        async def mock_stream():
            for chunk_index in range(5):
                mock_delta = mocker.Mock()
                mock_delta.content = f"chunk {chunk_index}"
                mock_delta.tool_calls = None

                mock_choice = mocker.Mock(spec=OpenAIChoiceChunk)
                mock_choice.delta = mock_delta
                mock_choice.finish_reason = None

                chunk = mocker.Mock()
                chunk.choices = [mock_choice]
                yield chunk

        mock_async_client.chat.completions.create.return_value = mock_stream()

        response = await model.async_invoke(messages, stream=True)
        assert isinstance(response, AsyncIterator)

        chunks = []
        chunk_index = 0
        async for chunk in response:
            chunks.append(chunk)
            assert isinstance(chunk, Completion)
            assert chunk.choices[0].content == f"chunk {chunk_index}"
            chunk_index += 1

        assert len(chunks) == 5
