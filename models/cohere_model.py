from typing import Any

from cohere import ClientV2 as Client, AsyncClientV2 as AsyncClient
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from pydantic import BaseModel

from documents import Document
from messages import BaseMessage
from models.api_model import APIModel
from tools import Tool


class CohereModel(APIModel):
    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
            sync_client_arguments: dict[str, Any] | None = None,
            async_client_arguments: dict[str, Any] | None = None,

    ):
        super().__init__(model_name, api_key, base_url)

        sync_client_arguments = sync_client_arguments or {}
        self.client = Client(
            api_key=self.api_key,
            base_url=self.base_url,
            **sync_client_arguments
        )

        async_client_arguments = async_client_arguments or {}
        self.async_client = AsyncClient(
            api_key=self.api_key,
            base_url=self.base_url,
            **async_client_arguments
        )

    def _invoke(
            self,
            messages: list[BaseMessage],
            stream: bool,
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None,
            max_tokens: int | None,
            temperature: float
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:

        if stream:
            chat_responses = self.client.chat_stream(
                model=self.model_name,
                messages=messages,
                tools=tools,
                documents=documents,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            chat_response = self.client.chat(
                model=self.model_name,
                messages=messages,
                tools=tools,
                documents=documents,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature
            )

    async def _async_invoke(
            self,
            messages: list[BaseMessage],
            stream: bool,
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None,
            max_tokens: int | None,
            temperature: float
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        pass

    def _process_arguments_for_prompt_creation(
            self,
            messages: list[BaseMessage],
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None
    ) -> tuple[list[dict[str, str]], list[dict[str, str]] | None, list[dict[str, Any]] | None, dict[str, Any]]:
        pass
