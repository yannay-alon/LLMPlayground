from typing import Any

from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from pydantic import BaseModel

from documents import Document
from messages import BaseMessage
from models.api_model import APIModel
from tools import Tool


# TODO: Implement all the methods in this class
class CohereModel(APIModel):

    def _invoke(self, messages: list[BaseMessage], stream: bool, tools: list[Tool] | None,
                documents: list[Document] | None, response_format: type[BaseModel] | None, max_tokens: int | None,
                temperature: float) -> ChatCompletion | Stream[ChatCompletionChunk]:
        pass

    async def _async_invoke(self, messages: list[BaseMessage], stream: bool, tools: list[Tool] | None,
                            documents: list[Document] | None, response_format: type[BaseModel] | None,
                            max_tokens: int | None, temperature: float) -> ChatCompletion | AsyncStream[
        ChatCompletionChunk]:
        pass

    def _process_arguments_for_prompt_creation(self, messages: list[BaseMessage], tools: list[Tool] | None,
                                               documents: list[Document] | None,
                                               response_format: type[BaseModel] | None) -> tuple[
        list[dict[str, str]], list[dict[str, str]] | None, list[dict[str, Any]] | None, dict[str, Any]]:
        pass