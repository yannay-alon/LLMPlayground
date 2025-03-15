from abc import ABC, abstractmethod
from typing import overload, Literal, Protocol, TypeVar, Any

from openai import Stream, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from documents import Document
from messages import BaseMessage, MessageFactory
from tools import Tool
from utilities import get_tokenizer

MESSAGE_TYPE = TypeVar("MESSAGE_TYPE", bound=BaseMessage)


class Client(Protocol):
    def __call__(
            self,
            messages: list[BaseMessage],
            stream: bool = False,
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            max_tokens: int | None = None,
            temperature: float = 1,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        pass


class AsyncClient(Protocol):
    async def __call__(
            self,
            messages: list[BaseMessage],
            stream: bool = False,
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            max_tokens: int | None = None,
            temperature: float = 1,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        pass


class APIModel(ABC):
    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
            **client_arguments
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

        self.client = self._initiate_client(api_key, base_url, **client_arguments)
        self.async_client = self._initiate_async_client(api_key, base_url, **client_arguments)

        self.tokenizer = get_tokenizer(model_name)

    # <editor-fold desc="Synchronous">

    @overload
    def invoke(
            self,
            messages: list[BaseMessage] | list[dict[str, str]],
            stream: Literal[False] = False,
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            max_tokens: int | None = None,
            temperature: float = 1,
    ) -> ChatCompletion:
        ...

    @overload
    def invoke(
            self,
            messages: list[BaseMessage] | list[dict[str, str]],
            stream: Literal[True] = True,
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            max_tokens: int | None = None,
            temperature: float = 1,
    ) -> Stream[ChatCompletionChunk]:
        ...

    def invoke(
            self,
            messages: list[BaseMessage | dict[str, str]],
            stream: bool = False,
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            max_tokens: int | None = None,
            temperature: float = 1,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        loaded_messages = self._load_messages(messages)

        return self.client(
            messages=loaded_messages,
            stream=stream,
            tools=tools,
            documents=documents,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature
        )

    # </editor-fold>

    # <editor-fold desc="Asynchronous">

    @overload
    async def async_invoke(
            self,
            messages: list[BaseMessage] | list[dict[str, str]],
            stream: Literal[False] = False,
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            max_tokens: int | None = None,
            temperature: float = 1,
    ) -> ChatCompletion:
        ...

    @overload
    async def async_invoke(
            self,
            messages: list[BaseMessage] | list[dict[str, str]],
            stream: Literal[True] = True,
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            max_tokens: int | None = None,
            temperature: float = 1,
    ) -> AsyncStream[ChatCompletionChunk]:
        ...

    async def async_invoke(
            self,
            messages: list[BaseMessage | dict[str, str]],
            stream: bool = False,
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            max_tokens: int | None = None,
            temperature: float = 1,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        loaded_messages = self._load_messages(messages)
        return await self.async_client(
            messages=loaded_messages,
            stream=stream,
            tools=tools,
            documents=documents,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature
        )

    # </editor-fold>

    # <editor-fold desc="Prompt Creation">

    @overload
    def create_prompt(
            self,
            messages: list[BaseMessage | dict[str, str]],
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            tokenize: Literal[False] = False,
            continue_final_message: bool = False,
            **kwargs
    ) -> str:
        ...

    @overload
    def create_prompt(
            self,
            messages: list[BaseMessage | dict[str, str]],
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            tokenize: Literal[True] = True,
            continue_final_message: bool = False,
            **kwargs
    ) -> list[int]:
        ...

    def create_prompt(
            self,
            messages: list[BaseMessage | dict[str, str]],
            tools: list[Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[BaseModel] | None = None,
            *,
            tokenize: bool = False,
            continue_final_message: bool = False,
            **kwargs
    ) -> str | list[int]:
        loaded_messages = self._load_messages(messages)

        processed_messages, processed_tools, processed_documents, additional_tokenization_arguments = (
            self._process_arguments_for_prompt_creation(
                loaded_messages,
                tools,
                documents,
                response_format
            )
        )

        tokenization_arguments = {
            "conversation": processed_messages,
            "tools": processed_tools,
            "documents": processed_documents,
            **additional_tokenization_arguments
        }

        non_empty_tokenization_arguments = {
            key: value
            for key, value in tokenization_arguments.items()
            if value is not None
        }

        return self.tokenizer.apply_chat_template(
            **non_empty_tokenization_arguments,
            tokenize=tokenize,
            continue_final_message=continue_final_message,
            **kwargs
        )

    # </editor-fold>

    @abstractmethod
    def _initiate_client(
            self,
            api_key: str,
            base_url: str,
            **kwargs
    ) -> Client:
        pass

    @abstractmethod
    def _initiate_async_client(
            self,
            api_key: str,
            base_url: str,
            **kwargs
    ) -> AsyncClient:
        pass

    @staticmethod
    def _load_messages(messages: list[MESSAGE_TYPE | dict[str, str]]) -> list[MESSAGE_TYPE]:
        loaded_messages = []
        for message in messages:
            if isinstance(message, BaseMessage):
                processed_message = message
            else:
                processed_message = MessageFactory.create_message(**message)
            loaded_messages.append(processed_message)

        return loaded_messages

    def _process_arguments_for_prompt_creation(
            self,
            messages: list[BaseMessage],
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None
    ) -> tuple[list[dict[str, str]], list[dict[str, str]] | None, list[dict[str, Any]] | None, dict[str, Any]]:
        dumped_messages = [
            message.model_dump(by_alias=True) for message in messages
        ]

        dumped_tools = [
            tools.model_dump(by_alias=True) for tools in tools
        ] if tools is not None else None

        dumped_documents = [
            document.model_dump(by_alias=True) for document in documents
        ] if documents is not None else None

        additional_tokenization_arguments = {}
        return dumped_messages, dumped_tools, dumped_documents, additional_tokenization_arguments
