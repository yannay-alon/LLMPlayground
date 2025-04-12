from abc import ABC, abstractmethod
from typing import overload, Literal, Any, Iterable, AsyncIterable

from pydantic import BaseModel

from components.documents import Document
from components.messages import BaseMessage, MessageFactory
from models.utilities import get_tokenizer
from components.responses import Completion
from components.tools import Tool


class APIModel(ABC):
    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

        try:
            self.tokenizer = get_tokenizer(model_name)
        except (ValueError, OSError):
            self.tokenizer = None

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
    ) -> Completion:
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
    ) -> Iterable[Completion]:
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
    ) -> Completion | Iterable[Completion]:
        loaded_messages = self._load_messages(messages)

        return self._invoke(
            messages=loaded_messages,
            stream=stream,
            tools=tools,
            documents=documents,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature
        )

    @abstractmethod
    def _invoke(
            self,
            messages: list[BaseMessage],
            stream: bool,
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None,
            max_tokens: int | None,
            temperature: float
    ) -> Completion | Iterable[Completion]:
        pass

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
    ) -> Completion:
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
    ) -> AsyncIterable[Completion]:
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
    ) -> Completion | AsyncIterable[Completion]:
        loaded_messages = self._load_messages(messages)

        return await self._async_invoke(
            messages=loaded_messages,
            stream=stream,
            tools=tools,
            documents=documents,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature
        )

    @abstractmethod
    async def _async_invoke(
            self,
            messages: list[BaseMessage],
            stream: bool,
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None,
            max_tokens: int | None,
            temperature: float
    ) -> Completion | AsyncIterable[Completion]:
        pass

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
            chat_template: str | None = None,
            **kwargs
    ) -> str | list[int]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not available for this model. Cannot create prompt.")

        loaded_messages = self._load_messages(messages)

        (
            processed_messages,
            processed_tools,
            processed_documents,
            additional_tokenization_arguments
        ) = self._process_arguments_for_prompt_creation(
            loaded_messages,
            tools,
            documents,
            response_format
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
            chat_template=chat_template,
            **kwargs
        )

    # </editor-fold>

    @staticmethod
    def _load_messages(messages: list[BaseMessage | dict[str, str]]) -> list[BaseMessage]:
        loaded_messages = []
        for message in messages:
            if isinstance(message, BaseMessage):
                processed_message = message
            else:
                processed_message = MessageFactory.create_message(**message)
            loaded_messages.append(processed_message)

        return loaded_messages

    @abstractmethod
    def _process_arguments_for_prompt_creation(
            self,
            messages: list[BaseMessage],
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None
    ) -> tuple[list[dict[str, str]], list[dict[str, str]] | None, list[dict[str, Any]] | None, dict[str, Any]]:
        """
        Process the arguments before applying the tokenizer's apply_chat_message method

        :param messages: The messages to process
        :param tools: The tools to process
        :param documents: The documents to process
        :param response_format: The output response format to process
        :return: Processed messages, processed tools, processed documents, and any additional tokenization arguments.
        If the tokenizer does not support any of the arguments, return None for that argument.
        """
        pass
