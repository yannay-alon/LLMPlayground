from typing import Any, cast

from openai import AsyncStream, Stream, OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam, ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel

from documents import Document
from messages import BaseMessage, UserMessage
from models.api_model import APIModel
from tools import Tool


class OpenAIModel(APIModel):
    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
            strict_mode: bool = True,
            sync_client_arguments: dict[str, Any] | None = None,
            async_client_arguments: dict[str, Any] | None = None,
    ):
        super().__init__(model_name, api_key, base_url)
        self.strict_mode = strict_mode

        sync_client_arguments = sync_client_arguments or {}
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **sync_client_arguments
        )

        async_client_arguments = async_client_arguments or {}
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **async_client_arguments
        )

    def _process_tools(
            self,
            tools: list[Tool] | None,
    ) -> list[ChatCompletionToolParam] | None:
        if tools is None:
            return None

        open_ai_compatible_tools = []
        for tool in tools:
            open_ai_compatible_tools.append(
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters={
                            "type": "object",
                            "properties": {
                                argument.name: {
                                    "type": argument.type,
                                    "description": argument.description,
                                } for argument in tool.arguments
                            },
                            "required": [
                                argument.name for argument in tool.arguments if argument.required or self.strict_mode
                            ],
                            "additionalProperties": False
                        },
                    ),
                    type="function",
                )
            )
        return open_ai_compatible_tools

    def _add_documents_to_messages(
            self,
            messages: list[BaseMessage],
            documents: list[Document] | None
    ) -> list[BaseMessage]:
        if documents is None:
            return messages

        formatted_documents = "\n\n".join(
            f"Document: {document_index}\n{document}"
            for document_index, document in enumerate(documents)
        )
        formatted_documents = f"Documents:\n{formatted_documents}"

        *history, last_message = messages

        messages_with_documents = history + [UserMessage(content=formatted_documents), last_message]
        return messages_with_documents

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
        dumped_messages, open_ai_compatible_tools, open_ai_compatible_response_format = (
            self._prepare_arguments(messages, tools, documents, response_format)
        )

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=dumped_messages,
            stream=stream,
            tools=open_ai_compatible_tools,
            response_format=open_ai_compatible_response_format,
            max_tokens=max_tokens,
            temperature=temperature,
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
        dumped_messages, open_ai_compatible_tools, open_ai_compatible_response_format = self._prepare_arguments(
            messages,
            tools,
            documents,
            response_format
        )

        return await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=dumped_messages,
            stream=stream,
            tools=open_ai_compatible_tools,
            response_format=open_ai_compatible_response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _prepare_arguments(
            self,
            messages: list[BaseMessage],
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None
    ) -> tuple[list[dict[str, str]], list[ChatCompletionToolParam] | None, ResponseFormat | None]:
        messages_with_documents = self._add_documents_to_messages(messages, documents)
        dumped_messages = [message.model_dump(by_alias=True) for message in messages_with_documents]
        open_ai_compatible_tools = self._process_tools(tools)

        if response_format is None:
            open_ai_compatible_response_format = None
        else:
            open_ai_compatible_response_format = cast(
                ResponseFormat, dict(
                    type="json_schema",
                    json_schema=response_format.model_json_schema(by_alias=True)
                )
            )

        return dumped_messages, open_ai_compatible_tools, open_ai_compatible_response_format

    def _process_arguments_for_prompt_creation(
            self,
            messages: list[BaseMessage],
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None
    ) -> tuple[list[dict[str, str]], list[dict[str, str]] | None, list[dict[str, Any]] | None, dict[str, Any]]:
        dumped_messages, open_ai_compatible_tools, _ = self._prepare_arguments(
            messages,
            tools,
            documents,
            response_format
        )
        additional_tokenization_arguments = {}
        return dumped_messages, open_ai_compatible_tools, None, additional_tokenization_arguments
