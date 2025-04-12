import json
from collections import defaultdict
from typing import Any, cast, Iterable, AsyncIterable

import httpx
from openai import Client, AsyncClient
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChoiceChunk
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel

from components.documents import Document
from components.messages import BaseMessage
from components.responses import Completion, Choice, ToolCall, Usage
from components.responses.choice import FinishReason
from components.tools import Tool
from models.api_model import APIModel
from models.utilities.json_parsing import parse_json


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

        sync_client_arguments = sync_client_arguments or dict(
            http_client=httpx.Client()
        )
        self.client = Client(
            api_key=api_key,
            base_url=base_url,
            **sync_client_arguments
        )

        async_client_arguments = async_client_arguments or dict(
            http_client=httpx.AsyncClient()
        )
        self.async_client = AsyncClient(
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

    @staticmethod
    def _add_documents_to_messages(
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

        messages_with_documents = history + [BaseMessage(role="developer", content=formatted_documents), last_message]
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
    ) -> Completion | Iterable[Completion]:
        (
            dumped_messages,
            open_ai_compatible_tools,
            open_ai_compatible_response_format
        ) = self._prepare_arguments(messages, tools, documents, response_format)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=dumped_messages,
            stream=stream,
            tools=open_ai_compatible_tools,
            response_format=open_ai_compatible_response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not stream:
            choices = [
                self._build_choice(
                    choice,
                    response_format,
                    tools=tools,
                )
                for choice in response.choices
            ]
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return Completion(choices=choices, usage=usage)
        else:
            def streaming_generator() -> Iterable[Completion]:
                for chunk in response:
                    choices = [
                        self._build_choice(
                            choice,
                            response_format=None,  # No support for structured output in streaming mode
                            tools=tools,
                        )
                        for choice in chunk.choices
                    ]
                    yield Completion(choices=choices, usage=None)

            return streaming_generator()

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
        (
            dumped_messages,
            open_ai_compatible_tools,
            open_ai_compatible_response_format
        ) = self._prepare_arguments(messages, tools, documents, response_format)

        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=dumped_messages,
            stream=stream,
            tools=open_ai_compatible_tools,
            response_format=open_ai_compatible_response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if not stream:
            choices = [
                self._build_choice(
                    choice,
                    response_format=response_format,
                    tools=tools,
                )
                for choice in response.choices
            ]
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return Completion(choices=choices, usage=usage)
        else:
            async def streaming_generator() -> AsyncIterable[Completion]:
                async for chunk in response:
                    choices = [
                        self._build_choice(
                            choice,
                            response_format=None,  # No support for structured output in streaming mode
                            tools=tools,
                        )
                        for choice in chunk.choices
                    ]
                    yield Completion(choices=choices, usage=None)

            return streaming_generator()

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
        (
            dumped_messages,
            open_ai_compatible_tools,
            _
        ) = self._prepare_arguments(messages, tools, documents, response_format)
        additional_tokenization_arguments = {}
        return dumped_messages, open_ai_compatible_tools, None, additional_tokenization_arguments

    @staticmethod
    def _build_choice(
            choice: OpenAIChoice | OpenAIChoiceChunk,
            response_format: type[BaseModel] | None = None,
            tools: list[Tool] | None = None
    ) -> Choice:
        if choice.finish_reason is None:
            finish_reason = FinishReason.NONE
        else:
            finish_reason = FinishReason(choice.finish_reason)

        if isinstance(choice, OpenAIChoiceChunk):
            message = choice.delta
        else:
            message = choice.message

        tool_mapping = defaultdict()
        for tool in tools or []:
            tool_mapping[tool.name] = tool

        parsed_message = None
        if response_format is not None:
            parsed_message = parse_json(message.content, response_format)

        return Choice(
            content=message.content or "",
            finish_reason=finish_reason,
            tool_calls=[
                ToolCall(
                    identifier=tool_call.id,
                    tool=tool_mapping[tool_call.function.name],
                    arguments_values=json.loads(tool_call.function.arguments)
                )
                for tool_call in message.tool_calls or []
            ],
            parsed=parsed_message,
        )
