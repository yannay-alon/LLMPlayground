import json
from collections import defaultdict
from typing import AsyncIterable, Iterable

from cohere import ClientV2, AsyncClientV2, ToolCallV2
from cohere.types import (
    ChatResponse,
    StreamedChatResponseV2,
    ChatFinishReason,
    ChatMessages,
    ToolV2,
    Document as CohereDocument,
    JsonObjectResponseFormatV2,
    ToolV2Function
)
from pydantic import BaseModel, ConfigDict

from components.documents import Document
from components.messages import BaseMessage
from components.responses import Completion, Usage, Choice, ToolCall
from components.responses.choice import ParsedType, FinishReason
from components.tools import Tool
from models.generation.remote.api_model import APIModel
from models.utilities.json_parsing import parse_json
from utilities.pydantic_utilities import make_strict_model, clear_empty_fields


class CohereCompatibleArguments(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: ChatMessages
    tools: list[ToolV2] | None = None
    documents: list[CohereDocument] | None = None
    response_format: JsonObjectResponseFormatV2 | None = None


class CohereModel(APIModel):
    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
    ):
        super().__init__(model_name, api_key, base_url)

        self.client = ClientV2(
            api_key=api_key,
            base_url=base_url,
        )
        self.async_client = AsyncClientV2(
            api_key=api_key,
            base_url=base_url,
        )

    def _invoke(self, messages: list[BaseMessage], stream: bool, tools: dict[str, Tool] | None,
                documents: list[Document] | None, response_format: type[ParsedType] | None, max_tokens: int | None,
                temperature: float) -> Completion[ParsedType] | Iterable[Completion[ParsedType]]:
        arguments = self._prepare_arguments(
            messages=messages,
            tools=tools,
            documents=documents,
            response_format=response_format,
        )
        non_empty_arguments = clear_empty_fields(arguments, max_tokens=max_tokens)
        if not stream:
            response = self.client.chat(
                model=self.model_name,
                temperature=temperature,
                **non_empty_arguments
            )
            completions = Completion(
                choices=[
                    self._build_choice(response, response_format, tools)
                ],
                usage=Usage(
                    input_tokens=int(response.usage.tokens.input_tokens),
                    output_tokens=int(response.usage.tokens.output_tokens)
                ),
            )
            return completions
        else:
            response = self.client.chat_stream(
                model=self.model_name,
                temperature=temperature,
                **non_empty_arguments
            )

            def streaming_generator() -> Iterable[Completion]:
                for chunk in response:
                    if chunk.type in ["content-start", "content-delta"]:
                        yield Completion(choices=[
                            self._build_choice(
                                chunk,
                                response_format=None,  # No support for structured output in streaming mode
                                tools=tools,
                            )
                        ], usage=None)

            return streaming_generator()

    async def _async_invoke(self, messages: list[BaseMessage], stream: bool, tools: dict[str, Tool] | None,
                            documents: list[Document] | None, response_format: type[ParsedType] | None,
                            max_tokens: int | None, temperature: float) -> Completion[ParsedType] | AsyncIterable[
        Completion[ParsedType]]:
        arguments = self._prepare_arguments(
            messages=messages,
            tools=tools,
            documents=documents,
            response_format=response_format
        )
        non_empty_arguments = clear_empty_fields(arguments, max_tokens=max_tokens)
        if not stream:
            response = await self.async_client.chat(
                model=self.model_name,
                temperature=temperature,
                **non_empty_arguments
            )
            completions = Completion(
                choices=[
                    self._build_choice(response, response_format, tools)
                ],
                usage=Usage(
                    input_tokens=int(response.usage.tokens.input_tokens),
                    output_tokens=int(response.usage.tokens.output_tokens)
                ),
            )
            return completions
        else:
            response = self.async_client.chat_stream(
                model=self.model_name,
                temperature=temperature,
                **non_empty_arguments
            )

            async def streaming_generator() -> AsyncIterable[Completion]:
                async for chunk in response:
                    if chunk.type in ["content-start", "content-delta"]:
                        yield Completion(choices=[
                            self._build_choice(
                                chunk,
                                response_format=None,  # No support for structured output in streaming mode
                                tools=tools,
                            )
                        ], usage=None)

            return streaming_generator()

    @staticmethod
    def _get_matching_finish_reason(finish_reason: ChatFinishReason) -> FinishReason:
        match finish_reason:
            case "COMPLETE" | "STOP_SEQUENCE":
                return FinishReason.STOP
            case "MAX_TOKENS":
                return FinishReason.LENGTH
            case "TOOL_CALL":
                return FinishReason.TOOL_CALLS
            case _:
                raise ValueError(f"Unsupported finish reason: {finish_reason}")

    @staticmethod
    def _build_choice(
            response: ChatResponse | StreamedChatResponseV2,
            response_format: type[ParsedType] | None,
            tools: dict[str, Tool] | None = None,
    ) -> Choice[ParsedType]:
        if isinstance(response, ChatResponse):
            message = response.message
            if message.content is not None:
                text = message.content[0].text
            elif message.tool_plan is not None:
                text = message.tool_plan
            else:
                raise ValueError("Message content is empty and no tool plan is provided.")
        else:
            message = response.delta.message
            text = message.content.text

        finish_reason = CohereModel._get_matching_finish_reason(response.finish_reason)

        tool_mapping = defaultdict()
        if tools is None:
            tools = {}
        for tool_name, tool in tools.items():
            tool_mapping[tool_name] = tool

        tool_calls = []
        for tool_call in message.tool_calls or []:
            tool_call: ToolCallV2
            chosen_tool = tool_mapping.get(tool_call.function.name)
            try:
                arguments = json.loads(tool_call.function.arguments)
            except AttributeError:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    identifier=tool_call.id,
                    tool=chosen_tool,
                    arguments_values=arguments,
                )
            )

        parsed_message = None
        if response_format is not None:
            parsed_message = parse_json(message.content[0].text, response_format)

        return Choice(
            content=text,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            parsed=parsed_message,
        )

    def _prepare_arguments(
            self,
            messages: list[BaseMessage],
            tools: dict[str, Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None
    ) -> CohereCompatibleArguments:
        dumped_messages = [message.model_dump(by_alias=True) for message in messages]
        cohere_compatible_tools = self._process_tools(tools)
        cohere_compatible_documents = self._process_documents(documents)
        cohere_compatible_response_format = self._process_response_format(response_format)

        return CohereCompatibleArguments(
            messages=dumped_messages,
            tools=cohere_compatible_tools,
            documents=cohere_compatible_documents,
            response_format=cohere_compatible_response_format
        )

    def _process_tools(
            self, tools: dict[str, Tool] | None
    ) -> list[ToolV2] | None:
        if tools is None:
            return None
        cohere_compatible_tools = []
        for tool_name, tool in tools.items():
            cohere_compatible_tools.append(
                ToolV2(
                    function=ToolV2Function(
                        name=tool_name,
                        description=tool.description,
                        parameters={
                            "type": "object",
                            "properties": {
                                argument.name: {
                                    "type": argument.type,
                                    "description": argument.description,
                                } for argument in tool.arguments
                            },
                        },
                    ),
                    type="function",
                )
            )
        return cohere_compatible_tools

    def _process_documents(
            self, documents: list[Document] | None
    ) -> list[CohereDocument] | None:
        if documents is None:
            return None
        cohere_compatible_documents = [
            CohereDocument(
                id=document.identifier,
                data={"text": document.content},
            )
            for document in documents
        ]
        return cohere_compatible_documents

    def _process_response_format(
            self, response_format: type[BaseModel] | None
    ) -> JsonObjectResponseFormatV2 | None:
        if response_format is None:
            return None
        strict_json_schema = make_strict_model(response_format)
        cohere_compatible_json_schema = JsonObjectResponseFormatV2(
            json_schema=strict_json_schema
        )
        return cohere_compatible_json_schema
