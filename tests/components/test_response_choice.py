from pydantic import BaseModel

from components.responses import Choice
from components.responses.tool_call import ToolCall
from components.tools.tools import Tool
from components.responses.choice import FinishReason


class TestChoice:
    class SampleResponse(BaseModel):
        text: str

    def test_choice_basic(self):
        choice = Choice(
            content="Test content",
            finish_reason=FinishReason.STOP
        )
        assert choice.content == "Test content"
        assert choice.finish_reason == FinishReason.STOP
        assert choice.tool_calls is None
        assert choice.parsed is None

    def test_choice_with_tool_calls(self):
        tool_call = ToolCall(
            identifier="call_123",
            tool=Tool(lambda arg: ""),
            arguments_values={"arg": "value"}
        )
        choice = Choice(
            content="Test with tool",
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=[tool_call]
        )
        assert choice.tool_calls == [tool_call]
        assert choice.finish_reason == FinishReason.TOOL_CALLS


    def test_choice_with_parsed_response(self):
        parsed = TestChoice.SampleResponse(text="parsed content")
        choice = Choice(
            content="Test with parsing",
            finish_reason=FinishReason.STOP,
            parsed=parsed
        )
        assert choice.parsed == parsed
