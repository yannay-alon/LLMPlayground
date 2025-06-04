from typing import Any

import pytest

from components.tools import Tool


class TestTool:
    def test_tool_from_simple_function(self):
        def sample_function(arg1: str, arg2: int = 42) -> str:
            """Sample function description
            
            Args:
                arg1: First argument description
                arg2: Second argument description
            """
            return f"{arg1} {arg2}"

        tool = Tool.from_function(sample_function)

        assert tool.name == "sample_function"
        assert "Sample function description" in tool.description
        assert len(tool.arguments) == 2

        # Test the first argument
        assert tool.arguments[0].name == "arg1"
        assert tool.arguments[0].annotation == str
        assert "First argument description" in tool.arguments[0].description
        assert tool.arguments[0].required == True

        # Test the second argument
        assert tool.arguments[1].name == "arg2"
        assert tool.arguments[1].annotation == int
        assert "Second argument description" in tool.arguments[1].description
        assert tool.arguments[1].required == False

        # Test the function call
        result = tool("test", 123)
        assert result == "test 123"

    def test_tool_from_class(self):
        class SampleTool:
            """Class level description"""

            def __call__(self, input_text: str, multiplier: int = 1) -> str:
                """Method description
                
                Args:
                    input_text: Input text description
                    multiplier: Multiplier description
                """
                return input_text * multiplier

        tool = Tool.from_class(SampleTool)

        assert tool.name == "SampleTool"
        assert "Class level description" in tool.description
        assert len(tool.arguments) == 2

        # Test arguments
        assert tool.arguments[0].name == "input_text"
        assert tool.arguments[0].required == True
        assert tool.arguments[1].name == "multiplier"
        assert tool.arguments[1].required == False

        # Test function call
        result = tool("test", 2)
        assert result == "testtest"

    def test_tool_as_decorator_function(self):
        @Tool
        def decorated_function(x: int) -> int:
            """Function with decorator"""
            return x * 2

        assert isinstance(decorated_function, Tool)
        assert decorated_function.name == "decorated_function"
        assert len(decorated_function.arguments) == 1
        assert decorated_function(5) == 10

    def test_tool_as_decorator_class(self):
        @Tool
        class DecoratedClass:
            """Class with decorator"""

            def __call__(self, x: int) -> int:
                return x * 3

        assert isinstance(DecoratedClass, Tool)
        assert DecoratedClass.name == "DecoratedClass"
        assert len(DecoratedClass.arguments) == 1
        assert DecoratedClass(5) == 15

    def test_tool_missing_docstring(self):
        def no_doc_function(x: int) -> int:
            return x

        tool = Tool.from_function(no_doc_function)
        assert tool.description == ""  # or whatever default you expect
        assert len(tool.arguments) == 1

    def test_tool_missing_type_hints(self):
        def untyped_function(x, y=None):
            """Function without type hints"""
            return x

        tool = Tool.from_function(untyped_function)
        assert tool.arguments[0].annotation == Any
        assert tool.arguments[1].annotation == Any

    def test_tool_copy(self):
        @Tool
        def original_tool(x: int) -> int:
            return x

        copied_tool = Tool(tool=original_tool)

        assert copied_tool is not original_tool
        assert copied_tool.name == original_tool.name
        assert copied_tool.arguments == original_tool.arguments
        assert copied_tool(5) == original_tool(5)

    @pytest.mark.parametrize("invalid_input", [
        42,  # number
        "string",  # string
        [],  # list
        None,  # None
    ])
    def test_tool_invalid_input(self, invalid_input):
        with pytest.raises((ValueError, AssertionError)):
            Tool(invalid_input)
