import pytest
from utilities.string_formatting import indent_code


def test_indent_code_with_explicit_length():
    indentation_length = 4
    indentation = " " * indentation_length

    text = "first line\nsecond line\nthird line"
    expected = f"first line\n{indentation}second line\n{indentation}third line"
    assert indent_code(text, indent_length=indentation_length) == expected


def test_indent_code_empty_lines():
    text = "first line\n\nsecond line"
    expected = "first line\n\n    second line"
    assert indent_code(text, indent_length=4) == expected


def test_indent_code_non_string_input():
    number = 12345
    expected = "12345"
    assert indent_code(number, indent_length=4) == expected


def test_indent_code_single_line():
    text = "single line"
    expected = "single line"
    assert indent_code(text, indent_length=4) == expected


@pytest.mark.parametrize("indent_length", [0, 2, 5, 8])
def test_indent_code_different_lengths(indent_length: int):
    expected_indent = " " * indent_length
    text = "first\nsecond"
    expected = f"first\n{expected_indent}second"
    assert indent_code(text, indent_length=indent_length) == expected


def test_indent_code_whitespace_lines():
    text = "first\n    \nsecond"
    expected = "first\n    \n    second"
    assert indent_code(text, indent_length=4) == expected


def test_indent_code_empty_input():
    assert indent_code("", indent_length=4) == ""
    assert indent_code("\n", indent_length=4) == "\n"


def test_indent_code_mixed_line_endings():
    text = "first\rsecond\r\nthird\nlast"
    expected = "first\r    second\r\n    third\n    last"
    assert indent_code(text, indent_length=4) == expected


def test_indent_code_tabs_and_spaces():
    text = "first\n\tsecond\n    third"
    expected = "first\n    \tsecond\n        third"
    assert indent_code(text, indent_length=4) == expected


def test_indent_code_auto_indent():
    def wrapper():
        return indent_code("first\nsecond")

    result = wrapper()
    expected = "first\n        second"
    assert result == expected


def test_indent_code_trailing_newlines():
    text = "first\nsecond\n\n"
    expected = "first\n    second\n\n"
    assert indent_code(text, indent_length=4) == expected
