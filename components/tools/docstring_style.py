import re
from griffe import Parser as DocstringStyle
from typing import NamedTuple


class DocstringStylePattern(NamedTuple):
    pattern: str
    replacements: list[str]
    style: DocstringStyle


# See https://github.com/mkdocstrings/griffe/issues/329#issuecomment-2425017804
docstring_style_patterns: list[DocstringStylePattern] = [
    DocstringStylePattern(
        pattern=r"\n[ \t]*:{0}([ \t]+\w+)*:([ \t]+.+)?\n",
        replacements=[
            "param",
            "parameter",
            "arg",
            "argument",
            "key",
            "keyword",
            "type",
            "var",
            "ivar",
            "cvar",
            "vartype",
            "returns",
            "return",
            "rtype",
            "raises",
            "raise",
            "except",
            "exception",
        ],
        style=DocstringStyle.sphinx,
    ),
    DocstringStylePattern(
        pattern=r"\n[ \t]*{0}:([ \t]+.+)?\n[ \t]+.+",
        replacements=[
            "args",
            "arguments",
            "params",
            "parameters",
            "keyword args",
            "keyword arguments",
            "other args",
            "other arguments",
            "other params",
            "other parameters",
            "raises",
            "exceptions",
            "returns",
            "yields",
            "receives",
            "examples",
            "attributes",
            "functions",
            "methods",
            "classes",
            "modules",
            "warns",
            "warnings",
        ],
        style=DocstringStyle.google,
    ),
    DocstringStylePattern(
        pattern=r"\n[ \t]*{0}\n[ \t]*---+\n",
        replacements=[
            "deprecated",
            "parameters",
            "other parameters",
            "returns",
            "yields",
            "receives",
            "raises",
            "warns",
            "attributes",
            "functions",
            "methods",
            "classes",
            "modules",
        ],
        style=DocstringStyle.numpy,
    ),
]


def infer_docstring_style(documentation: str) -> DocstringStyle:
    """
    Infer the docstring style of a given documentation string

    :param documentation: The documentation string to infer the style of
    :return: The inferred docstring style
    """
    for pattern, replacements, style in docstring_style_patterns:
        matches = (
            re.search(pattern.format(replacement), documentation, re.IGNORECASE | re.MULTILINE)
            for replacement in replacements
        )
        if any(matches):
            return style
    return DocstringStyle.google
