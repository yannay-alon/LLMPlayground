import inspect
from typing import Any

def indent_code(text: Any, indent_length: int | None = None) -> str:
    text = str(text)

    if indent_length is None:
        current_frame = inspect.currentframe()
        outer_frame_information = inspect.getframeinfo(current_frame.f_back)
        context_lines = outer_frame_information.code_context
        context_line_index = outer_frame_information.index
        context_line = context_lines[context_line_index]

        indentation_length = len(context_line) - len(context_line.lstrip())
        line_indentation = context_line[:indentation_length]
    else:
        line_indentation = " " * indent_length

    indented_lines = []
    for line_index, line in enumerate(text.splitlines(True)):
        if line_index == 0:
            indented_lines.append(line)
            continue
        else:
            indented_lines.append(line_indentation + line if line.strip() else line)
    return "".join(indented_lines)
