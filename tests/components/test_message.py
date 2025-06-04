from components.messages import UserMessage, SystemMessage, AssistantMessage, ToolMessage, MessageFactory


class TestMessages:
    def test_user_message(self):
        message = UserMessage(content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

    def test_system_message(self):
        message = SystemMessage(content="System instruction")
        assert message.role == "system"
        assert message.content == "System instruction"

    def test_assistant_message(self):
        message = AssistantMessage(content="Assistant response")
        assert message.role == "assistant"
        assert message.content == "Assistant response"

    def test_tool_message(self):
        message = ToolMessage(content="Tool output")
        assert message.role == "tool"
        assert message.content == "Tool output"
        assert message.identifier  # Should have auto-generated UUID

    def test_tool_message_with_custom_id(self):
        message = ToolMessage(content="Tool output", id="custom-id")
        assert message.identifier == "custom-id"

    def test_message_factory(self):
        messages = [
            ("user", "User input"),
            ("system", "System message"),
            ("assistant", "Assistant reply"),
            ("tool", "Tool result"),
            ("unknown", "Unknown type")
        ]

        for role, content in messages:
            message = MessageFactory.create_message(role=role, content=content)
            assert message.role == role
            assert message.content == content
