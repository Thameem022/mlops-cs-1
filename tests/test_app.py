from app import chat

def test_chat_response():
    user_input = "I'm buying a plastic bottle of water."
    response = chat(user_input)
    assert isinstance(response, str)
    assert len(response) > 0
