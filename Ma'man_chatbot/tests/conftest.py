import pytest
import time
from app.core.chatbot import ChatBot


@pytest.fixture(scope="session")
def bot():
    """
    Create a single ChatBot instance for all tests, and wait
    until the embedding model has finished loading before
    handing it to any test. This avoids the same race condition
    documented in Section 5.14.7 (background model loading).
    """
    chatbot = ChatBot()

    max_wait = 60  # seconds
    waited = 0
    while not chatbot.engine.is_ready() and waited < max_wait:
        time.sleep(1)
        waited += 1

    if not chatbot.engine.is_ready():
        pytest.fail(f"Embedding model did not become ready within {max_wait}s")

    return chatbot