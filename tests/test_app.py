import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app
import types
import app


def test_respond_function_exists():
    """Check that the app has a respond() function."""
    assert hasattr(app, "respond")
    assert callable(app.respond)


def test_respond_returns_generator():
    """respond() should return a generator when called with minimal args."""
    class DummyToken:
        token = "dummy"

    gen = app.respond(
        message="I'm buying a bottle of water.",
        history=[],
        system_message="You are Sustainable.ai.",
        max_tokens=10,
        temperature=0.7,
        top_p=0.9,
        hf_token=DummyToken(),
    )

    assert isinstance(gen, types.GeneratorType)