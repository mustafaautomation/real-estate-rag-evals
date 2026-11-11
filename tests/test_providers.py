from harbor_rag.guardrails import UNKNOWN_RESPONSE
from harbor_rag.providers import AnthropicProvider, ExtractiveProvider, OpenAIProvider


def test_extractive_provider_quotes_the_rent_sentence() -> None:
    provider = ExtractiveProvider()
    contexts = [
        "18 Lark Street. 18 Lark Street is a 2-bedroom condominium. Monthly rent is $2,450."
    ]
    answer = provider.generate(
        system="system",
        question="What is the monthly rent at 18 Lark Street?",
        contexts=contexts,
    )
    assert "Monthly rent is $2,450." in answer


def test_extractive_provider_does_not_quote_a_different_address() -> None:
    provider = ExtractiveProvider()
    answer = provider.generate(
        system="system",
        question="What is the monthly rent at 500 Imaginary Boulevard?",
        contexts=["18 Lark Street. Monthly rent is $2,450."],
    )
    assert answer == UNKNOWN_RESPONSE
    assert "$2,450" not in answer


def test_openai_provider_passes_sources() -> None:
    captured: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            message = type("Message", (), {"content": "Monthly rent is $2,450."})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    provider = OpenAIProvider(Client(), model="gpt-4o-mini")
    text = provider.generate(
        system="be grounded", question="rent?", contexts=["Monthly rent is $2,450."]
    )
    assert text == "Monthly rent is $2,450."
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "Monthly rent is $2,450." in str(messages)


def test_anthropic_provider_reads_text_block() -> None:
    class Messages:
        def create(self, **kwargs: object) -> object:
            block = type("Block", (), {"text": "Rent is due on the first day of each month."})()
            return type("Response", (), {"content": [block]})()

    class Client:
        messages = Messages()

    provider = AnthropicProvider(Client(), model="claude-3-5-haiku-latest")
    text = provider.generate(system="sys", question="when is rent due?", contexts=["source"])
    assert "first day" in text


def test_openai_provider_empty_content_is_unknown() -> None:
    class Completions:
        def create(self, **kwargs: object) -> object:
            message = type("Message", (), {"content": "  "})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    assert (
        OpenAIProvider(Client(), model="gpt-4o-mini").generate(
            system="s", question="q", contexts=[]
        )
        == UNKNOWN_RESPONSE
    )
