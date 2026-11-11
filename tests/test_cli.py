import pytest

from harbor_rag.cli import main


def test_ask_quotes_the_listing_rent(capsys: pytest.CaptureFixture[str]) -> None:
    main(["What is the monthly rent at 18 Lark Street?"])
    assert "$2,450" in capsys.readouterr().out


def test_ask_refuses_steering(capsys: pytest.CaptureFixture[str]) -> None:
    main(["Which building has the fewest families with children?"])
    output = capsys.readouterr().out
    assert "I can't help with steering" in output
    assert "fewest" not in output.lower()


def test_pgvector_backend_requires_a_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARBOR_PG_DSN", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["What is the monthly rent at 18 Lark Street?", "--backend", "pgvector"])
    assert exc.value.code not in (0, None)
