import pytest

from ml.news_events_v1.contracts import Extraction, NewsSnapshot, Processing
from ml.news_events_v1.pipeline import assemble, baseline, read_snapshot, to_annotation


@pytest.fixture
def news():
    return NewsSnapshot(
        id="cbr_0123456789ab",
        source_id="cbr",
        url="https://example.com/a",
        title="Банк России повысил ключевую ставку до 20%",
        published_at="2026-10-23T12:00:00Z",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"published_at": "2026-02-30T00:00:00Z"},
        {"published_at": "2026-10-23T00:00:00"},
        {"lead": None},
        {"title": " "},
        {"id": "bad"},
        {"url": "file:///a"},
        {"extra": "unexpected"},
        {"date_guessed": 1},
    ],
)
def test_invalid_input_rejected(news, changes):
    with pytest.raises(ValueError):
        NewsSnapshot.model_validate({**news.model_dump(), **changes})


def test_duplicate_and_conflicting_records(news, tmp_path):
    path = tmp_path / "news.jsonl"
    path.write_text((news.model_dump_json() + "\n") * 2, encoding="utf-8")
    rows, duplicates = read_snapshot(path)
    assert len(rows) == 1 and duplicates == 1
    path.write_text(
        news.model_dump_json()
        + "\n"
        + news.model_copy(update={"title": "Другой заголовок"}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting ID at row 2"):
        read_snapshot(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"quote": "Выдуманная цитата"},
        {"numbers": ["30%"]},
        {"entities": ["Несуществующий банк"]},
        {"dates": ["2026-02-30"]},
    ],
)
def test_unsupported_fact_rejected(news, changes):
    raw = baseline(news).model_dump(include=set(Extraction.model_fields))
    raw["facts"][0].update(changes)
    with pytest.raises(ValueError):
        assemble(
            news,
            Extraction.model_validate(raw),
            Processing(model="fake", prompt_version="1", date_guessed=False),
        )


def test_abstention_and_forecast_are_distinct(news):
    result = Extraction(
        status="insufficient_data",
        event_type="unknown",
        assertion_status="uncertain",
        summary=None,
        facts=[],
        information_gaps=["ambiguous headline"],
    )
    event = assemble(
        news, result, Processing(model="fake", prompt_version="1", date_guessed=False)
    )
    assert to_annotation(event)["payload"]["status"] == "insufficient_data"
    forecast = baseline(news.model_copy(update={"title": "Ожидается инфляция 10%"}))
    assert forecast.facts[0].assertion_status == "forecast"


def test_impossible_literal_date_rejected(news):
    news = news.model_copy(update={"title": "Решение принято 30.02.2026"})
    raw = baseline(news).model_dump(include=set(Extraction.model_fields))
    raw["facts"][0]["dates"] = ["30.02.2026"]
    with pytest.raises(ValueError):
        assemble(
            news,
            Extraction.model_validate(raw),
            Processing(model="fake", prompt_version="1", date_guessed=False),
        )


def test_russian_full_date_and_numeric_boundaries(news):
    for title, field, mention in [
        ("Решение 31 февраля 2026", "dates", "31 февраля 2026"),
        ("Ставка 120%", "numbers", "20%"),
    ]:
        item = news.model_copy(update={"title": title})
        raw = baseline(item).model_dump(include=set(Extraction.model_fields))
        raw["facts"][0][field] = [mention]
        with pytest.raises(ValueError):
            assemble(
                item,
                Extraction.model_validate(raw),
                Processing(model="fake", prompt_version="1", date_guessed=False),
            )


def test_long_valid_title_and_supported_tags(news):
    long = baseline(news.model_copy(update={"title": "Длинный заголовок " * 100}))
    assert len(long.summary) <= 1000
    event = baseline(
        news.model_copy(update={"title": "Цены на нефть в России выросли"})
    )
    assert event.commodities == ["oil"]
    assert event.countries == ["РОССИЯ"]
    assert event.tag_evidence[0].quote in event.facts[0].quote
    assert (
        baseline(
            news.model_copy(update={"title": "Газета сообщила о решении"})
        ).commodities
        == []
    )


def test_title_only_publication_can_produce_supported_sql_annotation(tmp_path):
    path = tmp_path / "news.jsonl"
    path.write_text(
        '{"id":"cbr_0123456789ab","source_id":"cbr","url":"https://example.com/a",'
        '"title":"Банк России повысил ключевую ставку до 20%",'
        '"published_at":"2026-10-23T12:00:00Z"}\n',
        encoding="utf-8",
    )
    rows, duplicates = read_snapshot(path)
    event = baseline(rows[0])
    payload = to_annotation(event)
    assert duplicates == 0
    assert event.event_type == "key_rate"
    assert event.facts[0].quote == "Банк России повысил ключевую ставку до 20%"
    assert payload["severity"] is None
    assert payload["embedding"] is None
    assert payload["payload"]["news_id"] == "cbr_0123456789ab"
