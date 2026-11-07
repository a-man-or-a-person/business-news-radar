"""Publication reader, deterministic comparator and pure storage adapter."""

import json
import re
from pathlib import Path

from .contracts import (
    Extraction,
    Fact,
    NewsEvent,
    NewsSnapshot,
    Processing,
    TagEvidence,
    validate_evidence,
)


def read_snapshot(path: Path) -> tuple[list[NewsSnapshot], int]:
    rows, seen, duplicates = [], {}, 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                news = NewsSnapshot.model_validate_json(line)
            except ValueError:
                raise ValueError(f"invalid snapshot row {line_no}") from None
            if news.id in seen:
                if seen[news.id] != news:
                    raise ValueError(f"conflicting ID at row {line_no}")
                duplicates += 1
                continue
            seen[news.id] = news
            rows.append(news)
    return rows, duplicates


def assemble(
    news: NewsSnapshot, extraction: Extraction, processing: Processing
) -> NewsEvent:
    validate_evidence(news, extraction)
    tags = {"countries": [], "regions": [], "commodities": [], "macro_factors": []}
    evidence = []
    mappings = [
        ("countries", "РОССИЯ", r"\bРосси[а-я]*\b"),
        ("countries", "США", r"\bСША\b"),
        ("countries", "ТУРЦИЯ", r"\bТурци[а-я]*\b"),
        ("regions", "Москва", r"\bМоскв[а-я]*\b"),
        ("commodities", "oil", r"\bнефт[а-я]*\b"),
        ("commodities", "gas", r"\bгаз(?:а|у|ом|е|ы|ов|ам|ами|ах)?\b"),
        ("commodities", "gold", r"\bзолот[а-я]*\b"),
        ("macro_factors", "inflation", r"\bинфляци[а-я]*\b"),
        ("macro_factors", "key_rate", r"\bключев[а-я]*\s+ставк[а-я]*\b"),
        ("macro_factors", "labour_market", r"\b(?:зарплат|безработиц)[а-я]*\b"),
    ]
    for kind, value, pattern in mappings:
        for fact in extraction.facts:
            match = re.search(pattern, fact.quote, flags=re.IGNORECASE)
            if match:
                tags[kind].append(value)
                evidence.append(
                    TagEvidence(
                        kind=kind, value=value, quote=match[0], field=fact.field
                    )
                )
                break
    return NewsEvent(
        **extraction.model_dump(),
        news_id=news.id,
        source_ref=f"news:{news.id}",
        url=news.url,
        entities=sorted({name for fact in extraction.facts for name in fact.entities}),
        processing=processing,
        **tags,
        tag_evidence=evidence,
    )


def baseline(news: NewsSnapshot) -> NewsEvent:
    text = f"{news.title} {news.lead}".lower()
    patterns = [
        ("key_rate", ("ключев", "ставк")),
        ("inflation", ("инфляц",)),
        ("sanctions", ("санкци",)),
        ("regulation", ("закон", "законопроект", "норматив")),
        ("currency", ("валют", "курс рубл", "курс доллар")),
        ("commodity_prices", ("нефт", "газ", "золот", "сырь")),
        ("labour_market", ("зарплат", "безработ", "рынок труда")),
        ("logistics", ("логист", "перевоз", "поставк")),
        ("consumer_demand", ("потребител", "рознич", "спрос")),
        ("corporate", ("компани", "дивиденд", "акци")),
    ]
    category = "other"
    for event_type, words in patterns:
        if (
            all(word in text for word in words)
            if event_type == "key_rate"
            else any(word in text for word in words)
        ):
            category = event_type
            break
    assertion = "occurred"
    for status, words in [
        ("forecast", ("прогноз", "ожида", "может", "будет")),
        ("proposed", ("предлож", "законопроект")),
        ("opinion", ("мнение", "считает")),
        ("announced", ("объяв", "анонс")),
    ]:
        if any(word in text for word in words):
            assertion = status
            break
    result = Extraction(
        status="partial",
        event_type=category,
        assertion_status=assertion,
        summary=news.title[:1000],
        facts=[
            Fact(quote=news.title[:1000], field="title", assertion_status=assertion)
        ],
        information_gaps=["keyword baseline; interpretation requires review"],
    )
    return assemble(
        news,
        result,
        Processing(
            model="offline-keyword-baseline",
            prompt_version="baseline/1",
            date_guessed=news.date_guessed,
        ),
    )


def to_annotation(event: NewsEvent) -> dict:
    payload = event.model_dump(mode="json")
    names = (
        "news_id",
        "schema_version",
        "event_type",
        "okved_classes",
        "regions",
        "countries",
        "commodities",
        "macro_factors",
        "summary",
        "facts",
    )
    return {
        **{key: payload[key] for key in names},
        "payload": payload,
        "model": event.processing.model,
        "prompt_version": event.processing.prompt_version,
        "severity": None,
        "embedding": None,
        "embedding_model": None,
    }


def write_json(path: Path, value) -> None:
    """Atomic replace in the same directory; callers control overwrite policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
