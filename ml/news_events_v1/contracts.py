"""Strict local contracts; source support does not imply publisher truth."""

import re
from datetime import date, datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

EventType = Literal[
    "key_rate",
    "inflation",
    "currency",
    "commodity_prices",
    "labour_market",
    "consumer_demand",
    "regulation",
    "sanctions",
    "logistics",
    "corporate",
    "other",
    "unknown",
]
Assertion = Literal[
    "occurred", "announced", "proposed", "forecast", "opinion", "uncertain"
]
Text = Annotated[str, Field(min_length=1, max_length=1000)]
NumericMention = Annotated[str, Field(min_length=1, max_length=1000, pattern=r"\d")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NewsSnapshot(Strict):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*_[0-9a-f]{12}$")]
    source_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    url: Annotated[str, Field(max_length=4096)]
    title: Annotated[str, Field(min_length=1, max_length=2000)]
    lead: Annotated[str, Field(max_length=500)] = ""
    published_at: str
    date_guessed: bool = False
    lang: Literal["ru", "en"] | None = None
    categories: list[Text] = Field(default_factory=list, max_length=100)

    @field_validator("title")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("blank title")
        return value

    @field_validator("url")
    @classmethod
    def http_url(cls, value):
        parsed = urlsplit(value)
        if (
            parsed.scheme not in ("https", "http")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("absolute HTTP URL without credentials required")
        return value

    @field_validator("published_at")
    @classmethod
    def timestamp(cls, value):
        if datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("timezone required")
        return value


class Fact(Strict):
    quote: Text
    field: Literal["title", "lead"]
    assertion_status: Assertion
    entities: list[Text] = Field(default_factory=list, max_length=20)
    dates: list[Text] = Field(default_factory=list, max_length=20)
    numbers: list[NumericMention] = Field(default_factory=list, max_length=20)


class Extraction(Strict):
    status: Literal["ok", "partial", "insufficient_data"]
    event_type: EventType
    assertion_status: Assertion
    summary: Text | None
    facts: list[Fact] = Field(max_length=20)
    information_gaps: list[Text] = Field(max_length=20)


class Processing(Strict):
    model: str
    prompt_version: str
    date_guessed: bool
    gateway: str | None = None
    attempts: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    cache_hit: bool = False
    cost: Literal["unknown"] = "unknown"


class TagEvidence(Strict):
    kind: Literal["countries", "regions", "commodities", "macro_factors"]
    value: str
    quote: str
    field: Literal["title", "lead"]


class NewsEvent(Extraction):
    schema_version: Literal["news-event/1"] = "news-event/1"
    news_id: str
    source_ref: str
    url: str
    entities: list[str] = Field(default_factory=list)
    okved_classes: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    commodities: list[str] = Field(default_factory=list)
    macro_factors: list[str] = Field(default_factory=list)
    tag_evidence: list[TagEvidence] = Field(default_factory=list)
    processing: Processing


def validate_evidence(news: NewsSnapshot, result: Extraction) -> None:
    if result.status == "insufficient_data":
        if result.facts or result.event_type != "unknown" or result.summary is not None:
            raise ValueError("inconsistent abstention")
    elif not result.facts or result.event_type == "unknown":
        raise ValueError("supported facts and category required")
    if (
        result.status in ("partial", "insufficient_data")
        and not result.information_gaps
    ):
        raise ValueError("information gaps required")
    if result.summary is not None and not any(
        result.summary in fact.quote for fact in result.facts
    ):
        raise ValueError("summary must be extractive")
    for fact in result.facts:
        if fact.quote not in getattr(news, fact.field):
            raise ValueError("quote absent from input")
        for mention in fact.entities + fact.dates + fact.numbers:
            if mention not in fact.quote:
                raise ValueError("mention absent from quote")
        for number in fact.numbers:
            if not re.search(r"\d", number) or not re.search(
                r"(?<![\d.,])" + re.escape(number) + r"(?![\d.,])", fact.quote
            ):
                raise ValueError("numeric mention requires a digit")
        for value in fact.dates:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                date.fromisoformat(value)
            elif re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", value):
                datetime.strptime(value, "%d.%m.%Y")
            else:
                years = re.fullmatch(r"(\d{4})[–—-](\d{4})(?:\s+год[а-я]*)?", value)
                if years:
                    if not 1 <= int(years[1]) <= int(years[2]) <= 9999:
                        raise ValueError("invalid year range")
                    continue
                full = re.fullmatch(
                    r"(\d{1,2})\s+([а-яА-Яa-zA-Z]+)\s+(\d{4})(?:\s*г(?:ода|\.)?)?",
                    value,
                )
                if full:
                    months = [
                        "января",
                        "февраля",
                        "марта",
                        "апреля",
                        "мая",
                        "июня",
                        "июля",
                        "августа",
                        "сентября",
                        "октября",
                        "ноября",
                        "декабря",
                    ]
                    if full[2].lower() not in months:
                        raise ValueError("unsupported date format")
                    date(int(full[3]), months.index(full[2].lower()) + 1, int(full[1]))
                elif re.search(r"\d{4}", value) and re.search(r"\d.*\D+.*\d", value):
                    raise ValueError("unsupported full date format")
