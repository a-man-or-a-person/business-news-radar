"""Small deterministic rules from ml/feature-map; no financial predictions."""

import json
import re
from pathlib import Path

from ml.company_profiles_v1.contracts import validate
from ml.news_events_v1.contracts import NewsEvent, validate_evidence
from ml.news_events_v1.pipeline import to_annotation


def load_profiles(path: Path) -> list[dict]:
    profiles = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    seen = set()
    for profile in profiles:
        validate(profile)
        if profile["company_id"] in seen:
            raise ValueError("duplicate company ID")
        seen.add(profile["company_id"])
    return profiles


def checked(event, news):
    validate_evidence(news, event)
    if (
        event.news_id != news.id
        or event.url != news.url
        or event.source_ref != f"news:{news.id}"
    ):
        raise ValueError("event source mismatch")
    for tag in event.tag_evidence:
        if (
            not tag.quote.strip()
            or tag.quote not in getattr(news, tag.field)
            or not any(
                fact.field == tag.field and tag.quote in fact.quote
                for fact in event.facts
            )
            or tag.value not in getattr(event, tag.kind)
        ):
            raise ValueError("unsupported tag evidence")


def read_events(path: Path, news_by_id: dict) -> list[NewsEvent]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        event = NewsEvent.model_validate(row.get("payload", row))
        if "payload" in row and any(
            row[key] != value
            for key, value in to_annotation(event).items()
            if key not in {"payload", "severity", "embedding", "embedding_model"}
            and key in row
        ):
            raise ValueError("annotation payload mismatch")
        checked(event, news_by_id[event.news_id])
        events.append(event)
    return events


def rank(
    profile: dict,
    events: list[NewsEvent],
    news_by_id: dict,
    clusters=None,
    *,
    version="baseline",
) -> list[dict]:
    if version != "baseline":
        raise ValueError("unknown matcher version")
    candidates = {}
    raw, derived = profile["raw"], profile["derived"]
    for event in events:
        news = news_by_id[event.news_id]
        checked(event, news)
        if event.status == "insufficient_data" or profile["quality"]["raw_id_conflict"]:
            continue
        features = {}

        def add(path, value, weight, rule, evidence=None, features=features):
            features[path] = {
                "path": path,
                "value": value,
                "weight": weight,
                "rule": rule,
                "evidence": evidence
                or profile["provenance"]["raw"][path.split(".")[-1]],
                "source_ref": f"profile:{profile['company_id']}:{path}",
            }

        supported = {(t.kind, t.value) for t in event.tag_evidence}
        # shortcut: only reviewed macro tags trigger product rules; extend the gazetteer with labelled evidence.
        products = []
        if (
            event.event_type == "key_rate"
            and ("macro_factors", "key_rate") in supported
        ):
            products = ["LOA", "LEA", "DEP"]
            if derived["products"]["FACTORING_MARK"] is True:
                add("raw.FACTORING_MARK", raw["FACTORING_MARK"], 20, "key_rate_product")
        if (
            event.event_type == "labour_market"
            and ("macro_factors", "labour_market") in supported
        ):
            products = ["ZPP"]
            for field in [
                "AMOUNT_WORKERS_FNS_GRADATION",
                "AMOUNT_WORKERS_SPARK_GRADATION",
            ]:
                if raw[field].strip():
                    add("raw." + field, raw[field], 10, "labour_workforce")
        for product in products:
            field = f"CL_PRODUCT_{product}_MARK"
            if derived["products"][field] is True:
                add("raw." + field, raw[field], 20, event.event_type + "_product")
        if (
            event.event_type in ("currency", "sanctions", "logistics")
            and derived["fts"]["flag"] is True
        ):
            countries = [
                c
                for c in derived["fts"]["reported_countries"]
                if any(
                    kind == "countries" and value.casefold() == c.casefold()
                    for kind, value in supported
                )
            ]
            if countries:
                add("raw.FTS_FLAG", raw["FTS_FLAG"], 15, "trade_country")
                add("raw.FTS_COUNTRY_NAME", countries, 20, "trade_country")
        for field in ["REGION", "CITY"]:
            if (
                event.event_type not in ("other", "unknown", "corporate")
                and raw[field].strip()
                and any(
                    kind == "regions" and value.casefold() == raw[field].casefold()
                    for kind, value in supported
                )
            ):
                add("raw." + field, raw[field], 20, "exact_region")
        # No automatic OKVED extraction: accept explicit caller classifications only, still requiring review.
        divisions = {c for c in event.okved_classes if re.fullmatch(r"\d{2}", c)}
        if derived["industry"]["okved_division"] in divisions:
            add("raw.MAINOKVED_CODE", raw["MAINOKVED_CODE"], 25, "own_okved")
        for direction in ["incoming", "outgoing"]:
            for sector in derived["payments"][direction]["sector_exposures"]:
                if sector["division"] in divisions and sector["amount"] > 0:
                    path = f"derived.payments.{direction}.sector_exposures:{sector['division']}"
                    add(
                        path,
                        sector,
                        10,
                        "external_confirmed_okved",
                        profile["provenance"]["derived"]["payments"],
                    )
        if not features:
            continue
        card = make_candidate(profile, event, news, features, clusters)
        event_id = card["event_id"]
        previous = candidates.get(event_id)
        if previous is None or (
            -card["score"],
            card["news_id"],
            json.dumps(card, sort_keys=True),
        ) < (
            -previous["score"],
            previous["news_id"],
            json.dumps(previous, sort_keys=True),
        ):
            candidates[event_id] = card
    return sorted(
        candidates.values(), key=lambda c: (-c["score"], c["event_id"], c["news_id"])
    )


def make_candidate(profile, event, news, features, clusters=None):
    event_id = str((clusters or {}).get(event.news_id) or event.news_id)
    provenance = (
        "offline_baseline"
        if event.processing.model == "offline-keyword-baseline"
        else "saved_extraction"
    )
    card = {
        "schema_version": "insight-candidate/1",
        "company_id": profile["company_id"],
        "news_id": event.news_id,
        "event_id": event_id,
        "event_type": event.event_type,
        "title": news.title,
        "facts": [f.model_dump() for f in event.facts],
        "assertion_status": event.assertion_status,
        "provenance": provenance,
        "processing": event.processing.model_dump(),
        "matched_features": list(features.values()),
        "score": min(100, sum(f["weight"] for f in features.values())),
        "score_meaning": "relative relevance points, not probability or financial impact",
        "verification_status": "review_required",
        "hypothesis": "Возможная связь события с наблюдаемыми признаками; направление и величина бизнес-эффекта неизвестны.",
        "limitations": event.information_gaps
        + profile["data_gaps"]
        + [
            "Evidence validation does not establish semantic correctness.",
            "Missing features do not prove absence of risk.",
        ],
        "sources": [
            {
                "id": event.source_ref,
                "url": news.url,
                "title": news.title,
                "published_at": news.published_at,
            }
        ],
        "explanation": "; ".join(
            f"{f['rule']}: {f['path']}={f['value']}" for f in features.values()
        ),
    }
    return card
