import json
import socket
from copy import deepcopy

import pytest

from demo import main, run_demo
from ml.company_event_matching_v1.matching import rank
from ml.news_events_v1.contracts import NewsEvent, NewsSnapshot


def test_demo_is_offline_and_personalized(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("No network in the demo")
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    result = run_demo()
    assert len(result["profiles"]) == 3
    assert len(result["news"]) == 3
    assert len(result["candidates"]["U_9000001"]) == 1
    assert result["candidates"]["U_9000003"] == []
    for cards in result["candidates"].values():
        for card in cards:
            assert card["verification_status"] == "review_required"
            assert card["provenance"] == "offline_baseline"
            assert all(feature["evidence"] for feature in card["matched_features"])


def test_source_and_quote_mismatch_rejected():
    result = run_demo()
    profile = result["profiles"][0]
    news = NewsSnapshot.model_validate(result["news"][0])
    event = NewsEvent.model_validate(result["events"][0])
    for field, value in (("url", "https://example.com/wrong"), ("source_ref", "news:wrong")):
        broken = event.model_copy(update={field: value})
        with pytest.raises(ValueError, match="source mismatch"):
            rank(profile, [broken], {news.id: news})
    event.facts[0].quote = "invented unsupported quote"
    with pytest.raises(ValueError):
        rank(profile, [event], {news.id: news})


def test_missing_product_does_not_invent_exposure():
    result = run_demo()
    profile = deepcopy(result["profiles"][0])
    profile["derived"]["products"] = dict.fromkeys(profile["derived"]["products"])
    by_id = {row["id"]: NewsSnapshot.model_validate(row) for row in result["news"]}
    events = [NewsEvent.model_validate(row) for row in result["events"]]
    assert rank(profile, events, by_id) == []


def test_rank_is_deterministic_and_deduplicates_clusters():
    result = run_demo()
    profile = result["profiles"][0]
    by_id = {row["id"]: NewsSnapshot.model_validate(row) for row in result["news"]}
    events = [NewsEvent.model_validate(row) for row in result["events"]]
    assert rank(profile, events, by_id) == rank(profile, list(reversed(events)), by_id)
    assert len(rank(profile, [events[0], events[0]], by_id, {events[0].news_id: "demo-cluster"})) == 1
    with pytest.raises(ValueError, match="unknown matcher"):
        rank(profile, events, by_id, version="unavailable")


def test_export_and_refusal_to_overwrite(tmp_path):
    output = tmp_path / "new-demo"
    main(["--output", str(output)])
    original = (output / "result.json").read_bytes()
    result = json.loads(original)
    assert result["data_source"] == "synthetic"
    assert "No supported match" in (output / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert (output / "result.json").read_bytes() == original
