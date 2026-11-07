"""Run the hackathon profiling and matching components on invented inputs."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.company_event_matching_v1.matching import rank
from ml.company_profiles_v1.contracts import RAW_FIELDS, validate
from ml.company_profiles_v1.pipeline import Batch, COMPANIES, LINKS, PAYMENTS, PRODUCTS
from ml.news_events_v1.contracts import NewsSnapshot
from ml.news_events_v1.pipeline import baseline


def write_inputs(folder: Path):
    companies = []
    for number in range(1, 4):
        row = dict.fromkeys(RAW_FIELDS, "")
        row.update(SYNTHETIC_U=f"U_{9000000 + number}", ORG_TYPE="КОММ_ОРГ",
                   MAINOKVED_CODE="24.10", MAINOKVED_NAME="Synthetic sector",
                   REVENUE_COMPANY="20000000000", GROUP_ID=f"demo_group_{number}",
                   CITY="Synthetic city", REGION="Synthetic region", CL_TYPE="CLIENT",
                   NEGATIVEMARK="0", FTS_FLAG="0")
        row.update(dict.fromkeys(PRODUCTS, "0"))
        if number == 1:
            row["CL_PRODUCT_LOA_MARK"] = "1"
        if number == 2:
            row["CL_PRODUCT_ZPP_MARK"] = "1"
        companies.append(row)
    with (folder / COMPANIES).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(companies)
    for filename, header, rows in [
        (PAYMENTS, ["PAYER", "BENEFICIARY", "PAY_AMT", "PAY_CNT"], [
            ["U_9000001", "U_9000001", 100, 1],
            ["U_9000001", "U_9000002", 300, 3],
            ["U_9000002", "U_9000001", 600, 6],
        ]),
        (LINKS, ["SYNTHETIC_F", "SYNTHETIC_U", "CITY_NAME", "RC_NAME"], []),
    ]:
        with (folder / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(header)
            writer.writerows(rows)


def run_demo() -> dict:
    with TemporaryDirectory(prefix="radar-synthetic-") as temporary:
        folder = Path(temporary)
        write_inputs(folder)
        batch = Batch(folder)
        profiles = [batch.build_company_profile(uid) for uid in batch.targets]
        for profile in profiles:
            validate(profile)
    news = [NewsSnapshot(
        id=f"demo_{number:012x}", source_id="demo", url=f"https://example.com/news/{number}",
        title=title, published_at="2026-01-01T12:00:00Z",
    ) for number, title in enumerate([
        "Вымышленная новость: ключевая ставка повышена до 10%",
        "Вымышленная новость: зарплаты в отрасли выросли",
        "Вымышленная новость: цены на нефть снизились",
    ], 1)]
    by_id = {item.id: item for item in news}
    events = [baseline(item) for item in news]
    candidates = {profile["company_id"]: rank(profile, events, by_id) for profile in profiles}
    assert len(profiles) == 3
    assert [card["event_type"] for card in candidates["U_9000001"]] == ["key_rate"]
    assert [card["event_type"] for card in candidates["U_9000002"]] == ["labour_market"]
    assert candidates["U_9000003"] == []
    assert profiles[0]["derived"]["payments"]["outgoing"]["all_observed"]["amount"] == 400
    return {"data_source": "synthetic", "profiles": profiles,
            "news": [item.model_dump() for item in news],
            "events": [item.model_dump() for item in events], "candidates": candidates}


def render_report(result: dict) -> str:
    lines = ["# Business News Radar — synthetic demo", "",
             "Invented inputs; deterministic baseline; all candidates require review.", "",
             "| Company | Selected event | Relevance points |", "|---|---|---|"]
    for uid, cards in result["candidates"].items():
        if not cards:
            lines.append(f"| {uid} | No supported match | — |")
        for card in cards:
            lines.append(f"| {uid} | {card['event_type']} | {card['score']} |")
    lines += ["", "Points are rule contributions, not probabilities or predicted financial effects.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError("Choose a new output directory.")
    result = run_demo()
    report = render_report(result)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
