"""Render an existing batch JSONL as one readable Markdown company catalogue."""
import argparse
import json
from pathlib import Path


def cell(value):
    if value is None or value == '':
        return 'неизвестно'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def percent(value):
    return 'неизвестно' if value is None else f'{value:.2%}'


def render(profiles):
    ids = [p['company_id'] for p in profiles]
    assert len(set(ids)) == len(ids), 'Duplicate company ID'
    lines = ['# Все компании — готовые описания', '', f'Компаний: {len(profiles)}. Источник: company_profiles.jsonl. ID обезличены.', '',
             'Числа выручки и платежей приведены в исходных единицах: масштаб, валюта платежей и периоды не подтверждены. Платежи не доказывают поставки; отраслевые доли относятся к наблюдаемой выборке. Гипотезы чувствительности не являются оценкой эффекта или мнением банка. Полный provenance и точные структурированные данные — в соседнем company_profiles.jsonl.', '',
             '## Список компаний', '']
    lines += [f'- [{uid}](#{uid.lower()})' for uid in ids]
    for p in profiles:
        uid, raw, derived = p['company_id'], p['raw'], p['derived']
        lines += ['', f'## {uid}', '', f'**Деятельность:** {cell(raw["MAINOKVED_NAME"])} (ОКВЭД {cell(raw["MAINOKVED_CODE"])}).', '',
                  f'**Местоположение регистрации:** {cell(raw["CITY"])}, {cell(raw["REGION"])}. **Статус:** {cell(raw["CL_TYPE"])}.', '',
                  '### Исходные характеристики', '', '| Поле CSV | Значение |', '|---|---|']
        lines += [f'| {key} | {cell(value)} |' for key, value in raw.items()]
        group, position = derived['group'], derived['observed_sector_position']
        lines += ['', '### Группа и положение внутри выборки', '',
                  f'Наблюдаемых компаний группы: {group["observed_members"]}. Сумма доступной выручки группы без консолидации: {group["observed_nonconsolidated_revenue_sum"]:,.0f}.', '',
                  'Участники группы: ' + ', '.join(group['observed_member_ids']) + '.', '',
                  f'Место по выручке в наблюдаемом классе ОКВЭД: {position["revenue_rank_competition"]} из {position["companies"]}; доля доступной выручки выборки: {percent(position["share_of_observed_sector_revenue"])}. Реальная доля рынка неизвестна.', '',
                  f'Страны, записанные в ФТС: {", ".join(derived["fts"]["reported_countries"]) or "неизвестны"}. Направление торговли и объём неизвестны.']
        for direction, label in [('outgoing', 'Исходящие платежи: компания → контрагенты'), ('incoming', 'Входящие платежи: контрагенты → компания')]:
            d = derived['payments'][direction]
            lines += ['', '### ' + label, '', '| Категория | Объём | Количество платежей | Рёбер |', '|---|---:|---:|---:|']
            lines += [f'| {cat} | {x["amount"]:,} | {x["count"]:,} | {x["edges"]} |' for cat, x in d['categories'].items()]
            lines += ['', f'Покрытие метаданными контрагентов по объёму всех наблюдаемых платежей: {percent(d["counterparty_metadata_amount_coverage_all"])}. Покрытие известной отраслью в candidate_external: {percent(d["candidate_external_known_sector_amount_share"])}.', '',
                      f'Знаменатель external_confirmed: {d["external_denominators"]["amount"]:,} объёма, {d["external_denominators"]["count"]:,} платежей. HHI объёма: {cell(d["hhi_amount"])}.', '',
                      'Топ подтверждённых внешних контрагентов:', '', '| ID | Класс ОКВЭД | Объём | Платежей | Доля объёма |', '|---|---|---:|---:|---:|']
            lines += [f'| {x["company_id"]} | {x["division"]} | {x["amount"]:,} | {x["count"]:,} | {percent(x["amount_share"])} |' for x in d['top_counterparties']]
            for field, title, den_field in [('sector_exposures', 'Подтверждённые внешние отрасли', 'external_denominators'), ('candidate_external_sector_exposures', 'Внешние и неизвестные группы: self и известные intra исключены', 'candidate_external_denominators')]:
                den = d[den_field]
                lines += ['', title + f'. Знаменатели: {den["amount"]:,} объёма / {den["count"]:,} платежей.', '', '| Класс ОКВЭД | Объём | Платежей | Доля объёма | Доля количества |', '|---|---:|---:|---:|---:|']
                lines += [f'| {x["division"]} | {x["amount"]:,} | {x["count"]:,} | {percent(x["amount_share"])} | {percent(x["count_share"])} |' for x in d[field]]
            lines += ['', 'Топ всех наблюдаемых контрагентов без self (категории сохраняются):', '', '| ID | Категория | Класс ОКВЭД | Объём | Платежей |', '|---|---|---|---:|---:|']
            lines += [f'| {x["company_id"]} | {x["category"]} | {x["division"]} | {x["amount"]:,} | {x["count"]:,} |' for x in d['top_observed_nonself_counterparties']]
        people = derived['person_links']
        lines += ['', '### Связи через физлиц — только агрегаты', '', f'Наблюдаемых связанных физлиц: {people["person_count"]}. Роли, владение и контроль неизвестны.']
        lines += [f'- {x["company_id"]}: общих связей {x["shared_person_count"]}; есть в справочнике: {x["in_company_table"]}.' for x in people['company_peers']]
        lines += ['', '### Возможные факторы новостей — гипотезы', '']
        lines += [f'- **{x["factor"]}:** {x["hypothesis"]} Основание: {", ".join(x["basis"])}; правило {x["rule"]}.' for x in p['inferred']['news_sensitivity_candidates']]
        lines += ['', '### Качество и ограничения', '', f'Заполненность исходных полей: {p["quality"]["score"]["value"]} из 100. Это не риск и не уверенность в выводе.', '']
        lines += ['- ' + gap for gap in p['data_gaps']]
        evidence = p['provenance']['raw']['SYNTHETIC_U']
        lines += ['', f'Источник: {evidence["file"]}, запись {evidence["source_row"]} с учётом заголовка. SHA256: `{evidence["sha256"]}`.', '']
    text = '\n'.join(lines).rstrip() + '\n'
    assert sum(line.startswith('## U_') for line in text.splitlines()) == len(profiles)
    assert all('person_ids' not in line for line in text.splitlines())
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/company_profiles.jsonl')
    parser.add_argument('--out', default='results/companies.md')
    args = parser.parse_args()
    profiles = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8').splitlines()]
    Path(args.out).write_text(render(profiles), encoding='utf-8')
    print(f'Rendered {len(profiles)} companies: {args.out}')
