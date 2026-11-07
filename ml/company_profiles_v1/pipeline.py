"""Offline CompanyProfile v1.0. No network access or model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

COMPANIES = 'df_Uid_data_hack.csv'
PAYMENTS = 'df_Uid_payment_hack.csv'
LINKS = 'df_Fid_Uid_link_hack.csv'
FLAGS = ['NEGATIVEMARK', 'FTS_FLAG', 'FACTORING_MARK']
PRODUCTS = ['CL_PRODUCT_' + x + '_MARK' for x in ['RKO', 'GAR', 'LOA', 'LEA', 'DEP', 'ZPP']] + ['FACTORING_MARK']
CATEGORIES = ['self', 'intragroup', 'external_confirmed', 'group_unknown']
GAPS = [
    'Нет даты среза, года выручки, периода платежей и валюты/единицы PAY_AMT.',
    'ТЗ задаёт порог в рублях, но единица REVENUE_COMPANY в CSV явно не обозначена; отбор по числу >15000000000 условен до подтверждения масштаба.',
    'Платежи не доказывают поставки, роль покупателя/поставщика и характер договора.',
    'Полнота банковского покрытия неизвестна; отсутствующие платежи не равны отсутствию бизнеса.',
    'Нет рыночного знаменателя, поэтому реальная доля рынка и позиция на рынке неизвестны.',
    'ФТС не задаёт направление торговли, объём, валюту, товары или даты операций.',
    'Связь с физлицом не задаёт роль, собственность, контроль или бенефициарное владение.',
    'Флаги продуктов не задают остатки, сроки, валюты, ставки или долю финансирования.',
    'Группы наблюдаются частично; суммы выручки по группе не консолидированы.',
]


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def division(code):
    # Keep raw OKVED verbatim; leading two digits are a separate derived feature.
    return code[:2] if isinstance(code, str) and re.fullmatch(r'\d{2}(?:\.\d{1,2}){0,2}', code) else None


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def records(frame):
    return native(frame.to_dict('records'))


def table_eda(df):
    columns = {}
    for col in df:
        series = df[col]
        nonempty = series[series != '']
        numbers = pd.to_numeric(nonempty, errors='coerce')
        columns[col] = {
            'empty': int((series == '').sum()), 'distinct_nonempty': int(nonempty.nunique()),
            'whitespace_only': int(series.str.fullmatch(r'\s+').sum()),
            'numeric_parse_failures_nonempty': int(numbers.isna().sum()),
            'numeric_summary': native(numbers.describe(percentiles=[.01, .25, .5, .75, .95, .99]).to_dict()) if numbers.notna().any() else None,
        }
    return {'rows': len(df), 'columns': len(df.columns), 'exact_duplicate_excess_rows': int(df.duplicated().sum()), 'fields': columns}


class Batch:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.frames = {name: pd.read_csv(self.data_dir / name, sep=';', dtype=str, keep_default_na=False) for name in [COMPANIES, PAYMENTS, LINKS]}
        c, p, links = (self.frames[name] for name in [COMPANIES, PAYMENTS, LINKS])
        required = {COMPANIES: ['SYNTHETIC_U', 'REVENUE_COMPANY', 'GROUP_ID', 'MAINOKVED_CODE', 'MAINOKVED_NAME', 'CL_TYPE', 'REGION', 'CITY', 'FTS_FLAG', 'FTS_COUNTRY_NAME'] + PRODUCTS,
                    PAYMENTS: ['PAYER', 'BENEFICIARY', 'PAY_AMT', 'PAY_CNT'], LINKS: ['SYNTHETIC_F', 'SYNTHETIC_U', 'CITY_NAME', 'RC_NAME']}
        for name, cols in required.items():
            if set(cols) - set(self.frames[name]):
                raise ValueError(f'{name}: missing columns {set(cols) - set(self.frames[name])}')
        for df, cols in [(c, ['SYNTHETIC_U']), (p, ['PAYER', 'BENEFICIARY']), (links, ['SYNTHETIC_F', 'SYNTHETIC_U'])]:
            for col in cols:
                pattern = r'F_\d+' if col == 'SYNTHETIC_F' else r'U_\d+'
                if not df[col].str.fullmatch(pattern).all():
                    raise ValueError(f'Empty or malformed key: {col}')
        for col in FLAGS + PRODUCTS:
            if not set(c[col]) <= {'', '0.0', '1.0', '0', '1'}:
                raise ValueError(f'Invalid flag values: {col}')
        for df, cols in [(c, ['REVENUE_COMPANY']), (p, ['PAY_AMT', 'PAY_CNT'])]:
            for col in cols:
                n = pd.to_numeric(df[col], errors='raise')
                if not np.isfinite(n).all() or (n < 0).any() or (n % 1 != 0).any() or (n > 2**53 - 1).any():
                    raise ValueError(f'Expected finite nonnegative exact integer-valued data: {col}')
        c.index = pd.RangeIndex(2, len(c) + 2, name='source_row')
        self.raw_groups = c.groupby('SYNTHETIC_U', sort=True)
        # Conflicting IDs are never resolved by first/last row or fuzzy matching.
        canonical = c.drop_duplicates('SYNTHETIC_U').set_index('SYNTHETIC_U').copy()
        self.conflicts = {}
        for uid, rows in c[c.SYNTHETIC_U.duplicated(False)].groupby('SYNTHETIC_U'):
            conflicts = [col for col in c if col != 'SYNTHETIC_U' and rows[col].nunique(dropna=False) > 1]
            self.conflicts[uid] = conflicts
            for col in conflicts:
                canonical.loc[uid, col] = None
        self.c = canonical.sort_index()
        self.c['division'] = self.c.MAINOKVED_CODE.map(division)
        self.c['revenue_numeric'] = pd.to_numeric(self.c.REVENUE_COMPANY, errors='coerce')
        raw_target_ids = set(c.loc[pd.to_numeric(c.REVENUE_COMPANY) > 15_000_000_000, 'SYNTHETIC_U'])
        uncertain_targets = raw_target_ids & set(self.conflicts)
        if uncertain_targets:
            raise ValueError(f'Target IDs with conflicting raw records require adjudication: {sorted(uncertain_targets)}')
        self.targets = sorted(raw_target_ids)
        self.link_pairs = links[['SYNTHETIC_F', 'SYNTHETIC_U']].drop_duplicates()
        self.person_companies = self.link_pairs.groupby('SYNTHETIC_F').SYNTHETIC_U.agg(list).to_dict()
        self.company_people = self.link_pairs.groupby('SYNTHETIC_U').SYNTHETIC_F.agg(list).to_dict()
        self.edges = p.assign(PAY_AMT=pd.to_numeric(p.PAY_AMT).astype('int64'), PAY_CNT=pd.to_numeric(p.PAY_CNT).astype('int64')).groupby(['PAYER', 'BENEFICIARY'], sort=True).agg(PAY_AMT=('PAY_AMT', 'sum'), PAY_CNT=('PAY_CNT', 'sum'), source_rows=('PAY_AMT', 'size')).reset_index()
        e = self.edges
        pg, bg = e.PAYER.map(self.c.GROUP_ID), e.BENEFICIARY.map(self.c.GROUP_ID)
        known = pg.notna() & bg.notna() & pg.ne('') & bg.ne('')
        e['category'] = np.select([e.PAYER.eq(e.BENEFICIARY), known & pg.eq(bg), known], ['self', 'intragroup', 'external_confirmed'], default='group_unknown')
        e['payer_in_company_table'] = e.PAYER.isin(self.c.index)
        e['beneficiary_in_company_table'] = e.BENEFICIARY.isin(self.c.index)
        e['payer_division'] = e.PAYER.map(self.c.division).fillna('UNKNOWN')
        e['beneficiary_division'] = e.BENEFICIARY.map(self.c.division).fillna('UNKNOWN')
        self.incident_out = {uid: part for uid, part in e[e.PAYER.isin(self.targets)].groupby('PAYER')}
        self.incident_in = {uid: part for uid, part in e[e.BENEFICIARY.isin(self.targets)].groupby('BENEFICIARY')}
        self.group_members = self.c[self.c.GROUP_ID.notna() & self.c.GROUP_ID.ne('')].groupby('GROUP_ID').groups
        self.sector_members = self.c[self.c.division.notna()].groupby('division').groups
        self.manifest = {name: {'sha256': sha256(self.data_dir / name), 'bytes': (self.data_dir / name).stat().st_size, 'rows': len(df)} for name, df in self.frames.items()}

    def direction(self, uid, direction):
        outgoing = direction == 'outgoing'
        rows = (self.incident_out if outgoing else self.incident_in).get(uid, self.edges.iloc[:0])
        counterparty_col = 'BENEFICIARY' if outgoing else 'PAYER'
        sector_col = 'beneficiary_division' if outgoing else 'payer_division'
        totals = {cat: {'amount': int(rows.loc[rows.category.eq(cat), 'PAY_AMT'].sum()), 'count': int(rows.loc[rows.category.eq(cat), 'PAY_CNT'].sum()), 'edges': int(rows.category.eq(cat).sum())} for cat in CATEGORIES}
        total_amount, total_count = int(rows.PAY_AMT.sum()), int(rows.PAY_CNT.sum())
        for value in totals.values():
            value['share_of_all_amount'] = ratio(value['amount'], total_amount)
            value['share_of_all_count'] = ratio(value['count'], total_count)
        ext = rows[rows.category.eq('external_confirmed')]
        amount, count = int(ext.PAY_AMT.sum()), int(ext.PAY_CNT.sum())
        peers = ext.sort_values(['PAY_AMT', counterparty_col], ascending=[False, True])
        counterparties = []
        for row in peers.itertuples(index=False):
            cp = getattr(row, counterparty_col)
            counterparties.append({'company_id': cp, 'amount': row.PAY_AMT, 'count': row.PAY_CNT,
                                   'amount_share': ratio(row.PAY_AMT, amount), 'count_share': ratio(row.PAY_CNT, count),
                                   'division': getattr(row, sector_col), 'source_rows': row.source_rows})
        sectors = []
        for sector, part in ext.groupby(sector_col, sort=True):
            sectors.append({'division': sector, 'amount': int(part.PAY_AMT.sum()), 'count': int(part.PAY_CNT.sum()),
                            'amount_share': ratio(int(part.PAY_AMT.sum()), amount), 'count_share': ratio(int(part.PAY_CNT.sum()), count),
                            'counterparties': len(part)})
        sectors.sort(key=lambda x: (-x['amount'], x['division']))
        cp_known = rows[counterparty_col].isin(self.c.index)
        grouped_known = rows.category.ne('group_unknown')
        nonself = rows[rows.category.ne('self')].sort_values(['PAY_AMT', counterparty_col], ascending=[False, True])
        observed_peers = [{'company_id': r[counterparty_col], 'category': r['category'], 'amount': int(r['PAY_AMT']), 'count': int(r['PAY_CNT']), 'division': r[sector_col], 'source_rows': int(r['source_rows'])} for r in nonself.to_dict('records')]
        candidates = rows[rows.category.isin(['external_confirmed', 'group_unknown'])]
        candidate_amount, candidate_count = int(candidates.PAY_AMT.sum()), int(candidates.PAY_CNT.sum())
        candidate_sectors = [{'division': s, 'amount': int(part.PAY_AMT.sum()), 'count': int(part.PAY_CNT.sum()),
                              'amount_share': ratio(int(part.PAY_AMT.sum()), candidate_amount), 'count_share': ratio(int(part.PAY_CNT.sum()), candidate_count), 'counterparties': len(part)} for s, part in candidates.groupby(sector_col, sort=True)]
        return {
            'all_observed': {'amount': total_amount, 'count': total_count, 'edges': len(rows)},
            'categories': totals,
            'external_denominators': {'amount': amount, 'count': count, 'counterparties': len(ext), 'population': 'external_confirmed only'},
            'counterparty_metadata_amount_coverage_all': ratio(int(rows.loc[cp_known, 'PAY_AMT'].sum()), total_amount),
            'counterparty_metadata_edge_coverage_all': ratio(int(cp_known.sum()), len(rows)),
            'group_classification_amount_coverage_all': ratio(int(rows.loc[grouped_known, 'PAY_AMT'].sum()), total_amount),
            'sector_amount_coverage_external': ratio(int(ext.loc[ext[sector_col].ne('UNKNOWN'), 'PAY_AMT'].sum()), amount),
            'top_counterparties': counterparties[:10],
            'top_1_amount_share': counterparties[0]['amount_share'] if counterparties else None,
            'top_5_amount_share': ratio(sum(x['amount'] for x in counterparties[:5]), amount),
            'hhi_amount': sum(x['amount_share'] ** 2 for x in counterparties) if amount else None,
            'sector_exposures': sectors,
            'candidate_external_denominators': {'amount': candidate_amount, 'count': candidate_count, 'population': 'external_confirmed + group_unknown; unknown group can contain undiscovered intragroup flows'},
            'candidate_external_sector_exposures': sorted(candidate_sectors, key=lambda x: (-x['amount'], x['division'])),
            'candidate_external_known_sector_amount_share': ratio(int(candidates.loc[candidates[sector_col].ne('UNKNOWN'), 'PAY_AMT'].sum()), candidate_amount),
            'top_observed_nonself_counterparties': observed_peers[:10],
            'all_observed_nonself_counterparties_file': f'counterparties/{uid}_{direction}_observed.csv',
            'all_counterparties_file': f'counterparties/{uid}_{direction}.csv',
        }, counterparties

    def build_company_profile(self, company_id):
        if company_id not in self.targets:
            raise ValueError('company_id must be in the selected >15e9 cohort')
        row = self.c.loc[company_id]
        raw_rows = self.raw_groups.get_group(company_id)
        raw = raw_rows.iloc[0].to_dict()
        out, _ = self.direction(company_id, 'outgoing')
        inc, _ = self.direction(company_id, 'incoming')
        group_ids = sorted(self.group_members.get(row.GROUP_ID, []))
        group_data = self.c.loc[group_ids]
        sector_ids = self.sector_members.get(row.division, [])
        sector = self.c.loc[sector_ids]
        sector_revenue = float(sector.revenue_numeric.sum())
        people = sorted(self.company_people.get(company_id, []))
        peer_people = {}
        for person in people:
            for cp in self.person_companies[person]:
                if cp != company_id:
                    peer_people.setdefault(cp, []).append(person)
        peers = [{'company_id': cp, 'shared_person_ids': sorted(ids), 'shared_person_count': len(ids), 'in_company_table': cp in self.c.index} for cp, ids in sorted(peer_people.items())]
        source_row = int(raw_rows.index[0])
        completeness = sum(value not in ('', None) for value in raw.values()) / len(raw)
        gaps = list(GAPS)
        missing = [key for key, value in raw.items() if value in ('', None)]
        if missing:
            gaps.append('Пустые поля исходного профиля: ' + ', '.join(missing))
        if not out['all_observed']['edges'] or not inc['all_observed']['edges']:
            gaps.append('Одна или обе стороны платежного графа не наблюдаются; это не нулевой реальный оборот.')
        if not people:
            gaps.append('Связанные физлица в предоставленном файле не наблюдаются.')
        if raw['FTS_FLAG'] in ('1', '1.0') and not raw['FTS_COUNTRY_NAME']:
            gaps.append('FTS_FLAG=1, но страна отсутствует; не восстанавливаем страну предположением.')
        if raw['FTS_FLAG'] in ('0', '0.0') and raw['FTS_COUNTRY_NAME']:
            gaps.append('FTS_FLAG=0 при непустой стране ФТС; противоречие сохранено без исправления.')
        factors = [
            {'factor': 'industry_regulation_and_demand', 'basis': ['MAINOKVED_CODE', 'MAINOKVED_NAME'], 'rule': 'main_okved_context_v1', 'hypothesis': 'Новости по основному ОКВЭД могут менять условия деятельности; основной код не описывает весь бизнес.', 'strength': None},
            {'factor': 'regional_events', 'basis': ['REGION', 'CITY'], 'rule': 'registered_location_v1', 'hypothesis': 'События региона регистрации могут быть релевантны; производственная география неизвестна.', 'strength': None},
            {'factor': 'counterparty_industry_outgoing', 'basis': ['derived.payments.outgoing.sector_exposures', 'derived.payments.outgoing.candidate_external_sector_exposures'], 'rule': 'directed_external_payment_share_v1', 'hypothesis': 'Отраслевые события у получателей исходящих платежей могут быть значимы; доля платежей не является долей закупок. У candidate_external принадлежность части контрагентов к группе неизвестна.', 'strength': None},
            {'factor': 'counterparty_industry_incoming', 'basis': ['derived.payments.incoming.sector_exposures', 'derived.payments.incoming.candidate_external_sector_exposures'], 'rule': 'directed_external_payment_share_v1', 'hypothesis': 'Отраслевые события у отправителей входящих платежей могут быть значимы; доля платежей не является долей продаж. У candidate_external принадлежность части контрагентов к группе неизвестна.', 'strength': None},
            {'factor': 'group_events', 'basis': ['GROUP_ID'], 'rule': 'observed_group_members_v1', 'hypothesis': 'События наблюдаемых участников группы могут быть релевантны; контроль и консолидация неизвестны.', 'strength': None},
        ]
        for flag, factor, hypothesis in [
            ('CL_PRODUCT_LOA_MARK', 'interest_rates_credit', 'Кредитный флаг делает новости ставок кандидатом для анализа; плавающая ставка и объём долга неизвестны.'),
            ('CL_PRODUCT_DEP_MARK', 'interest_rates_deposits', 'Депозитный флаг делает ставки кандидатом для анализа; сроки и объём неизвестны.'),
            ('CL_PRODUCT_LEA_MARK', 'leasing', 'Лизинговый флаг позволяет рассматривать новости лизинга; предмет и условия неизвестны.'),
            ('CL_PRODUCT_GAR_MARK', 'guarantees', 'Флаг гарантий позволяет рассматривать регулирование гарантий; обязательства неизвестны.'),
            ('CL_PRODUCT_ZPP_MARK', 'employment', 'Зарплатный проект даёт повод рассматривать трудовое регулирование; фонд оплаты труда неизвестен.'),
            ('FACTORING_MARK', 'factoring', 'Факторинг даёт повод рассматривать условия оборотного финансирования; требования и объём неизвестны.'),
            ('FTS_FLAG', 'trade_customs_fx', 'Флаг ФТС даёт повод рассматривать таможенные и валютные новости; импорт/экспорт, валюта и величина эффекта неизвестны.'),
        ]:
            if raw.get(flag) in ('1', '1.0'):
                factors.append({'factor': factor, 'basis': [flag] + (['FTS_COUNTRY_NAME'] if flag == 'FTS_FLAG' else []), 'rule': 'observed_flag_candidate_v1', 'hypothesis': hypothesis, 'strength': None})
        derived = {
            'industry': {'okved_division': row.division, 'activities': [row.MAINOKVED_NAME], 'activities_completeness': 'main_okved_only'},
            'scale': {'annual_revenue_numeric': int(row.revenue_numeric), 'threshold': 15000000000, 'selection_unit_assumption': 'raw numeric values treated as RUB for selection; not confirmed by CSV', 'revenue_year': None},
            'observed_sector_position': {'population': 'unique company IDs with consensus OKVED division and revenue in this supplied sample', 'division': row.division,
                                         'companies': len(sector), 'companies_with_revenue': int(sector.revenue_numeric.notna().sum()), 'observed_revenue_sum': sector_revenue,
                                         'revenue_rank_competition': int((sector.revenue_numeric > row.revenue_numeric).sum()) + 1,
                                         'share_of_observed_sector_revenue': ratio(row.revenue_numeric, sector_revenue), 'real_market_share': None, 'real_market_position': None},
            'group': {'group_id': row.GROUP_ID, 'observed_member_ids': group_ids, 'observed_members': len(group_ids),
                      'observed_nonconsolidated_revenue_sum': float(group_data.revenue_numeric.sum()), 'members_with_revenue': int(group_data.revenue_numeric.notna().sum()), 'complete_group': None},
            'products': {key: (None if raw.get(key) in ('', None) else raw[key] in ('1', '1.0')) for key in PRODUCTS},
            'fts': {'flag': None if raw['FTS_FLAG'] == '' else raw['FTS_FLAG'] in ('1', '1.0'), 'reported_countries': sorted(set(x.strip() for x in raw['FTS_COUNTRY_NAME'].split(',') if x.strip())), 'trade_direction': None, 'trade_volume': None},
            'payments': {'amount_unit': None, 'currency': None, 'period': None, 'outgoing': out, 'incoming': inc},
            'person_links': {'person_ids': people, 'person_count': len(people), 'company_peers': peers, 'roles': None, 'source_rows': [int(i) + 2 for i in self.frames[LINKS].index[self.frames[LINKS].SYNTHETIC_U.eq(company_id)]]},
        }
        quality = {'raw_missing_fields': missing, 'raw_id_conflict': False, 'linked_people_observed': bool(people),
                   'score': {'value': round(100 * completeness, 2), 'range': [0, 100], 'name': 'raw_field_completeness', 'formula': '100 * nonempty raw columns / all raw columns', 'interpretation': 'Completeness only; not credit risk, news relevance or calibrated confidence'},
                   'critical_limitations': ['unknown monetary units/periods', 'unknown sample coverage', 'no market denominator']}
        return native({
            'schema_version': '1.0', 'company_id': company_id, 'selection': {'rule': 'REVENUE_COMPANY > 15000000000', 'scope': 'company, not group', 'cohort_size': len(self.targets)},
            'raw': raw, 'derived': derived, 'inferred': {'news_sensitivity_candidates': factors, 'effect_sign': None, 'effect_size': None, 'confidence': None},
            'provenance': {
                'raw': {key: {'kind': 'raw', 'file': COMPANIES, 'sha256': self.manifest[COMPANIES]['sha256'], 'source_row': source_row, 'column': key, 'empty_string_means': 'missing'} for key in raw},
                'derived': {
                    'industry': {'kind': 'derived', 'inputs': ['raw.MAINOKVED_CODE', 'raw.MAINOKVED_NAME'], 'method': 'first two valid OKVED digits; no full business taxonomy'},
                    'scale': {'kind': 'derived', 'inputs': ['raw.REVENUE_COMPANY'], 'method': 'numeric parsing and conditional raw-unit threshold'},
                    'observed_sector_position': {'kind': 'derived', 'file': COMPANIES, 'method': 'consensus unique IDs; same OKVED division; nonconsolidated revenue sum and competition rank'},
                    'group': {'kind': 'derived', 'file': COMPANIES, 'predicate': {'GROUP_ID': row.GROUP_ID}, 'method': 'consensus unique IDs; sum of available revenues; not consolidation'},
                    'products': {'kind': 'derived', 'inputs': PRODUCTS, 'method': '0/1 flag conversion; empty -> null'},
                    'fts': {'kind': 'derived', 'inputs': ['raw.FTS_FLAG', 'raw.FTS_COUNTRY_NAME'], 'method': 'split comma-separated labels; preserve names; no direction inference'},
                    'payments': {'kind': 'derived', 'file': PAYMENTS, 'sha256': self.manifest[PAYMENTS]['sha256'], 'predicate': {'PAYER_OR_BENEFICIARY': company_id}, 'method': 'sum input rows by ordered pair; retain exact duplicates; classify self first, then consensus groups; directional external_confirmed denominators'},
                    'person_links': {'kind': 'derived', 'file': LINKS, 'sha256': self.manifest[LINKS]['sha256'], 'method': 'unique person-company pairs; peers share >=1 person; roles unknown'},
                },
                'inferred': {'kind': 'inferred', 'method': 'unvalidated candidate rules v1; each factor includes its basis and rule; not bank expert opinion'},
                'quality': {'kind': 'derived', 'method': quality['score']['formula']},
            },
            'data_gaps': gaps, 'quality': quality,
        })

    def eda(self, out):
        out.mkdir(parents=True, exist_ok=True)
        c, p, links = (self.frames[name] for name in [COMPANIES, PAYMENTS, LINKS])
        report = {'tables': {name: table_eda(df) for name, df in self.frames.items()}, 'cohort_company_fields': table_eda(c[c.SYNTHETIC_U.isin(self.targets)]), 'manifest': self.manifest,
                  'selection': {'threshold_raw': 15000000000, 'comparison': 'strictly greater', 'company_count': len(self.targets), 'types': self.c.loc[self.targets].CL_TYPE.value_counts().to_dict(), 'group_count': self.c.loc[self.targets].GROUP_ID.nunique()},
                  'company_keys': {'unique_ids': len(self.c), 'duplicate_id_excess_rows': int(c.SYNTHETIC_U.duplicated().sum()), 'conflicting_ids': self.conflicts},
                  'links': {'unique_people': links.SYNTHETIC_F.nunique(), 'unique_company_ids': links.SYNTHETIC_U.nunique(), 'duplicate_pair_excess': int(links.duplicated(['SYNTHETIC_F', 'SYNTHETIC_U']).sum()),
                            'orphan_company_rows': int((~links.SYNTHETIC_U.isin(self.c.index)).sum()), 'orphan_unique_companies': links.loc[~links.SYNTHETIC_U.isin(self.c.index), 'SYNTHETIC_U'].nunique(),
                            'companies_in_catalog_with_person_links': len(set(links.SYNTHETIC_U) & set(self.c.index)), 'target_companies_with_person_links': len(set(links.SYNTHETIC_U) & set(self.targets)),
                            'people_with_conflicting_city': int((links.groupby('SYNTHETIC_F').CITY_NAME.nunique() > 1).sum()), 'people_with_conflicting_region': int((links.groupby('SYNTHETIC_F').RC_NAME.nunique() > 1).sum())},
                  'fts_consistency': {'flag_1_country_empty': int((c.FTS_FLAG.eq('1.0') & c.FTS_COUNTRY_NAME.eq('')).sum()), 'flag_0_country_present': int((c.FTS_FLAG.eq('0.0') & c.FTS_COUNTRY_NAME.ne('')).sum())},
                  'limitations': GAPS}
        for label, ids in [('all_rows', None), ('cohort', self.targets)]:
            frame = c if ids is None else c[c.SYNTHETIC_U.isin(ids)]
            for col in c.columns:
                if col not in ['SYNTHETIC_U', 'REVENUE_COMPANY']:
                    frame[col].value_counts(dropna=False).rename_axis('value').reset_index(name='rows').to_csv(out / f'{label}_{col}.csv', index=False, encoding='utf-8-sig')
        c[c.SYNTHETIC_U.duplicated(False)].to_csv(out / 'conflicting_company_rows.csv', encoding='utf-8-sig')
        p[p.duplicated(keep=False)].to_csv(out / 'exact_duplicate_payment_rows.csv', index_label='zero_based_dataframe_index', encoding='utf-8-sig')
        p[p.duplicated(['PAYER', 'BENEFICIARY'], False)].to_csv(out / 'repeated_ordered_pair_rows.csv.gz', index_label='zero_based_dataframe_index', compression='gzip')
        links[~links.SYNTHETIC_U.isin(self.c.index)].to_csv(out / 'orphan_person_company_links.csv', index=False, encoding='utf-8-sig')
        c[(c.FTS_FLAG.isin(['1', '1.0']) & c.FTS_COUNTRY_NAME.eq('')) | (c.FTS_FLAG.isin(['0', '0.0']) & c.FTS_COUNTRY_NAME.ne(''))].to_csv(out / 'fts_contradictions.csv', encoding='utf-8-sig')
        groups = self.c.groupby('GROUP_ID', dropna=False).agg(observed_companies=('revenue_numeric', 'size'), observed_revenue_sum_nonconsolidated=('revenue_numeric', 'sum'), known_revenues=('revenue_numeric', 'count'))
        groups.to_csv(out / 'groups.csv', encoding='utf-8-sig')
        self.c.groupby('division', dropna=False).agg(companies=('revenue_numeric', 'size'), revenue_sum=('revenue_numeric', 'sum'), revenue_count=('revenue_numeric', 'count')).to_csv(out / 'okved_divisions.csv', encoding='utf-8-sig')
        pd.concat([c.FTS_COUNTRY_NAME.str.split(',').explode().str.strip().value_counts().rename('all_rows'), c.loc[c.SYNTHETIC_U.isin(self.targets), 'FTS_COUNTRY_NAME'].str.split(',').explode().str.strip().value_counts().rename('cohort')], axis=1).fillna(0).to_csv(out / 'fts_countries.csv', encoding='utf-8-sig')
        for col in PRODUCTS + ['FTS_FLAG']:
            pd.crosstab(c.CL_TYPE, c[col]).to_csv(out / f'client_type_vs_{col}.csv', encoding='utf-8-sig')
        report['monetary_diagnostics'] = {'revenue_equal_1000_rows': int(c.REVENUE_COMPANY.eq('1000.0').sum()), 'payment_amount_multiple_1000_rows': int(pd.to_numeric(p.PAY_AMT).mod(1000).eq(0).sum()), 'units_confirmed': False, 'payment_period_confirmed': False}
        report['okved'] = {'invalid_syntax_rows': int(c.MAINOKVED_CODE.map(division).isna().sum()), 'codes_with_multiple_names': int((c.groupby('MAINOKVED_CODE').MAINOKVED_NAME.nunique() > 1).sum()), 'names_with_multiple_raw_codes': int((c.groupby('MAINOKVED_NAME').MAINOKVED_CODE.nunique() > 1).sum()), 'official_classification_validation': 'not performed; no official OKVED dictionary supplied'}
        e = self.edges
        nodes = set(e.PAYER) | set(e.BENEFICIARY)
        nonself = e[e.category.ne('self')]
        pairs = set(zip(nonself.PAYER, nonself.BENEFICIARY))
        report['graph'] = {'nodes': len(nodes), 'ordered_edges': len(e), 'input_rows': len(p), 'source_amount_total': int(pd.to_numeric(p.PAY_AMT).sum()), 'source_count_total': int(pd.to_numeric(p.PAY_CNT).sum()),
                           'repeated_pair_excess_rows': len(p) - len(e), 'catalog_nodes_observed': len(nodes & set(self.c.index)), 'nodes_outside_catalog': len(nodes - set(self.c.index)),
                           'target_nodes_observed': len(nodes & set(self.targets)), 'catalog_nodes_without_payments': len(set(self.c.index) - nodes),
                           'nonself_directed_density': ratio(len(nonself), len(nodes) * (len(nodes) - 1)), 'reciprocal_nonself_directed_edges': sum((b, a) in pairs for a, b in pairs),
                           'categories': records(e.groupby('category').agg(amount=('PAY_AMT', 'sum'), count=('PAY_CNT', 'sum'), edges=('PAY_AMT', 'size'), input_rows=('source_rows', 'sum')).reset_index()),
                           'metadata_coverage': records(e.groupby(['payer_in_company_table', 'beneficiary_in_company_table']).agg(amount=('PAY_AMT', 'sum'), count=('PAY_CNT', 'sum'), edges=('PAY_AMT', 'size')).reset_index()),
                           'duplicate_policy': 'retain all rows: no transaction identifier or period; do not assume duplicate aggregates are erroneous',
                           'drop_exact_duplicates_sensitivity': {'amount_difference': int(pd.to_numeric(p.loc[p.duplicated(), 'PAY_AMT']).sum()), 'count_difference': int(pd.to_numeric(p.loc[p.duplicated(), 'PAY_CNT']).sum())}}
        node_table = pd.DataFrame(index=sorted(nodes))
        for direction, key in [('outgoing', 'PAYER'), ('incoming', 'BENEFICIARY')]:
            node_table = node_table.join(e.groupby(key).agg(**{direction + '_edges': ('PAY_AMT', 'size'), direction + '_amount': ('PAY_AMT', 'sum'), direction + '_count': ('PAY_CNT', 'sum')}))
        node_table.fillna(0).to_csv(out / 'node_metrics.csv.gz', index_label='company_id', compression='gzip')
        report['graph']['invariants'] = {'amount_conserved': int(e.PAY_AMT.sum()) == report['graph']['source_amount_total'], 'count_conserved': int(e.PAY_CNT.sum()) == report['graph']['source_count_total'], 'source_rows_conserved': int(e.source_rows.sum()) == len(p)}
        write_json(out / 'eda.json', report)
        return report

    def run(self, output_dir):
        out = Path(output_dir)
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {out}')
        out.mkdir(parents=True)
        (out / 'profiles').mkdir()
        (out / 'counterparties').mkdir()
        report = self.eda(out / 'eda')
        self.edges.to_csv(out / 'payment_graph_all_categories.csv.gz', index=False, compression='gzip')
        self.edges[self.edges.category.eq('external_confirmed')].to_csv(out / 'payment_graph_external.csv.gz', index=False, compression='gzip')
        exposures = self.edges.groupby(['category', 'payer_division', 'beneficiary_division'], sort=True).agg(amount=('PAY_AMT', 'sum'), count=('PAY_CNT', 'sum'), edges=('PAY_AMT', 'size')).reset_index()
        for key, prefix in [('payer_division', 'outgoing'), ('beneficiary_division', 'incoming')]:
            for metric in ['amount', 'count']:
                den = exposures.groupby(['category', key])[metric].transform('sum')
                exposures[prefix + '_' + metric + '_denominator'] = den
                exposures[prefix + '_' + metric + '_share'] = exposures[metric] / den.replace(0, np.nan)
        exposures.to_csv(out / 'directed_sector_exposures.csv', index=False, encoding='utf-8-sig')
        registry = []
        with (out / 'company_profiles.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for uid in self.targets:
                profile = self.build_company_profile(uid)
                validate_profile(profile)
                write_json(out / 'profiles' / f'{uid}.json', profile)
                stream.write(json.dumps(profile, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                for direction in ['outgoing', 'incoming']:
                    _, counterparts = self.direction(uid, direction)
                    pd.DataFrame(counterparts, columns=['company_id', 'amount', 'count', 'amount_share', 'count_share', 'division', 'source_rows']).to_csv(out / 'counterparties' / f'{uid}_{direction}.csv', index=False, encoding='utf-8-sig')
                    outgoing = direction == 'outgoing'
                    observed = (self.incident_out if outgoing else self.incident_in).get(uid, self.edges.iloc[:0])
                    observed = observed[observed.category.ne('self')].sort_values(['PAY_AMT', 'BENEFICIARY' if outgoing else 'PAYER'], ascending=[False, True])
                    observed.to_csv(out / 'counterparties' / f'{uid}_{direction}_observed.csv', index=False, encoding='utf-8-sig')
                r, d = profile['raw'], profile['derived']
                registry.append({'company_id': uid, 'CL_TYPE': r['CL_TYPE'], 'revenue_raw_numeric': d['scale']['annual_revenue_numeric'], 'OKVED': r['MAINOKVED_CODE'], 'division': d['industry']['okved_division'], 'REGION': r['REGION'], 'GROUP_ID': r['GROUP_ID'], 'FTS_FLAG': r['FTS_FLAG'],
                                 'outgoing_external_amount': d['payments']['outgoing']['external_denominators']['amount'], 'incoming_external_amount': d['payments']['incoming']['external_denominators']['amount'],
                                 'outgoing_unknown_group_amount': d['payments']['outgoing']['categories']['group_unknown']['amount'], 'incoming_unknown_group_amount': d['payments']['incoming']['categories']['group_unknown']['amount'],
                                 'linked_people': d['person_links']['person_count'], 'raw_completeness_score': profile['quality']['score']['value'], 'profile_path': f'profiles/{uid}.json'})
        pd.DataFrame(registry).to_csv(out / 'company_registry.csv', index=False, encoding='utf-8-sig')
        write_json(out / 'manifest.json', {'schema_version': '1.0', 'inputs': self.manifest, 'target_ids': self.targets, 'pandas': pd.__version__, 'numpy': np.__version__, 'deterministic_semantics': True})
        self.write_report(out, report, registry)
        return report

    def write_report(self, out, report, registry):
        lines = ['# Этап 1: отчёт по фактическим файлам', '',
                 'Отбор: строго REVENUE_COMPANY > 15 000 000 000 в исходных числовых единицах. ТЗ говорит о рублях, но CSV не содержит масштаба и года; соответствие единиц требует подтверждения.', '',
                 f'Получено {len(self.targets)} профилей; типы: {report["selection"]["types"]}.', '',
                 '| Файл | Строки | Колонки | Полные дубли (избыточные строки) |', '|---|---:|---:|---:|']
        lines += [f'| {name} | {info["rows"]} | {info["columns"]} | {info["exact_duplicate_excess_rows"]} |' for name, info in report['tables'].items()]
        lines += ['', '## Ключи и связи', '', f'Уникальных компаний: {len(self.c)}. Конфликтующие ID: {json.dumps(self.conflicts, ensure_ascii=False)}.',
                  'Полные варианты конфликтующих записей сохранены; неоднозначные поля становятся null только в справочнике расчёта. Ни одна строка не подменяется выбранным первым вариантом.',
                  '', json.dumps(native(report['links']), ensure_ascii=False, indent=2), '', '## Платежный граф', '',
                  '| Категория | Объём в неизвестных единицах | Количество платежей | Направленных рёбер | Исходных строк |', '|---|---:|---:|---:|---:|']
        lines += [f'| {x["category"]} | {x["amount"]} | {x["count"]} | {x["edges"]} | {x["input_rows"]} |' for x in report['graph']['categories']]
        lines += ['', 'Граф: PAYER → BENEFICIARY. Self учитывается один раз в глобальном графе; в профиле он виден с обеих сторон, поэтому складывать incoming и outgoing как общий оборот нельзя.',
                  'Основной внешний граф использует только external_confirmed: разные непустые однозначные GROUP_ID. group_unknown хранится отдельно и не подмешивается во внешние знаменатели.',
                  'Полные дубли платежей сохранены: ключа операции и периода нет. В eda.json дано изменение сумм при их удалении. Глобальные категории образуют непересекающееся разбиение и сохраняют объём, число платежей и строки.',
                  '', f'Покрытие узлов: {report["graph"]["catalog_nodes_observed"]} из {len(self.c)} компаний справочника; узлов вне справочника: {report["graph"]["nodes_outside_catalog"]}.',
                  '', '## Реестр, распределения и качество', '',
                  'company_registry.csv содержит весь отобранный набор и ссылки на профили. eda/eda.json содержит EDA каждой колонки, числовые квантили, ключи, ФТС и покрытие графа. eda/all_rows_*.csv и eda/cohort_*.csv содержат полные распределения ОКВЭД, регионов, групп, продуктов и остальных категориальных полей. Групповые суммы не консолидированы.',
                  'Score — только доля заполненных исходных полей (включая идентификатор); это не риск, вероятность, уверенность или оценка релевантности новости. Покрытие платежей вынесено в отдельные метрики.',
                  '', '## Неразрешённые ограничения', ''] + ['- ' + gap for gap in GAPS]
        (out / 'EDA_REPORT_RU.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def validate_profile(profile):
    """Dependency-free semantic checks; published JSON Schema is the integration contract."""
    assert profile['schema_version'] == '1.0'
    assert profile['company_id'] == profile['raw']['SYNTHETIC_U']
    assert profile['derived']['scale']['annual_revenue_numeric'] > 15000000000
    assert set(profile['raw']) == set(profile['provenance']['raw'])
    assert set(profile['derived']) == set(profile['provenance']['derived'])
    assert 0 <= profile['quality']['score']['value'] <= 100
    for direction in ['outgoing', 'incoming']:
        d = profile['derived']['payments'][direction]
        for metric in ['amount', 'count', 'edges']:
            assert sum(d['categories'][cat][metric] for cat in CATEGORIES) == d['all_observed'][metric]
        for metric in ['amount', 'count']:
            denominator = d['external_denominators'][metric]
            assert sum(x[metric] for x in d['sector_exposures']) == denominator
            if denominator:
                assert math.isclose(sum(x[metric + '_share'] for x in d['sector_exposures']), 1.0, abs_tol=1e-12)
            else:
                assert not d['sector_exposures']
            candidate_denominator = d['candidate_external_denominators'][metric]
            assert sum(x[metric] for x in d['candidate_external_sector_exposures']) == candidate_denominator
            if candidate_denominator:
                assert math.isclose(sum(x[metric + '_share'] for x in d['candidate_external_sector_exposures']), 1.0, abs_tol=1e-12)
    json.dumps(profile, allow_nan=False)


@lru_cache(maxsize=1)
def load_batch(data_dir):
    return Batch(data_dir)


def build_company_profile(company_id, data_dir=None):
    """Public offline API. Load data once, then reuse the cached batch."""
    path = Path(data_dir) if data_dir is not None else Path(__file__).parent / 'data'
    return load_batch(str(path.resolve())).build_company_profile(company_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(Path(__file__).parent / 'data'))
    parser.add_argument('--out', default='results')
    parser.add_argument('--company-id')
    args = parser.parse_args()
    batch = Batch(args.data_dir)
    if args.company_id:
        profile = batch.build_company_profile(args.company_id)
        validate_profile(profile)
        target = Path(args.out)
        if target.exists():
            raise FileExistsError(target)
        write_json(target, profile)
    else:
        report = batch.run(args.out)
        print(json.dumps(native({'companies': len(batch.targets), 'types': report['selection']['types'], 'invariants': report['graph']['invariants']}), ensure_ascii=False))


if __name__ == '__main__':
    main()
