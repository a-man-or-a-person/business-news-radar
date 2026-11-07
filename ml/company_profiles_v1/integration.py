"""Explicit bridge to the existing Go API and a separate LLM-safe batch context."""
from copy import deepcopy
from decimal import Decimal
from .contracts import PRODUCT_FIELDS


def llm_context(profile):
    """Do not pass person IDs, person locations or raw source link rows to a model."""
    result = deepcopy(profile)
    links = result['derived']['person_links']
    result['derived']['person_links'] = {
        'person_count': links['person_count'],
        'company_peers': [{k: value for k, value in peer.items() if k != 'shared_person_ids'} for peer in links['company_peers']],
        'roles': None,
    }
    result['provenance']['derived']['person_links'] = {'kind': 'derived', 'method': 'aggregate counts of shared links; no person-level data'}
    return result


def _go_base(profile, directions, evidence):
    raw, derived = profile['raw'], profile['derived']
    names = ['rko', 'guarantees', 'loans', 'leasing', 'deposits', 'payroll', 'factoring']
    return {
        'company_id': profile['company_id'], 'data_source': 'dataset', 'data_updated_at': None,
        'industry': {'okved_code': raw['MAINOKVED_CODE'] or None, 'okved_name': raw['MAINOKVED_NAME'] or None},
        'scale': {'annual_revenue_rub': format(Decimal(raw['REVENUE_COMPANY']), '.2f'), 'employees_fns': raw['AMOUNT_WORKERS_FNS_GRADATION'] or None, 'employees_spark': raw['AMOUNT_WORKERS_SPARK_GRADATION'] or None},
        'location': {'city': raw['CITY'] or None, 'region': raw['REGION'] or None},
        'bank_products': dict(zip(names, (derived['products'][key] for key in PRODUCT_FIELDS))),
        'foreign_trade': {'has_customs_data': derived['fts']['flag'], 'countries': derived['fts']['reported_countries']},
        'group': {'id': derived['group']['group_id'], 'known_companies_count': derived['group']['observed_members']},
        'counterparties': directions,
        'data_gaps': ['payments.period', 'company_profile_v1:raw monetary RUB unit evidence=' + evidence, 'company_profile_v1:Go semantics include intragroup; do not substitute for external_confirmed metrics'],
    }


def to_go_profile_with_edges(profile, edges, *, rub_units_evidence=None):
    """edges: complete pandas table payment_graph_all_categories.csv.gz."""
    if not isinstance(rub_units_evidence, str) or not rub_units_evidence.strip():
        raise ValueError('Unconfirmed RUB units: no Go API export generated')
    uid = profile['company_id']
    directions = {}
    for direction, key, sector in [('outgoing', 'PAYER', 'beneficiary_division'), ('incoming', 'BENEFICIARY', 'payer_division')]:
        rows = edges[edges[key].eq(uid) & edges.PAYER.ne(edges.BENEFICIARY)]
        total = int(rows.PAY_AMT.sum())
        known = rows[rows[sector].ne('UNKNOWN')]
        industries = known.groupby(sector).PAY_AMT.sum().reset_index().sort_values(['PAY_AMT', sector], ascending=[False, True])
        directions[direction] = {
            'counterparties_count': len(rows), 'known_industry_count': len(known), 'total_amount_rub': str(total),
            'known_industry_payment_share': int(known.PAY_AMT.sum()) / total if total else None,
            'top_industries': [{'okved_class': r[sector], 'amount_rub': str(int(r['PAY_AMT'])), 'payment_share': int(r['PAY_AMT']) / total} for r in industries.head(5).to_dict('records')] if total else [],
        }
    return _go_base(profile, directions, rub_units_evidence)
