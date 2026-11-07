"""Published JSON Schema and a validator for the small schema vocabulary used here."""
import json
import re
from copy import deepcopy
from pathlib import Path


def obj(properties):
    return {'type': 'object', 'required': list(properties), 'properties': properties, 'additionalProperties': False}


def array(items):
    return {'type': 'array', 'items': items}


S = {'type': 'string'}
I = {'type': 'integer', 'minimum': 0}
N = {'type': 'number', 'minimum': 0}
NULL = {'type': 'null'}
NS = {'type': ['string', 'null']}
B = {'type': 'boolean'}
NB = {'type': ['boolean', 'null']}
SHARE = {'type': ['number', 'null'], 'minimum': 0, 'maximum': 1}
RAW_FIELDS = 'SYNTHETIC_U ORG_TYPE OKFS_NAME COMPANY_AGE_YEARS_GRADATION ECO_MARK_LABEL MAINOKVED_CODE MAINOKVED_NAME NEGATIVEMARK AMOUNT_WORKERS_FNS_GRADATION AMOUNT_WORKERS_SPARK_GRADATION CL_TYPE CL_PRODUCT_RKO_MARK CL_PRODUCT_GAR_MARK CL_PRODUCT_LOA_MARK CL_PRODUCT_LEA_MARK CL_PRODUCT_DEP_MARK CL_PRODUCT_ZPP_MARK FTS_FLAG FTS_COUNTRY_NAME FACTORING_MARK REVENUE_COMPANY GROUP_ID CITY REGION'.split()
PRODUCT_FIELDS = ['CL_PRODUCT_' + x + '_MARK' for x in ['RKO', 'GAR', 'LOA', 'LEA', 'DEP', 'ZPP']] + ['FACTORING_MARK']
CATEGORIES = ['self', 'intragroup', 'external_confirmed', 'group_unknown']
SECTOR = obj({'division': S, 'amount': I, 'count': I, 'amount_share': SHARE, 'count_share': SHARE, 'counterparties': I})
PEER = obj({'company_id': S, 'amount': I, 'count': I, 'amount_share': SHARE, 'count_share': SHARE, 'division': S, 'source_rows': I})
OBSERVED_PEER = obj({'company_id': S, 'category': {'type': 'string', 'enum': CATEGORIES}, 'amount': I, 'count': I, 'division': S, 'source_rows': I})
CATEGORY = obj({'amount': I, 'count': I, 'edges': I, 'share_of_all_amount': SHARE, 'share_of_all_count': SHARE})
DIRECTION = obj({
    'all_observed': obj({'amount': I, 'count': I, 'edges': I}), 'categories': obj({key: CATEGORY for key in CATEGORIES}),
    'external_denominators': obj({'amount': I, 'count': I, 'counterparties': I, 'population': S}),
    'counterparty_metadata_amount_coverage_all': SHARE, 'counterparty_metadata_edge_coverage_all': SHARE,
    'group_classification_amount_coverage_all': SHARE, 'sector_amount_coverage_external': SHARE,
    'top_counterparties': array(PEER), 'top_1_amount_share': SHARE, 'top_5_amount_share': SHARE, 'hhi_amount': SHARE,
    'sector_exposures': array(SECTOR), 'all_counterparties_file': S,
    'candidate_external_denominators': obj({'amount': I, 'count': I, 'population': S}), 'candidate_external_sector_exposures': array(SECTOR),
    'candidate_external_known_sector_amount_share': SHARE, 'top_observed_nonself_counterparties': array(OBSERVED_PEER), 'all_observed_nonself_counterparties_file': S,
})
DERIVED = obj({
    'industry': obj({'okved_division': NS, 'activities': array(S), 'activities_completeness': S}),
    'scale': obj({'annual_revenue_numeric': I, 'threshold': {'const': 15000000000}, 'selection_unit_assumption': S, 'revenue_year': NULL}),
    'observed_sector_position': obj({'population': S, 'division': NS, 'companies': I, 'companies_with_revenue': I, 'observed_revenue_sum': N, 'revenue_rank_competition': I, 'share_of_observed_sector_revenue': SHARE, 'real_market_share': NULL, 'real_market_position': NULL}),
    'group': obj({'group_id': NS, 'observed_member_ids': array(S), 'observed_members': I, 'observed_nonconsolidated_revenue_sum': N, 'members_with_revenue': I, 'complete_group': NULL}),
    'products': obj({key: NB for key in PRODUCT_FIELDS}),
    'fts': obj({'flag': NB, 'reported_countries': array(S), 'trade_direction': NULL, 'trade_volume': NULL}),
    'payments': obj({'amount_unit': NULL, 'currency': NULL, 'period': NULL, 'outgoing': DIRECTION, 'incoming': DIRECTION}),
    'person_links': obj({'person_ids': array(S), 'person_count': I, 'company_peers': array(obj({'company_id': S, 'shared_person_ids': array(S), 'shared_person_count': I, 'in_company_table': B})), 'roles': NULL, 'source_rows': array(I)}),
})
FACTOR = obj({'factor': S, 'basis': array(S), 'rule': S, 'hypothesis': S, 'strength': NULL})
RAW_PROVENANCE = obj({'kind': {'const': 'raw'}, 'file': S, 'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}, 'source_row': {'type': 'integer', 'minimum': 2}, 'column': S, 'empty_string_means': S})
SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    '$id': 'urn:alfa-hack:offline-company-profile:1.0',
    'title': 'Offline CompanyProfile v1.0 (not the Go HTTP CompanyProfile)',
    'description': 'raw/derived/inferred provenance is inherited by descendants; units and period remain unknown. Separate LLM context omits person-level information.',
    **obj({
        'schema_version': {'const': '1.0'}, 'company_id': {'type': 'string', 'pattern': '^U_[0-9]+$'},
        'selection': obj({'rule': S, 'scope': S, 'cohort_size': I}), 'raw': obj({key: S for key in RAW_FIELDS}), 'derived': DERIVED,
        'inferred': obj({'news_sensitivity_candidates': array(FACTOR), 'effect_sign': NULL, 'effect_size': NULL, 'confidence': NULL}),
        'provenance': obj({'raw': obj({key: RAW_PROVENANCE for key in RAW_FIELDS}), 'derived': obj({key: {'type': 'object', 'required': ['kind', 'method'], 'properties': {'kind': {'const': 'derived'}, 'method': S}} for key in DERIVED['properties']}),
                           'inferred': obj({'kind': {'const': 'inferred'}, 'method': S}), 'quality': obj({'kind': {'const': 'derived'}, 'method': S})}),
        'data_gaps': array(S), 'quality': obj({'raw_missing_fields': array(S), 'raw_id_conflict': B, 'linked_people_observed': B,
                                           'score': obj({'value': {'type': 'number', 'minimum': 0, 'maximum': 100}, 'range': array(I), 'name': S, 'formula': S, 'interpretation': S}), 'critical_limitations': array(S)}),
    }),
}

LLM_SCHEMA = deepcopy(SCHEMA)
LLM_SCHEMA['$id'] = 'urn:alfa-hack:llm-company-context:1.0'
LLM_SCHEMA['title'] = 'CompanyProfile v1.0 LLM context, person-level data removed'
LLM_SCHEMA['properties']['derived']['properties']['person_links'] = obj({
    'person_count': I, 'company_peers': array(obj({'company_id': S, 'shared_person_count': I, 'in_company_table': B})), 'roles': NULL,
})


def validate(value, schema=SCHEMA, path='$'):
    """Only validates the schema keywords present in SCHEMA; not a general library."""
    types = schema.get('type')
    checks = {'null': lambda x: x is None, 'object': lambda x: isinstance(x, dict), 'array': lambda x: isinstance(x, list),
              'string': lambda x: isinstance(x, str), 'boolean': lambda x: isinstance(x, bool), 'integer': lambda x: type(x) is int, 'number': lambda x: type(x) in (int, float)}
    if types and not any(checks[t](value) for t in ([types] if isinstance(types, str) else types)):
        raise ValueError(f'{path}: expected {types}, got {type(value).__name__}')
    if 'const' in schema and value != schema['const']:
        raise ValueError(f'{path}: wrong constant')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: invalid enum')
    if value is None:
        return
    if type(value) in (int, float):
        if 'minimum' in schema and value < schema['minimum'] or 'maximum' in schema and value > schema['maximum']:
            raise ValueError(f'{path}: outside bounds')
    if isinstance(value, str) and 'pattern' in schema and not re.search(schema['pattern'], value):
        raise ValueError(f'{path}: wrong pattern')
    if isinstance(value, dict):
        missing = set(schema.get('required', [])) - set(value)
        extra = set(value) - set(schema.get('properties', {}))
        if missing or (extra and schema.get('additionalProperties') is False):
            raise ValueError(f'{path}: missing={missing}, extra={extra}')
        for key, subschema in schema.get('properties', {}).items():
            if key in value:
                validate(value[key], subschema, path + '.' + key)
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate(item, schema.get('items', {}), f'{path}[{index}]')


if __name__ == '__main__':
    Path(__file__).with_name('company_profile_v1.schema.json').write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    Path(__file__).with_name('llm_context_v1.schema.json').write_text(json.dumps(LLM_SCHEMA, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
