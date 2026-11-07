import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.company_profiles_v1.pipeline import Batch, COMPANIES, PAYMENTS, LINKS, PRODUCTS, validate_profile
from ml.company_profiles_v1.integration import llm_context, to_go_profile_with_edges
from ml.company_profiles_v1.contracts import validate, LLM_SCHEMA
from ml.company_profiles_v1.render_company_profiles import render


def fixture(path):
    base = {'SYNTHETIC_U': 'U_1', 'ORG_TYPE': 'КОММ_ОРГ', 'OKFS_NAME': 'ЧАСТНАЯ СОБСТВЕННОСТЬ', 'COMPANY_AGE_YEARS_GRADATION': '8-15', 'ECO_MARK_LABEL': '',
            'MAINOKVED_CODE': '24.10', 'MAINOKVED_NAME': 'Отрасль', 'NEGATIVEMARK': '0.0', 'AMOUNT_WORKERS_FNS_GRADATION': 'Более 251', 'AMOUNT_WORKERS_SPARK_GRADATION': '', 'CL_TYPE': 'CLIENT',
            'FTS_FLAG': '1.0', 'FTS_COUNTRY_NAME': 'КИТАЙ, ТУРЦИЯ', 'REVENUE_COMPANY': '20000000000.0', 'GROUP_ID': 'group_a', 'CITY': 'Город', 'REGION': 'Регион'}
    base.update({key: '0.0' for key in PRODUCTS})
    companies = [base, dict(base, SYNTHETIC_U='U_2', GROUP_ID='group_b', MAINOKVED_CODE='46.1'),
                 dict(base, SYNTHETIC_U='U_3', REVENUE_COMPANY='15000000000.0'),
                 dict(base, SYNTHETIC_U='U_4', REVENUE_COMPANY='1000.0', GROUP_ID='group_c'),
                 dict(base, SYNTHETIC_U='U_4', REVENUE_COMPANY='1000.0', GROUP_ID='group_d', MAINOKVED_CODE='52.1'),
                 dict(base, SYNTHETIC_U='U_5', GROUP_ID='group_e')]
    pd.DataFrame(companies).to_csv(path / COMPANIES, sep=';', index=False)
    pd.DataFrame([
        ['U_1', 'U_1', 100, 1], ['U_1', 'U_3', 200, 2], ['U_1', 'U_2', 300, 3], ['U_1', 'U_2', 300, 3],
        ['U_2', 'U_1', 600, 6], ['U_1', 'U_999', 400, 4], ['U_998', 'U_1', 500, 5], ['U_1', 'U_4', 50, 1],
    ], columns=['PAYER', 'BENEFICIARY', 'PAY_AMT', 'PAY_CNT']).to_csv(path / PAYMENTS, sep=';', index=False)
    pd.DataFrame([['F_1', 'x', 'y', 'U_1'], ['F_1', 'x', 'y', 'U_2'], ['F_2', '', '', 'U_999']], columns=['SYNTHETIC_F', 'CITY_NAME', 'RC_NAME', 'SYNTHETIC_U']).to_csv(path / LINKS, sep=';', index=False)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.path = Path(self.tmp.name)
        assert self.path.resolve().is_relative_to(Path(__file__).resolve().parent)
        fixture(self.path)
        self.batch = Batch(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_direction_categories_duplicates_and_denominators(self):
        p = self.batch.build_company_profile('U_1')
        validate_profile(p)
        validate(p)
        out = p['derived']['payments']['outgoing']
        self.assertEqual(out['all_observed']['amount'], 1350)
        self.assertEqual({c: out['categories'][c]['amount'] for c in out['categories']}, {'self': 100, 'intragroup': 200, 'external_confirmed': 600, 'group_unknown': 450})
        self.assertEqual(out['external_denominators']['count'], 6)
        self.assertEqual(out['candidate_external_denominators']['amount'], 1050)
        self.assertAlmostEqual(out['candidate_external_known_sector_amount_share'], 600 / 1050)
        self.assertEqual(p['derived']['payments']['incoming']['external_denominators']['amount'], 600)
        self.assertEqual(out['top_counterparties'][0]['company_id'], 'U_2')
        self.assertEqual(out['top_counterparties'][0]['source_rows'], 2)
        self.assertEqual(self.batch.targets, ['U_1', 'U_2', 'U_5'])
        self.assertTrue(pd.isna(self.batch.c.loc['U_4', 'division']))

    def test_missing_direction_and_non_target(self):
        p = self.batch.build_company_profile('U_5')
        validate_profile(p)
        validate(p)
        self.assertIsNone(p['derived']['payments']['outgoing']['hhi_amount'])
        with self.assertRaises(ValueError):
            self.batch.build_company_profile('U_3')

    def test_schema_rejects_wrong_types_and_missing_fields(self):
        profile = self.batch.build_company_profile('U_1')
        profile['derived']['payments']['outgoing']['all_observed']['amount'] = '1350'
        with self.assertRaises(ValueError):
            validate(profile)
        profile = self.batch.build_company_profile('U_1')
        del profile['raw']['SYNTHETIC_U']
        with self.assertRaises(ValueError):
            validate(profile)

    def test_readable_catalogue_keeps_company_data_and_hides_people(self):
        profile = self.batch.build_company_profile('U_1')
        text = render([profile])
        self.assertEqual(text.count('## U_1\n'), 1)
        self.assertIn('[U_1](#u_1)', text)
        self.assertIn('group_unknown', text)
        self.assertIn('MAINOKVED_CODE', text)
        self.assertIn('не подтверждены', text)
        self.assertNotIn('F_1', text)

    def test_links_and_safe_integration(self):
        p = self.batch.build_company_profile('U_1')
        self.assertEqual(p['derived']['person_links']['company_peers'][0]['company_id'], 'U_2')
        context = llm_context(p)
        validate(context, LLM_SCHEMA)
        self.assertNotIn('F_1', json.dumps(context))
        self.assertIn('F_1', json.dumps(p))
        with self.assertRaises(ValueError):
            to_go_profile_with_edges(p, self.batch.edges)
        api = to_go_profile_with_edges(p, self.batch.edges, rub_units_evidence='Synthetic test fixture units, not a production confirmation')
        self.assertEqual(api['counterparties']['outgoing']['total_amount_rub'], '1250')
        self.assertAlmostEqual(api['counterparties']['outgoing']['known_industry_payment_share'], 800 / 1250)
        self.assertEqual(api['scale']['annual_revenue_rub'], '20000000000.00')
        reordered = json.loads(json.dumps(p, sort_keys=True))
        api_reordered = to_go_profile_with_edges(reordered, self.batch.edges, rub_units_evidence='Synthetic fixture')
        self.assertEqual(api['bank_products'], api_reordered['bank_products'])

    def test_boundary_rejects_negative_and_bad_ids(self):
        p = pd.read_csv(self.path / PAYMENTS, sep=';')
        p.loc[0, 'PAY_AMT'] = -1
        p.to_csv(self.path / PAYMENTS, sep=';', index=False)
        with self.assertRaises(ValueError):
            Batch(self.path)
        fixture(self.path)
        p = pd.read_csv(self.path / PAYMENTS, sep=';')
        p.loc[0, 'PAYER'] = 'real name'
        p.to_csv(self.path / PAYMENTS, sep=';', index=False)
        with self.assertRaises(ValueError):
            Batch(self.path)

    def test_batch_exports_and_refuses_overwrite(self):
        out = self.path / 'results'
        report = self.batch.run(out)
        self.assertTrue(all(report['graph']['invariants'].values()))
        self.assertEqual(len(list((out / 'profiles').glob('*.json'))), 3)
        self.assertEqual(report['graph']['drop_exact_duplicates_sensitivity']['amount_difference'], 300)
        with self.assertRaises(FileExistsError):
            self.batch.run(out)


if __name__ == '__main__':
    unittest.main()
