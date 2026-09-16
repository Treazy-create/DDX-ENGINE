import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preprocess', ROOT / 'scripts/preprocess_medication_reference.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceTests(unittest.TestCase):
    def test_all_conditions_are_draft(self):
        data = module.preprocess((ROOT / 'data/reference/condition_medication_reference.txt').read_text())
        self.assertEqual(len(data['conditions']), 30)
        self.assertTrue(all(not row['recommendation_enabled'] for row in data['associations']))
        self.assertTrue(all(row['review_status'] == 'unverified' for row in data['associations']))

    def test_shared_ingredient_keeps_condition_specific_names(self):
        data = module.preprocess('[A]\nX | Brand1\n[B]\nx | Brand2')
        self.assertEqual(len(data['medicines']), 1)
        self.assertEqual(data['associations'][0]['supplied_names'], ['Brand1'])
        self.assertEqual(data['associations'][1]['supplied_names'], ['Brand2'])

    def test_bad_data_fails_instead_of_silently_disappearing(self):
        for source in ('', 'X | Y', '[A]\nX | ', '[A]\nX | Y\nX | Z'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                module.preprocess(source)


if __name__ == '__main__':
    unittest.main()
