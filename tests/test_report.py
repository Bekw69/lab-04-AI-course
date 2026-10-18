"""Regression checks for report aggregation, independent of OCR and Qwen."""
import unittest

from lab04_core import arithmetic_checks, score_fields
from lab04_report import accuracy, arithmetic_summary, kazakh_confusions, submission_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.docs = [{'id': 'inv01_kk', 'lang': 'kk'}]
        self.truth = {'inv01_kk': {
            'seller_name': 'Қала',
            'items': [{'qty': 2, 'unit_price': 100, 'amount': 200}],
            'total': 200,
        }}

    def record(self, pred):
        truth = self.truth['inv01_kk']
        return {'pred': pred, 'norm': score_fields(truth, pred), 'arith': arithmetic_checks(pred)}

    def test_balanced_wrong_total_can_pass(self):
        pred = {'seller_name': 'Кала', 'items': [{'qty': 3, 'unit_price': 100, 'amount': 300}], 'total': 300}
        results = {('inv01_kk', 'clean'): self.record(pred)}
        stats = arithmetic_summary(self.docs, results, 'clean')
        self.assertEqual(stats['wrong_totals'], 1)
        self.assertEqual(stats['caught'], 0)
        self.assertEqual(stats['passed_with_errors'], 1)

    def test_correct_total_with_bad_row_is_flagged(self):
        pred = {'seller_name': 'Қала', 'items': [{'qty': 3, 'unit_price': 100, 'amount': 200}], 'total': 200}
        stats = arithmetic_summary(self.docs, {('inv01_kk', 'clean'): self.record(pred)}, 'clean')
        self.assertEqual(stats['right_totals'], 1)
        self.assertEqual(stats['right_flagged'], 1)
        self.assertEqual(stats['passed'], 0)

    def test_confusion_direction_is_truth_to_prediction(self):
        pred = dict(self.truth['inv01_kk'], seller_name='Кала')
        results = {('inv01_kk', c): self.record(pred) for c in ('clean', 'degraded')}
        self.assertEqual(kazakh_confusions(self.docs, self.truth, results), {('Қ', 'К'): 2})

    def test_report_rejects_incomplete_measurements(self):
        with self.assertRaises(ValueError):
            submission_report(self.docs, self.truth, {}, {}, {})

    def test_report_counts_are_measured(self):
        pred = dict(self.truth['inv01_kk'], seller_name='Кала')
        results = {('inv01_kk', c): self.record(pred) for c in ('clean', 'degraded')}
        ocr = {k: {'found': {p: True for p in r['norm']}} for k, r in results.items()}
        report = submission_report(self.docs, self.truth, ocr, results, results)
        self.assertIn('5/5 = 100.0%', report)
        self.assertIn('4/5 = 80.0%', report)
        self.assertIn('Қ (U+049A) → К (U+041A): 2', report)
        self.assertIn('прошли 1 документов, из них 1 с ошибкой', report)

    def test_wilson_at_boundaries(self):
        self.assertEqual(accuracy([]), 'нет данных')
        self.assertIn('[0.0; 79.3]', accuracy([False]))
        self.assertIn('[20.7; 100.0]', accuracy([True]))


if __name__ == '__main__':
    unittest.main()
