"""Crash recovery must preserve finished inference and reject corrupt scores."""

import json
from pathlib import Path
import tempfile
import unittest

from lab04_checkpoint import atomic_json, load_records, recover_raw, run_lock, score_record


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.truths = {'d': {'total': 20}}

    def test_reload_keeps_the_measured_raw_answer(self):
        row = score_record('d', 'clean', '{"total":20}', self.truths['d'], seconds=12)
        atomic_json(self.root/'vlm.json', [row])
        loaded = load_records(self.root/'vlm.json', self.truths, {('d','clean')})
        self.assertEqual(loaded[('d','clean')], row)
        self.assertFalse((self.root/'vlm.json.tmp').exists())

    def test_modified_score_is_rejected(self):
        row = score_record('d', 'clean', '{"total":19}', self.truths['d'])
        row['norm']['total'] = True
        atomic_json(self.root/'vlm.json', [row])
        with self.assertRaisesRegex(ValueError, 'inconsistent norm'):
            load_records(self.root/'vlm.json', self.truths)

    def test_finished_raw_journal_recovers_without_model_call(self):
        atomic_json(self.root/'raw'/'d_clean.json', {
            'doc_id':'d','condition':'clean','raw':'{"total":20}', 'seconds':12,
            'prompt':'instruction','image_sha256':'hash','finished_at_utc':'measured-time',
        })
        manifest = {'docs':[{'id':'d','sha256':{'clean':'hash'}}]}
        rows = recover_raw(self.root/'raw', {}, self.truths, {('d','clean')}, 'instruction', manifest)
        self.assertTrue(rows[('d','clean')]['norm']['total'])
        self.assertTrue(rows[('d','clean')]['recovered_from_raw'])

    def test_lock_prevents_a_second_worker(self):
        path = self.root/'inference.lock'
        with run_lock(path):
            with self.assertRaisesRegex(RuntimeError, 'Another Lab 04'):
                with run_lock(path):
                    self.fail('Second worker obtained the lock')
        with run_lock(path):
            pass


if __name__ == '__main__':
    unittest.main()
