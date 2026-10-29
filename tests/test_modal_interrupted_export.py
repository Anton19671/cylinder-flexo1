"""Interrupted runs retain real data and export their available files."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cylinder_stop_archive import archive_stopped_run, archive_failed_run
from cylinder_reference_export import export


class InterruptedExportTests(unittest.TestCase):
    def test_full_export_on_stop_and_failure_without_completed_modes(self):
        for status in ('stop', 'failure'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'existing.xlsx').write_bytes(b'existing spreadsheet')
                archive = archive_stopped_run if status == 'stop' else archive_failed_run
                archive(root, calculation_kind='MODAL', method='MVI', rank=1, expected_steps=6)
                result = export(root, 'FULL')
                self.assertEqual((result/'FILES'/'existing.xlsx').read_bytes(), b'existing spreadsheet')
                self.assertTrue((result/'RUN_STATUS.xlsx').is_file())
                self.assertFalse((result/'FREQUENCIES.csv').exists())

    def test_partial_modes_are_counted_and_exported_without_fabrication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'REFERENCE_CASE.json').write_text(json.dumps({'theory':'CUF','cuf_model':'TE1'}))
            np.savez_compressed(root/'MODAL_PARTIAL_MODES.npz',
                frequencies_hz=np.array([42.]), omega_rad_s=np.array([2*np.pi*42]),
                omega_squared_rad2_s2=np.array([(2*np.pi*42)**2]),
                mechanical_modes=np.zeros((1,6,2,3)), x_m=np.array([0.,1.]), s_m=np.arange(3.))
            record=archive_stopped_run(root, calculation_kind='MODAL', method='MVI', rank=1, expected_steps=6)
            self.assertEqual(record['accepted_steps_verified'], 1)
            result=export(root,'FAST')
            self.assertTrue((result/'MODAL_PARTIAL.xlsx').exists())
            lines=(result/'FREQUENCIES.csv').read_text(encoding='utf-8-sig').splitlines()
            self.assertEqual(len(lines),2)
            self.assertIn('42.0',lines[1])


if __name__ == '__main__': unittest.main()
