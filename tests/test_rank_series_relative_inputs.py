"""Rank-series path handling; runner and validation are mocked, no solve."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cylinder_reference_case import run_reference_mvi_series


class RankSeriesRelativeInputsTest(TestCase):
    def test_time_tables_keep_original_case_base_and_caller_is_unchanged(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            input_dir=root/"input";input_dir.mkdir()
            for name in ("load.csv","left.csv","electric.csv"):
                (input_dir/name).write_text("time_s,factor\n0,0\n1,1\n")
            case={"method":"MVI","output_dir":"../results","rank":1,
                  "dynamics":{"time_law":{"kind":"TABULATED","csv_path":"load.csv"}},
                  "boundary_conditions":{"variational":{
                      "mechanical":{"left":{"time_law":{"kind":"TABULATED","csv_path":"left.csv"}}},
                      "electrical":{"inner":{"time_law":{"kind":"TABULATED","csv_path":"electric.csv"}}}}}}
            original=copy.deepcopy(case)
            case_path=input_dir/"case.json";case_path.write_text(json.dumps(case))
            seen=[]
            def validate(candidate,base_dir):
                self.assertEqual(Path(base_dir),input_dir)
                self.assertEqual(candidate["dynamics"]["time_law"]["csv_path"],"load.csv")
            def fake_runner(candidate,path):
                bc=candidate["boundary_conditions"]["variational"]
                actual=[candidate["dynamics"]["time_law"]["csv_path"],
                        bc["mechanical"]["left"]["time_law"]["csv_path"],
                        bc["electrical"]["inner"]["time_law"]["csv_path"]]
                self.assertEqual(actual,[str((input_dir/name).resolve()) for name in ("load.csv","left.csv","electric.csv")])
                self.assertEqual(json.loads(Path(path).read_text()),candidate)
                seen.append(candidate["rank"])
                return {"converged":True,"rank":candidate["rank"]}
            with patch("cylinder_reference_case.prepare_reference_case",side_effect=validate),\
                 patch("cylinder_reference_case.run_reference_case",side_effect=fake_runner):
                result=run_reference_mvi_series(case,case_path,ranks=(1,2))
            self.assertEqual(seen,[1,2])
            self.assertTrue(result["all_converged"])
            self.assertEqual(case,original)
