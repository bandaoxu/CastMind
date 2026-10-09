import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from castmind.config import ExperimentConfig, load_config, ABLATION_CHOICES
from castmind.eval import align_predictions
from castmind.run_layout import (
    resolve_experiment_paths, build_run_fingerprint, write_run_manifest,
    write_library_manifest, CASE_LIBRARY_CORE, find_matching_case_library,
)
from scripts.migrate_outputs_to_runs import _audit_predictions


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ['train.csv', 'test.csv']:
            (self.root/name).write_text('date,OT\n2020-01-01,1\n')
        self.cfg = ExperimentConfig(datasets=[], output_dir=str(self.root/'out'), run_name='Full_1')
        self.args = dict(dataset_name='ETTh1', training_csv=str(self.root/'train.csv'),
                         test_csv=str(self.root/'test.csv'), target_column='OT',
                         look_back=96, sliding_window=96, predicted_window=96, ablation_id='')

    def save(self, resolved):
        resolved.case_library_dir.mkdir(parents=True, exist_ok=True)
        for name in CASE_LIBRARY_CORE:
            (resolved.case_library_dir/name).write_text('{"test": 1}')
        lib_args = {k:v for k,v in self.args.items() if k not in ['dataset_name','test_csv','ablation_id']}
        write_library_manifest(resolved.case_library_dir, fingerprint=resolved.lib_id, **lib_args)
        manifest = build_run_fingerprint(self.cfg, **self.args, lib_id=resolved.lib_id)
        manifest['case_library_dir'] = str(resolved.case_library_dir)
        write_run_manifest(resolved.run_dir, manifest)

    def test_new_run_collision_and_isolation(self):
        first = resolve_experiment_paths(self.cfg, **self.args)
        with self.assertRaises(FileExistsError):
            resolve_experiment_paths(self.cfg, **self.args)
        self.cfg.run_name='Full_2'
        second = resolve_experiment_paths(self.cfg, **self.args)
        self.assertNotEqual(first.run_dir, second.run_dir)

    def test_resume_and_changed_config_or_data(self):
        first=resolve_experiment_paths(self.cfg, **self.args); self.save(first)
        self.cfg.resume=True
        self.assertEqual(resolve_experiment_paths(self.cfg, **self.args).run_dir,first.run_dir)
        self.cfg.use_reflector=False
        with self.assertRaises(RuntimeError): resolve_experiment_paths(self.cfg, **self.args)
        self.cfg.use_reflector=True
        (self.root/'test.csv').write_text('date,OT\n2020-01-01,2\n')
        with self.assertRaises(RuntimeError): resolve_experiment_paths(self.cfg, **self.args)

    def test_missing_manifest_and_changed_library(self):
        first=resolve_experiment_paths(self.cfg, **self.args)
        self.cfg.resume=True
        with self.assertRaises(RuntimeError): resolve_experiment_paths(self.cfg, **self.args)
        self.save(first)
        (first.case_library_dir/'memory.json').write_text('{"changed": true}')
        with self.assertRaises(RuntimeError): resolve_experiment_paths(self.cfg, **self.args)

    def test_legacy_library_not_certified_or_overwritten(self):
        first=resolve_experiment_paths(self.cfg, **self.args)
        first.case_library_dir.mkdir(parents=True)
        for name in CASE_LIBRARY_CORE: (first.case_library_dir/name).write_text('{"legacy": true}')
        self.cfg.run_name='Full_2'
        second=resolve_experiment_paths(self.cfg, **self.args)
        self.assertNotEqual(first.case_library_dir, second.case_library_dir)
        self.assertFalse((first.case_library_dir/'library_manifest.json').exists())

    def test_duplicate_scoring_refused(self):
        gt=pd.DataFrame({'date':['2020-01-01'], 'OT':[1]})
        pred=pd.DataFrame({'time_stamp':['2020-01-01']*2,'prediction':[1,2]})
        with self.assertRaises(ValueError): align_predictions(gt,pred)

    def test_migration_format_is_not_trust(self):
        path=self.root/'predictions.csv'
        path.write_text('time_stamp,prediction\n2020-01-01,1\n')
        self.assertEqual(_audit_predictions(path,{'n':1})['status'],'format_checked')

    def test_full_config_has_no_feature_case(self):
        cfg=load_config('config.yaml')
        self.assertTrue(cfg.use_case_library)
        self.assertFalse(hasattr(cfg,'use_feature_case_library'))
        self.assertNotIn('feature_case',ABLATION_CHOICES)
