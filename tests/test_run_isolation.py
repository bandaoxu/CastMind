import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from castmind.config import ExperimentConfig, load_config, ABLATION_CHOICES, ablation_flags_dict
from castmind.eval import align_predictions
from castmind.run_layout import (
    resolve_experiment_paths, build_run_fingerprint, write_run_manifest,
    write_library_manifest, CASE_LIBRARY_CORE, find_matching_case_library,
    merge_legacy_fingerprint, upgrade_legacy_run_manifest, verify_resume_manifest,
    load_run_manifest,
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

    def _legacy_and_current(self):
        first = resolve_experiment_paths(self.cfg, **self.args)
        self.save(first)
        current = build_run_fingerprint(self.cfg, **self.args, lib_id=first.lib_id)
        current['case_library_dir'] = str(first.case_library_dir)
        # Build a pre-isolation legacy shape matching EPF_FR/Full_1.
        legacy = {
            k: current[k]
            for k in (
                'dataset', 'ablation', 'model', 'orchestration_mode',
                'train_sha256', 'test_sha256', 'target_column', 'look_back',
                'sliding_window', 'predicted_window', 'lib_id',
                'feature_selection', 'use_exogenous',
            )
        }
        legacy['ablation_flags'] = {
            **ablation_flags_dict(self.cfg),
            'use_feature_case_library': False,
        }
        legacy['prompt_hashes'] = {
            k: current['prompt_hashes'][k]
            for k in (
                'generator_agent.md',
                'investigator_agent.md',
                'knowledge_base/_shared.txt',
                'reflector_agent.md',
            )
            if k in current['prompt_hashes']
        }
        legacy['case_library_dir'] = str(first.case_library_dir)
        legacy['training_csv'] = self.args['training_csv']
        legacy['test_csv'] = self.args['test_csv']
        legacy['run_name'] = 'Full_1'
        legacy['created_at'] = '2026-10-08T00:00:00+00:00'
        write_run_manifest(first.run_dir, legacy)
        return first, legacy, current

    def test_legacy_manifest_upgrade_allows_resume(self):
        first, _legacy, current = self._legacy_and_current()
        with self.assertRaises(RuntimeError):
            verify_resume_manifest(load_run_manifest(first.run_dir), current)
        upgraded = upgrade_legacy_run_manifest(first.run_dir, current)
        self.assertIn('runtime_config', upgraded)
        self.assertIn('source_hashes', upgraded)
        self.assertNotIn('use_feature_case_library', upgraded['ablation_flags'])
        self.assertEqual(upgraded['prompt_hashes'], current['prompt_hashes'])
        self.assertTrue((first.run_dir / 'run_manifest.pre_upgrade.json').is_file())
        verify_resume_manifest(load_run_manifest(first.run_dir), current)
        self.cfg.resume = True
        resolved = resolve_experiment_paths(self.cfg, **self.args)
        self.assertEqual(resolved.run_dir, first.run_dir)

    def test_legacy_manifest_upgrade_refuses_mismatch(self):
        first, legacy, current = self._legacy_and_current()
        bad = dict(legacy)
        bad['train_sha256'] = '0' * 64
        write_run_manifest(first.run_dir, bad)
        with self.assertRaises(RuntimeError):
            merge_legacy_fingerprint(bad, current)
        bad2 = dict(legacy)
        ph = dict(bad2['prompt_hashes'])
        first_key = next(iter(ph))
        ph[first_key] = '1' * 64
        bad2['prompt_hashes'] = ph
        with self.assertRaises(RuntimeError):
            merge_legacy_fingerprint(bad2, current)

    def test_upgrade_fingerprint_must_match_dataset_filter(self):
        """Upgrade CLI must narrow cfg.datasets like run_experiment --dataset."""
        from castmind.config import DatasetConfig
        ds_a = DatasetConfig(
            name='EPF_FR', training_csv=str(self.root/'train.csv'),
            test_csv=str(self.root/'test.csv'), look_back=168,
            predicted_window=24, sliding_window=24, frequency='h',
        )
        ds_b = DatasetConfig(
            name='EPF_BE', training_csv=str(self.root/'train.csv'),
            test_csv=str(self.root/'test.csv'), look_back=168,
            predicted_window=24, sliding_window=24, frequency='h',
        )
        full_cfg = ExperimentConfig(
            datasets=[ds_a, ds_b], output_dir=str(self.root/'out'), run_name='Full_1',
        )
        args = dict(
            dataset_name='EPF_FR', training_csv=str(self.root/'train.csv'),
            test_csv=str(self.root/'test.csv'), target_column='OT',
            look_back=168, sliding_window=24, predicted_window=24, ablation_id='',
        )
        unfiltered = build_run_fingerprint(full_cfg, **args, lib_id='lib')
        filtered_cfg = ExperimentConfig(
            datasets=[ds_a], output_dir=str(self.root/'out'), run_name='Full_1',
        )
        filtered = build_run_fingerprint(filtered_cfg, **args, lib_id='lib')
        self.assertNotEqual(unfiltered['runtime_config'], filtered['runtime_config'])
        self.assertEqual(len(filtered['runtime_config']['datasets']), 1)
        self.assertEqual(filtered['runtime_config']['datasets'][0]['name'], 'EPF_FR')
