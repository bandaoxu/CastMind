import unittest

import pandas as pd

from castmind.prediction_segments import resolve_segment
from castmind.run_layout import replace_window_predictions
from castmind.eval import align_predictions


class SegmentTests(unittest.TestCase):
    def context(self, offset, start, length):
        anchor = pd.Timestamp('2017-08-21') + pd.Timedelta(hours=offset + start)
        return dict(dataset='ETTh1', window_offset=offset, horizon_start=start,
                    length=length, timestamps=pd.date_range(anchor, periods=length, freq='h').tolist())

    def chunk(self, offset, start, length, value=1):
        ts, index = resolve_segment(self.context(offset, start, length),
                                   dataset_name='ETTh1', window_offset=offset, length=length)
        return pd.DataFrame(dict(time_stamp=ts, window_offset=offset,
                                 horizon_index=range(index, index + length), prediction=value))

    def test_two_halves_tail_and_odd_horizon(self):
        for total in [96, 48, 5]:
            with self.subTest(total=total):
                first = total // 2
                a = self.chunk(0, 0, first)
                b = self.chunk(0, first, total-first, 2)
                combined = replace_window_predictions(a, b, window_offset=0, horizon_start=first)
                self.assertEqual(combined.horizon_index.tolist(), list(range(total)))
                self.assertFalse(combined.time_stamp.duplicated().any())
                gt = pd.DataFrame({'date': combined.time_stamp, 'OT': combined.prediction})
                actual, pred = align_predictions(gt, combined, 'ETTh1')
                self.assertTrue((actual == pred).all())

    def test_retry_replaces_segment_and_preserves_other_windows(self):
        old = pd.concat([self.chunk(0, 0, 96), self.chunk(96, 0, 48)])
        b = self.chunk(0, 48, 48, 3)
        for _ in range(2):
            old = replace_window_predictions(old, b, window_offset=0, horizon_start=48)
        self.assertEqual(len(old), 144)
        self.assertEqual(old[old.window_offset == 96].prediction.tolist(), [1]*48)
        self.assertEqual(old[(old.window_offset == 0) & (old.horizon_index < 48)].prediction.tolist(), [1]*48)
        self.assertEqual(old[(old.window_offset == 0) & (old.horizon_index >= 48)].prediction.tolist(), [3]*48)
        restarted = replace_window_predictions(old, self.chunk(0, 0, 48, 4), window_offset=0)
        self.assertEqual(len(restarted[restarted.window_offset == 0]), 48)

    def test_reject_missing_prefix_or_segment_state(self):
        with self.assertRaises(ValueError):
            replace_window_predictions(self.chunk(0, 0, 47), self.chunk(0, 48, 48),
                                       window_offset=0, horizon_start=48)
        for context in [None, self.context(96, 48, 48), self.context(0, 48, 24)]:
            with self.assertRaises(ValueError):
                resolve_segment(context, dataset_name='ETTh1', window_offset=0, length=48)

    def test_real_emitter_continuation_and_retry(self):
        import asyncio
        import tempfile
        from unittest.mock import patch
        from castmind.config import ExperimentConfig
        from castmind.agents.generator_agent import create_generator_agent

        class OfflineAgent:
            def __init__(self, *args, **kwargs):
                self.tools = {}
            def tool(self, fn):
                self.tools[fn.__name__] = fn
                return fn

        async def exercise():
            with tempfile.TemporaryDirectory() as tmp:
                cfg = ExperimentConfig(datasets=[], output_dir=tmp, run_name="test",
                                       two_stage=True, use_reflector=False, use_features=False)
                with patch("castmind.agents.generator_agent.Agent", OfflineAgent):
                    agent = create_generator_agent("offline", cfg, {}, {}, {}, lambda *a, **kw: {},
                                                   str, None, lambda *a: {})
                for start, length, value in [(0, 48, 1), (48, 48, 2), (48, 48, 3)]:
                    cfg._active_forecast_segment = self.context(0, start, length)
                    await agent.tools["emit_predictions"](
                        None, [float(value)]*length, "unused.csv", length, tmp, "ETTh1",
                        window_offset=0, frequency="h", start_timestamp="2017-08-21")
                result = pd.read_csv(f"{tmp}/ETTh1/runs/test/predictions.csv")
                self.assertEqual(result.horizon_index.tolist(), list(range(96)))
                self.assertEqual(result.prediction.tolist(), [1.0]*48 + [3.0]*48)
                self.assertEqual(pd.to_datetime(result.time_stamp).tolist(),
                                 pd.date_range("2017-08-21", periods=96, freq="h").tolist())
        asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()
