"""Known-answer spatial weighting, alignment and persistence tests."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perthwx.validation import grid_scores, station_baseline_scores, station_pairs_with_persistence, fetch_truth


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.truth = xr.Dataset({"t2m": (("lead_time", "lat", "lon"), np.array([[[0], [0]], [[2], [4]]], dtype=float))}, coords={"lead_time": np.array([0, 6], dtype="timedelta64[h]"), "lat": [0, 60], "lon": [116]})
        self.forecast = self.truth.expand_dims(time=[np.datetime64("2022-01-01")]).copy(deep=True)
        self.forecast["t2m"] = self.forecast.t2m + 1

    def test_area_weighted_scores(self):
        scores = grid_scores(self.forecast, self.truth).set_index("model")
        self.assertEqual(scores.loc["FCN", "rmse"], 1)
        self.assertEqual(scores.loc["FCN", "bias"], 1)
        self.assertAlmostEqual(scores.loc["persistence", "rmse"], np.sqrt(8))
        self.assertAlmostEqual(scores.loc["persistence", "bias"], -8 / 3)
        self.assertEqual(scores.loc["FCN", "n_expected"], 2)
        self.assertEqual(len(scores), 2)  # lead zero omitted

    def test_exact_alignment(self):
        with self.assertRaises(ValueError):
            grid_scores(self.forecast, self.truth.assign_coords(lon=[115]))
        with self.assertRaises(ValueError):
            grid_scores(self.forecast, self.truth.assign_coords(lead_time=np.array([0, 12], dtype="timedelta64[h]")))

    def test_scores_do_not_depend_on_zarr_variable_enumeration(self):
        truth = self.truth.assign(t850=self.truth.t2m + 2)
        forecast = self.forecast.assign(t850=self.forecast.t2m + 2)
        before = grid_scores(forecast, truth)
        after = grid_scores(forecast[["t850", "t2m"]], truth[["t850", "t2m"]])
        pd.testing.assert_frame_equal(before, after, check_exact=True)

    def test_common_mask(self):
        self.forecast.t2m.values[0, 1, 1, 0] = np.nan
        scores = grid_scores(self.forecast, self.truth).set_index("model")
        self.assertTrue((scores.n_matched == 1).all())
        self.assertEqual(scores.loc["persistence", "rmse"], 2)
        self.truth.t2m.values[1, 0, 0] = np.nan
        scores = grid_scores(self.forecast, self.truth)
        self.assertTrue((scores.n_matched == 0).all())
        self.assertTrue(scores.rmse.isna().all())

    def test_station_persistence_never_uses_future_report(self):
        from perthwx.verification import Station, station_forecast
        ds = self.forecast.assign_coords(lat=[-31.75, -32], lon=[116])
        fc = station_forecast(ds, Station(lon=116))
        obs = pd.DataFrame(dict(station=["YPPH"] * 3, variable=["t2m"] * 3,
            time=["2021-12-31T23:40Z", "2022-01-01T00:01Z", "2022-01-01T06:00Z"], observation=[10, 20, 30]))
        pairs = station_pairs_with_persistence(fc, obs)
        self.assertTrue((pairs.persistence == 10).all())
        pairs = station_pairs_with_persistence(fc, obs.iloc[1:])
        self.assertTrue(pairs.persistence.isna().all())

    def test_truth_offline_requires_snapshots(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                fetch_truth(self.forecast, Path(folder), offline=True)

    def test_station_shared_baseline_mask(self):
        pairs = pd.DataFrame(dict(station=["YPPH"] * 3, variable=["t2m"] * 3, units=["K"] * 3, lead_hours=[0, 6, 12], forecast=[999, 12, 20], observation=[10, 10, 10], persistence=[10, 9, np.nan], matched=[True] * 3))
        scores = station_baseline_scores(pairs).set_index("model")
        self.assertTrue((scores.n_matched == 1).all())
        self.assertTrue((scores.n_expected == 2).all())
        self.assertEqual(scores.loc["FCN", "rmse"], 2)
        self.assertEqual(scores.loc["persistence", "bias"], -1)


if __name__ == "__main__":
    unittest.main()
