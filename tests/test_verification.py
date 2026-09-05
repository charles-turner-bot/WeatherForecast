"""Offline known-answer tests: pixi run test-verification."""
import sys
from io import StringIO
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perthwx.verification import Station, station_forecast, match_observations, score_pairs


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset(
            {"t2m": (("time", "lead_time", "lat", "lon"), np.full((1, 3, 2, 2), 280.0))},
            coords={"time": [np.datetime64("2022-01-01")], "lead_time": np.array([0, 6, 12], dtype="timedelta64[h]"), "lat": [-31.75, -32.0], "lon": [115.75, 116.0]},
        )
        self.fc = station_forecast(self.ds, Station())

    def obs(self, times, values):
        return pd.DataFrame({"station": "YPPH", "variable": "t2m", "time": times, "observation": values})

    def test_spatial_and_valid_time(self):
        self.assertTrue((self.fc.grid_lat == -32).all())
        self.assertTrue((self.fc.grid_lon == 116).all())
        self.assertEqual(self.fc.valid_time.iloc[1], pd.Timestamp("2022-01-01T06:00"))
        with self.assertRaises(ValueError):
            station_forecast(self.ds, Station(lat=-50))
        with self.assertRaises(ValueError):
            station_forecast(self.ds.expand_dims(member=[0, 1]), Station())

    def test_matching_scores_and_lead_zero(self):
        obs = self.obs(["2022-01-01T00:00Z", "2022-01-01T05:30Z", "2022-01-01T06:30Z", "2022-01-01T12:31Z"], [100, 278, 290, 300])
        pairs = match_observations(self.fc, obs)
        self.assertEqual(pairs.observation.iloc[1], 278)  # earlier tie, inclusive boundary
        self.assertEqual(pairs.offset_minutes.iloc[1], -30)
        score = score_pairs(pairs).iloc[0]
        self.assertEqual((score.n_expected, score.n_matched, score.n_missing), (2, 1, 1))
        self.assertEqual((score.rmse, score.bias, score.mae), (2, 2, 2))
        self.assertEqual(len(score_pairs(pairs, by_lead=True)), 2)

    def test_duplicates_missing_and_station_isolation(self):
        obs = self.obs(["2022-01-01T06:00"] * 3, [276, 280, np.inf])
        extra = self.obs(["2022-01-01T12:00"], [280]).assign(station="OTHER")
        pairs = match_observations(self.fc, pd.concat([obs, extra]))
        self.assertEqual(pairs.observation.iloc[1], 278)
        self.assertFalse(pairs.matched.iloc[2])
        empty = match_observations(self.fc, obs.iloc[:0])
        self.assertEqual(score_pairs(empty).n_matched.iloc[0], 0)
        self.assertTrue(np.isnan(score_pairs(empty).rmse.iloc[0]))
        with self.assertRaises(ValueError):
            match_observations(self.fc, obs, -1)

    def test_multiple_initializations_and_nonfinite_forecast(self):
        ds = xr.concat([self.ds, self.ds.assign_coords(time=self.ds.time + np.timedelta64(6, "h"))], dim="time")
        fc = station_forecast(ds, Station())
        fc.loc[fc.lead_hours == 12, "forecast"] = np.nan
        obs = self.obs(["2022-01-01T06:00", "2022-01-01T12:00"], [278, 279])
        pairs = match_observations(fc, obs)
        score = score_pairs(pairs).iloc[0]
        self.assertEqual(score.n_expected, 4)
        self.assertEqual(score.n_matched, 2)
        self.assertAlmostEqual(score.rmse, np.sqrt(2.5))

    def test_snapshot_roundtrip(self):
        obs = self.obs(["2022-01-01T06:00"], np.array([278.12345], dtype=np.float32))
        obs["observation"] = obs.observation.astype("float64")
        before = score_pairs(match_observations(self.fc, obs))
        replay = pd.read_csv(StringIO(obs.to_csv(index=False)), float_precision="round_trip")
        after = score_pairs(match_observations(self.fc, replay))
        pd.testing.assert_frame_equal(before, after, check_exact=True)

    def test_wind_speed(self):
        ds = self.ds.assign(u10m=xr.full_like(self.ds.t2m, 3), v10m=xr.full_like(self.ds.t2m, 4))
        fc = station_forecast(ds, Station())
        self.assertTrue((fc.loc[fc.variable == "ws10m", "forecast"] == 5).all())


if __name__ == "__main__":
    unittest.main()
