"""Offline RH thermodynamics and data-wiring regression tests."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perthwx.data import ERA5WithRH
from perthwx.config import ForecastConfig
from perthwx.forecast import _parse_args, run_forecast


class ERA5Tests(unittest.TestCase):
    def test_rh_known_answers_and_order(self):
        # Triple point: es=611.21 Pa. Independent inversion e -> specific humidity.
        epsilon = 0.621981
        temperature = np.array([273.16, 273.16, 273.16, 273.16])
        def source(time, variables):
            self.assertEqual(variables, ["t850", "q850", "t2m", "t500", "q500"])
            data = []
            for v in variables:
                if v.startswith("q"):
                    pressure = int(v[1:]) * 100
                    e = np.array([0, 0.5, 1, 1.5]) * 611.21
                    data.append(epsilon * e / (pressure - (1 - epsilon) * e))
                else:
                    data.append(temperature)
            return xr.DataArray(np.array(data)[None, :, None, :], dims=["time", "variable", "lat", "lon"], coords={"time": [np.datetime64("2022-01-01")], "variable": variables, "lat": [-32], "lon": [114, 115, 116, 117]})
        result = ERA5WithRH(source)([np.datetime64("2022-01-01")], ["r850", "t2m", "r500", "t850", "r850"])
        self.assertEqual(result["variable"].values.tolist(), ["r850", "t2m", "r500", "t850", "r850"])
        np.testing.assert_allclose(result.isel(variable=0).values.ravel(), [0, 50, 100, 100], atol=0.001)
        self.assertEqual(result.dims, ("time", "variable", "lat", "lon"))
        self.assertEqual(result.dtype, np.float32)

    def test_ice_mixed_and_warm_rh(self):
        # ECMWF mixed-phase reference states, each at 50% RH and 850 hPa.
        t = [240., 260., 280., 300.]
        q = [9.95756344593248e-5, 0.000733431686517053, 0.003633108554891113, 0.013023248993848833]
        def source(time, variables):
            return xr.DataArray(np.array([[t, q]])[:, :, None, :], dims=["time", "variable", "lat", "lon"], coords={"time": [np.datetime64("2022-01-01")], "variable": variables, "lat": [-32], "lon": [114, 115, 116, 117]})
        result = ERA5WithRH(source)([np.datetime64("2022-01-01")], ["r850"])
        np.testing.assert_allclose(result.values.ravel(), [50] * 4, atol=0.001)

    def test_passthrough(self):
        expected = xr.DataArray(np.ones((1, 1, 1, 1)), dims=["time", "variable", "lat", "lon"], coords={"time": [np.datetime64("2022-01-01")], "variable": ["t2m"], "lat": [-32], "lon": [116]})
        actual = ERA5WithRH(lambda t, v: expected)("2022-01-01", "t2m")
        xr.testing.assert_allclose(actual, expected.astype(np.float32))

    def test_cli_and_steps(self):
        self.assertEqual(_parse_args([]).model, "FCN")
        self.assertEqual(ForecastConfig().nsteps, 40)
        for days in (0, -1, float("nan"), 0.1):
            with self.assertRaises(ValueError):
                _ = ForecastConfig(lead_days=days).nsteps

    def test_dry_run_does_not_touch_existing_output(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "existing.zarr"
            out.mkdir()
            marker = out / "marker"
            marker.write_text("keep")
            with patch("perthwx.forecast._model_registry") as reg, patch("perthwx.forecast._data_source"):
                reg.return_value = {"FCN": unittest.mock.Mock()}
                run_forecast(ForecastConfig(out_path=str(out)), dry_run=True)
                self.assertEqual(marker.read_text(), "keep")
                with self.assertRaises(FileExistsError):
                    run_forecast(ForecastConfig(out_path=str(out)))


if __name__ == "__main__":
    unittest.main()
