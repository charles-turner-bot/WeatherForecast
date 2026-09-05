"""ERA5 initial conditions with pressure-level relative humidity for FCN."""
from __future__ import annotations

from collections import OrderedDict
import re

import numpy as np
import xarray as xr


class ERA5WithRH:
    """Wrap ARCO and derive requested r<pressure> fields using Earth2Studio.

    ARCO supplies temperature in K and specific humidity in kg/kg. DerivedRH
    returns percent using ECMWF mixed ice/water saturation and clips to [0,100].
    Derive on CPU in float32; preserve source coordinates and requested ordering.
    A source can be injected for offline tests or saved initial-condition data.
    """

    def __init__(self, source=None):
        if source is None:
            from earth2studio.data import ARCO
            source = ARCO()
        self.source = source

    def __call__(self, time, variable) -> xr.DataArray:
        names = [variable] if isinstance(variable, str) else list(variable)
        if not names:
            raise ValueError("At least one variable is required")
        levels = list(dict.fromkeys(int(v[1:]) for v in names if re.fullmatch(r"r\d+", v)))
        if any(level <= 0 or level > 1000 for level in levels):
            raise ValueError("RH pressure levels must be in (0, 1000] hPa")
        needed = []
        for name in names:
            fields = [f"t{int(name[1:])}", f"q{int(name[1:])}"] if re.fullmatch(r"r\d+", name) else [name]
            needed.extend(field for field in fields if field not in needed)
        data = self.source(time, needed).transpose("time", "variable", "lat", "lon")
        if levels:
            import torch
            from earth2studio.models.dx import DerivedRH
            diagnostic = DerivedRH(levels)
            inputs = data.sel(variable=diagnostic.in_variables)
            coords = OrderedDict((dim, inputs[dim].values) for dim in inputs.dims)
            values, out_coords = diagnostic(torch.as_tensor(inputs.values, dtype=torch.float32), coords)
            rh = xr.DataArray(values.numpy(), dims=list(out_coords), coords=out_coords)
            data = xr.concat([data, rh], dim="variable")
        return data.sel(variable=names).astype(np.float32)
