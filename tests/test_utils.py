import xarray as xr
import pandas as pd
import numpy as np
from xview.utils import to_json_types

def test_to_friendly_types():
    ds = xr.Dataset(
        {
            "temp": ("time", [1.0, np.nan, 3.0, 4.0, 5.0]),
            "precip": ("time", [np.nan, 0.1, 0.2, 0.3, 0.4]),
        },
        coords={"time": ("time", pd.date_range("2020-01-01", periods=5))},
    )
    ds = to_json_types(ds)
    assert ds["time"].dtype.kind == "O"
    assert ds["temp"].dtype.kind == "O"
    assert ds["time"].values[0] == "2020-01-01T00:00:00"
    assert ds["temp"].values[0] == 1.0
    assert ds["temp"].values[1] == None
