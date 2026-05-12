import xarray as xr
import cf_xarray  # noqa: F401 – registers the .cf accessor
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

import os
import requests

import elementpath
from lxml import etree
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Dimension/coordinate helpers
# ---------------------------------------------------------------------------


def time_dim_name(ds: xr.Dataset) -> str:
    """Return the name of the time dimension, or '' if not found."""
    if "time" in ds.cf:
        return ds.cf["time"].name
    if "T" in ds.cf.axes:
        return ds.cf.axes["T"][0]
    if "time" in ds.dims:
        return "time"
    if "TIME" in ds.dims:
        return "TIME"
    return ""


def depth_dim_name(ds: xr.Dataset) -> str:
    """Return the name of the vertical/depth dimension, or '' if not found."""
    if "Z" in ds.cf.axes:
        return ds.cf.axes["Z"][0]
    for name in ("depth", "DEPTH", "z", "altitude", "level", "lev"):
        if name in ds.dims:
            return name
    return ""


# ---------------------------------------------------------------------------
# Ragged-array timeSeriesProfile helpers (CF H.5.3)
# ---------------------------------------------------------------------------


def is_ragged_tsp(ds: xr.Dataset) -> bool:
    """Return True if *ds* uses a ragged-array timeSeriesProfile layout.

    Detects the presence of a counting variable that carries the
    ``sample_dimension`` CF attribute (e.g. ``rowSize``).
    """
    return any("sample_dimension" in ds[v].attrs for v in ds.data_vars)


def ragged_counting_vars(ds: xr.Dataset) -> tuple[str | None, str | None]:
    """Return ``(row_size_var, station_index_var)`` names.

    ``station_index_var`` is ``None`` for single-station ragged arrays.
    """
    row_size_var = next(
        (v for v in ds.data_vars if "sample_dimension" in ds[v].attrs), None
    )
    station_index_var = next(
        (v for v in ds.data_vars if "instance_dimension" in ds[v].attrs), None
    )
    return row_size_var, station_index_var


def _decode_bytes(arr: np.ndarray) -> np.ndarray:
    """Decode a numpy array of byte strings to Unicode strings.

    Tries UTF-8 first, falls back to Latin-1 (covers Norwegian/Western-European
    characters stored as single-byte encodings, e.g. ``ø`` = 0xf8 in Latin-1).
    """
    if arr.dtype.kind == "S":
        # Fixed-length byte strings (b'...')
        try:
            return arr.astype("U")
        except UnicodeDecodeError:
            return np.array([v.decode("latin-1") for v in arr])
    if arr.dtype == object:
        decoded = []
        for v in arr:
            if isinstance(v, (bytes, np.bytes_)):
                try:
                    decoded.append(v.decode("utf-8"))
                except UnicodeDecodeError:
                    decoded.append(v.decode("latin-1"))
            else:
                decoded.append(v)
        return np.array(decoded, dtype=object)
    return arr


def expand_ragged_tsp(ds: xr.Dataset) -> pd.DataFrame:
    """Expand a ragged-array timeSeriesProfile dataset into a flat DataFrame.

    Handles both H.5.3 (indexed + contiguous ragged, multi-station) and
    simpler single-station contiguous ragged arrays.
    """
    row_size_var, station_index_var = ragged_counting_vars(ds)
    if row_size_var is None:
        raise ValueError("No variable with 'sample_dimension' attribute found in dataset")

    obs_dim = ds[row_size_var].attrs["sample_dimension"]
    profile_dim = ds[row_size_var].dims[0]

    row_sizes = ds[row_size_var].values.astype(int)
    profile_of_obs = np.repeat(np.arange(len(row_sizes)), row_sizes)

    result: dict[str, np.ndarray] = {}
    skip = {row_size_var, station_index_var}

    # Station-level variables → broadcast through stationIndex → obs
    if station_index_var is not None:
        instance_dim = ds[station_index_var].attrs["instance_dimension"]
        stn_of_obs = ds[station_index_var].values[profile_of_obs]
        for v in list(ds.data_vars) + list(ds.coords):
            if v not in ds:
                continue
            if ds[v].dims == (instance_dim,):
                result[v] = _decode_bytes(ds[v].values)[stn_of_obs]

    # Profile-level variables (time, etc.) → repeat per obs
    for v in list(ds.data_vars) + list(ds.coords):
        if v not in ds or v in skip:
            continue
        if ds[v].dims == (profile_dim,):
            result[v] = ds[v].values[profile_of_obs]

    # Obs-level variables (depth, data variables)
    for v in list(ds.data_vars) + list(ds.coords):
        if v not in ds or v == row_size_var:
            continue
        if ds[v].dims == (obs_dim,):
            result[v] = ds[v].values

    df = pd.DataFrame(result)

    # Ensure any remaining byte-string columns are decoded to str
    for col in df.columns:
        if df[col].dtype.kind == "S" or (
            df[col].dtype == object and df[col].apply(lambda v: isinstance(v, (bytes, np.bytes_))).any()
        ):
            df[col] = _decode_bytes(df[col].values)

    return df


# ---------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------


def subset(
    ds: xr.Dataset,
    vars,
    start: Optional[int | datetime],
    end: Optional[int | datetime],
    step: Optional[int],
) -> xr.Dataset:
    if vars:
        ds = ds[[v for v in ds.data_vars if v in vars]]
    dim_name = time_dim_name(ds)
    if not dim_name:
        return ds
    if isinstance(start, datetime):
        ds = ds.sel({dim_name: slice(start, end)})
        return ds.isel({dim_name: slice(None, None, step)})
    else:
        return ds.isel({dim_name: slice(start, end, step)})


def bytes_to_str(ds: xr.Dataset) -> xr.Dataset:
    for var in ds.data_vars:
        if ds[var].dtype.kind == "S":
            ds[var] = ds[var].astype(str)
    return ds

def datetime_with_attrs(ds, var) -> xr.Dataset:
    attrs = ds[var].attrs
    ds[var] = ds[var].dt.strftime("%Y-%m-%dT%H:%M:%S")
    ds[var].attrs = attrs

    return ds


def to_json_types(ds: xr.Dataset, fill_nan: bool = True) -> xr.Dataset:
    for var in ds.data_vars:
        if ds[var].dtype.kind == "M":
            ds = datetime_with_attrs(ds, var)
        elif ds[var].dtype.kind == "S":  # Check if the variable is a float
            ds[var] = ds[var].astype(str)
        elif fill_nan and ds[var].dtype.kind == "f":
            # convert float to object to allow for None as null value
            ds[var] = ds[var].astype(object)

    for var in ds.coords:
        if ds[var].dtype.kind == "M":  # Check if the variable is datetime
            ds = datetime_with_attrs(ds, var)

    return ds.fillna(None) if fill_nan else ds


def walk_catalog(catalog_base: str) -> dict[str,list]:
    catalog_url = f"{urlparse(catalog_base).scheme}://{urlparse(catalog_base).netloc}"
    catalogs = [catalog_base]
    dap_lookup = {}
    datasets = {}
    while catalogs:
        cat = catalogs.pop(0)
        base = cat.rsplit(".", 1)[0]
        cat_doc = etree.fromstring(requests.get(cat).content)
        name = os.path.basename(cat).split(".")[0]
        datasets[name] = []
        for el in cat_doc:
            if el.tag.endswith("catalogRef"):
                catalogs.append(f"{base}/{el.get('{http://www.w3.org/1999/xlink}href')}")
            elif el.tag.endswith("service"):
                services = elementpath.select(el, "//*[lower-case(@serviceType)='opendap']")
                if services:
                    for service in services:
                        dap_lookup[service.getparent().get("name")] = service.get("base")
            elif el.tag.endswith("dataset"):
                for t in el:
                    if t.tag.endswith("serviceName") and t.text in dap_lookup:
                        datasets[name].append(f"{catalog_url}{dap_lookup[t.text]}{el.get('urlPath')}")
    
    return datasets