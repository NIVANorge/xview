import xarray as xr
from typing import Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

import os
import requests

import elementpath
from lxml import etree
from urllib.parse import urlparse

def subset(
    ds: xr.Dataset,
    vars,
    start: Optional[int | datetime],
    end: Optional[int | datetime],
    step: Optional[int],
) -> xr.Dataset:
    if vars:
        ds = ds[[v for v in ds.data_vars if v in vars]]
    if "time" not in ds.dims:
        return ds
    if isinstance(start, datetime):
        ds = ds.sel(time=slice(start, end))
    else:
        ds = ds.isel(time=slice(start, end, step))
    return ds.isel(time=slice(None, None, step))


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