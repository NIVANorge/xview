import hvplot.xarray
import hvplot.pandas
import panel as pn
import cf_xarray
import geoviews as gv
import numpy as np
import xarray as xr
import urllib.parse
import pandas as pd
from datetime import datetime, timedelta


from xview import utils
from xview.config import SETTINGS
from dataclasses import dataclass

DEFAULT_N_POINTS = 100000


@dataclass
class Params:
    current_param: str
    param_list: list[str]
    start: datetime | int | None
    end: datetime | int | None
    step: str | int = 1


def create_app():
    url = urllib.parse.unquote(pn.state.session_args.get("url", [None])[0].decode("utf-8"))
    ds = xr.open_dataset(url)

    title, info, ds_pane = create_info(ds, url)

    if len(ds.dims) == 1 and utils.time_dim_name(ds):
        app = pn.FlexBox()
        params = query_params(ds)
        map_col, plot_col = discrete_time_widgets(ds, url, params)
        app.extend(
            [title, pn.FlexBox(pn.Column(info, ds_pane, max_width=600), map_col, flex_direction="row"), plot_col]
        )
        return app

    if utils.is_ragged_tsp(ds):
        app = pn.FlexBox()
        map_col, plot_col = tsp_ragged_widgets(ds, url)
        app.extend(
            [title, pn.FlexBox(pn.Column(info, ds_pane, max_width=600), map_col, flex_direction="row"), plot_col]
        )
        return app

    if utils.time_dim_name(ds) and utils.depth_dim_name(ds):
        app = pn.FlexBox()
        params = query_params(ds)
        map_col, plot_col = tsp_orthogonal_widgets(ds, url, params)
        app.extend(
            [title, pn.FlexBox(pn.Column(info, ds_pane, max_width=600), map_col, flex_direction="row"), plot_col]
        )
        return app

    return pn.FlexBox(
        pn.Column(
            title, info, ds_pane, data_links(url),
            "### Plotting is only supported for timeSeries, trajectory, or timeSeriesProfile."
        )
    )


def create_info(ds, url):

    info_txt = (
        f"## Welcome to xview!\n\n"
        f"xview allows you to visualize datasets and export them in various formats.\n\n"
        f"Also see the **[API docs]({SETTINGS.server_url}/xview/docs)** or the **[xview repo](https://github.com/NIVANorge/xview)** for more information.\n\n"
    )
    ds_txt = (
        f"### Dataset Information\n"
        f"- **Title:** {ds.attrs.get('title', 'N/A')}\n"
        f"- **Summary:** {ds.attrs.get('summary', 'N/A')}\n"
        f"- **OPeNDAP URL:** [{url}]({url}.html)\n"
    )

    return pn.pane.Markdown(info_txt), pn.pane.Markdown(ds_txt), pn.panel(ds)


def query_params(ds):
    params = {
        p: pn.state.session_args[p][0].decode("utf-8")
        for p in ["parameter-name", "start", "end", "step"]
        if pn.state.session_args.get(p)
    }

    time_default = True

    if "start" in params:
        if params["start"].isdigit():
            time_default = False
            params["start"] = int(params["start"])
        else:
            params["start"] = pd.to_datetime(params["start"])

    if "end" in params:
        if params["end"].isdigit():
            time_default = False
            params["end"] = int(params["end"])
        else:
            params["end"] = pd.to_datetime(params["end"])

    dim_name = utils.time_dim_name(ds)
    if "step" in params and params["step"].isdigit():
        params["step"] = int(params["step"])

    if "start" not in params and len(ds[dim_name]) > DEFAULT_N_POINTS:
        # If the dataset is large and no start time is provided, set a default start time
        if time_default:
            params["start"] = pd.to_datetime(ds[dim_name].values[-DEFAULT_N_POINTS])
        else:
            params["start"] = -DEFAULT_N_POINTS

    param_list = [
        f"{ds[v].attrs['long_name']}[{v}]" if "long_name" in ds[v].attrs else v
        for v in ds.data_vars
        if len(ds[v].dims) > 0 and not v.endswith("_qc")
    ]
    if "parameter-name" not in params:
        params["parameter-name"] = param_list[0] if param_list else None

    return Params(
        current_param=params["parameter-name"],
        param_list=param_list,
        start=params.get("start", pd.to_datetime(ds[dim_name].min().values) if time_default else 0),
        end=params.get("end", pd.to_datetime(ds[dim_name].max().values) if time_default else len(ds[dim_name]) - 1),
        step=params.get("step", 1),
    )


def data_links(url, start=None, end=None, step=None):
    if isinstance(start, datetime):
        start = pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S")
        end = pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S")

    query_params = [f"{p}={v}" for p, v in [("start", start), ("end", end), ("step", step)] if v is not None]
    query_string = "&" + "&".join(query_params) if query_params else ""
    html_url = f"{SETTINGS.server_url}/xview/data?f=html&url={urllib.parse.quote(url)}{query_string}"
    json_url = f"{SETTINGS.server_url}/xview/data?f=json&url={urllib.parse.quote(url)}{query_string}"
    csv_url = f"{SETTINGS.server_url}/xview/data?f=csv&url={urllib.parse.quote(url)}{query_string}"

    html_url = f"[{html_url}]({html_url})"
    json_url = f"[{json_url}]({json_url})"
    csv_url = f"[{csv_url}]({csv_url})"
    txt = f"### Data Links\n" f"- **HTML:** {html_url}\n" f"- **JSON:** {json_url}\n" f"- **CSV:** {csv_url}\n"
    return pn.pane.Markdown(txt)


@pn.cache
def sel(ds, dim_name, start, end, step):

    if isinstance(start, datetime):
        return ds.sel({dim_name: slice(start, end)}).isel({dim_name: slice(None, None, step)})

    return ds.isel({dim_name: slice(start, end, step)})


def varname_from_selector(variable_selector):
    if "[" in variable_selector:
        return variable_selector.split("[")[1].split("]")[0]
    return variable_selector


def time_dim_name(ds) -> str:
    return utils.time_dim_name(ds)


@pn.cache
def time_plot_widget(variable_selector, ds, dim_name, start, end, step):
    var = varname_from_selector(variable_selector)

    point_size = 5
    if ds[var].size < 10000:
        point_size = 50
    return sel(ds[var], dim_name, start, end, step).hvplot.scatter(
        x=dim_name, size=point_size, sizing_mode="stretch_width", min_height=400, max_height=600, responsive=True
    )


@pn.cache
def map_plot_widget(ds, variable_selector, dim_name, end, start, step, apply_to_map):

    var = varname_from_selector(variable_selector)

    x = ds.cf["longitude"].name
    y = ds.cf["latitude"].name

    if isinstance(end, int):
        end = pd.to_datetime(ds[dim_name].values[end])

    if not apply_to_map:
        # use default range
        start = end - timedelta(days=1)
    elif isinstance(start, int):
        start = pd.to_datetime(ds[dim_name].values[start])

    df = ds.sel({dim_name: slice(start, end)}).isel({dim_name: slice(None, None, step)}).to_dataframe()
    if len(df) == 0:
        return pn.pane.Markdown(f"No data in range {start} - {end}")

    is_cbar = True
    point_size = 15
    hover_cols = [dim_name, x, y, var]
    if ds.cf["longitude"].size == 1:
        hover_cols = [x, y]
        is_cbar = False
        point_size = 150

    return (
        df.hvplot.points(
            x=x,
            y=y,
            c=var,
            hover_cols=hover_cols,
            geo=True,
            tiles="OSM",
            cmap="viridis",
            size=point_size,
            height=600,
            colorbar=is_cbar,
            clabel=f"{ds[var].attrs.get('long_name', var)}[{ds[var].attrs.get('units', '')}]",
            min_width=200,
            max_width=800,
        )
    ).opts(default_span=800.0, width=800, responsive=True)


def time_control_widgets(ds, params, dim_name):
    variable_selector = pn.widgets.Select(name="Variable", options=params.param_list, value=params.current_param)
    step_slider = pn.widgets.IntSlider(name="plot every n point", value=params.step, start=1, end=100, step=10)

    if isinstance(params.start, datetime):
        start_slider = pn.widgets.DatetimeSlider(
            name="Start Time", start=ds[dim_name].min().values, end=ds[dim_name].max().values, value=params.start
        )
        end_slider = pn.widgets.DatetimeSlider(
            name="End Time", start=ds[dim_name].min().values, end=ds[dim_name].max().values, value=params.end
        )
    else:
        start_slider = pn.widgets.IntSlider(
            name="Start Range", start=0, end=len(ds[dim_name]) - 1, step=1, value=params.start
        )
        end_slider = pn.widgets.IntSlider(
            name="End Range", start=0, end=len(ds[dim_name]) - 1, step=1, value=params.end
        )

    return variable_selector, step_slider, start_slider, end_slider


def discrete_time_widgets(ds: xr.Dataset, url, params: Params):

    dim_name = utils.time_dim_name(ds)
    variable_selector, step_slider, start_slider, end_slider = time_control_widgets(ds, params, dim_name)

    if pn.state.location:
        pn.state.location.sync(start_slider, {"value": "start"})
        pn.state.location.sync(end_slider, {"value": "end"})
        pn.state.location.sync(variable_selector, {"value": "parameter-name"})

    time_plot = pn.bind(
        time_plot_widget,
        variable_selector=variable_selector,
        ds=ds,
        dim_name=dim_name,
        start=start_slider,
        end=end_slider,
        step=step_slider,
    )
    download_binding = pn.bind(data_links, url=url, start=start_slider, end=end_slider, step=step_slider)

    map_title = "### Location"
    apply_to_map = None
    if ds.cf["longitude"].size > 1:
        apply_to_map = pn.widgets.Checkbox(name="Apply to map", value=False)
        map_title = "### Map Preview (default is 24 hours)"
    

    map_plot = pn.bind(
        map_plot_widget,
        ds=ds,
        variable_selector=variable_selector,
        dim_name=dim_name,
        end=end_slider,
        start=start_slider,
        step=step_slider,
        apply_to_map=apply_to_map,
    )

    controls = [
        pn.Column("### Controls", variable_selector, step_slider),
        pn.Column("### Time Range", apply_to_map, start_slider, end_slider),
    ]

    time_box = pn.FlexBox(sizing_mode="stretch_width")
    time_box.extend([*controls, download_binding, time_plot])

    return pn.Column(map_title, map_plot), time_box


# ---------------------------------------------------------------------------
# Orthogonal timeSeriesProfile (time × depth 2-D grid)
# ---------------------------------------------------------------------------


@pn.cache
def tsp_heatmap_widget(variable_selector, ds, dim_time, dim_depth, start, end, step):
    var = varname_from_selector(variable_selector)
    ds_sub = sel(ds, dim_time, start, end, step)
    label = f"{ds[var].attrs.get('long_name', var)} [{ds[var].attrs.get('units', '')}]"
    positive_down = ds[dim_depth].attrs.get("positive", "down") == "down"
    return ds_sub[var].hvplot.quadmesh(
        x=dim_time,
        y=dim_depth,
        cmap="viridis",
        rasterize=True,
        responsive=True,
        min_height=400,
        max_height=700,
        clabel=label,
        flip_yaxis=positive_down,
    )


def _tsp_location_widget(ds):
    """Return a map pane showing a single station from global attrs, or a placeholder."""
    lat = ds.attrs.get("geospatial_lat_min") or ds.attrs.get("geospatial_lat_max")
    lon = ds.attrs.get("geospatial_lon_min") or ds.attrs.get("geospatial_lon_max")
    if lat is None or lon is None:
        return pn.pane.Markdown("### Location\nNo location data available.")
    location_df = pd.DataFrame({"lon": [float(lon)], "lat": [float(lat)]})
    plot = location_df.hvplot.points(
        x="lon",
        y="lat",
        geo=True,
        tiles="OSM",
        size=150,
        color="red",
        hover_cols=["lat", "lon"],
        height=400,
        width=500,
    )
    return pn.Column("### Location", plot)


def tsp_orthogonal_widgets(ds: xr.Dataset, url: str, params: Params):
    """Widgets for orthogonal timeSeriesProfile (time × depth 2-D grid)."""
    dim_time = utils.time_dim_name(ds)
    dim_depth = utils.depth_dim_name(ds)

    variable_selector, step_slider, start_slider, end_slider = time_control_widgets(ds, params, dim_time)

    if pn.state.location:
        pn.state.location.sync(start_slider, {"value": "start"})
        pn.state.location.sync(end_slider, {"value": "end"})
        pn.state.location.sync(variable_selector, {"value": "parameter-name"})

    heatmap = pn.bind(
        tsp_heatmap_widget,
        variable_selector=variable_selector,
        ds=ds,
        dim_time=dim_time,
        dim_depth=dim_depth,
        start=start_slider,
        end=end_slider,
        step=step_slider,
    )
    download_binding = pn.bind(data_links, url=url, start=start_slider, end=end_slider, step=step_slider)

    controls = [
        pn.Column("### Controls", variable_selector, step_slider),
        pn.Column("### Time Range", start_slider, end_slider),
    ]

    plot_box = pn.FlexBox(sizing_mode="stretch_width")
    plot_box.extend([*controls, download_binding, heatmap])

    return _tsp_location_widget(ds), plot_box


# ---------------------------------------------------------------------------
# Ragged-array timeSeriesProfile (CF H.5.3)
# ---------------------------------------------------------------------------


@pn.cache
def _get_expanded_ragged_df(url: str) -> pd.DataFrame:
    """Open dataset and expand ragged arrays; result is cached by URL."""
    ds = xr.open_dataset(url)
    return utils.expand_ragged_tsp(ds)


@pn.cache
def tsp_ragged_plot_widget(station, variable_selector, url, dim_time, dim_depth, stn_id_var):
    df = _get_expanded_ragged_df(url)
    var = varname_from_selector(variable_selector)

    if stn_id_var and station is not None:
        plot_df = df[df[stn_id_var] == station].copy()
    else:
        plot_df = df.copy()

    if len(plot_df) == 0:
        return pn.pane.Markdown("No data for the selected station.")

    if dim_time and dim_time in plot_df.columns:
        plot_df[dim_time] = pd.to_datetime(plot_df[dim_time])

    label = variable_selector.split("[")[0].strip() if "[" in variable_selector else variable_selector

    hover = [c for c in [dim_time, dim_depth, var, stn_id_var] if c and c in plot_df.columns]

    return plot_df.hvplot.scatter(
        x=dim_time,
        y=dim_depth,
        c=var,
        cmap="viridis",
        responsive=True,
        min_height=400,
        max_height=700,
        clabel=label,
        hover_cols=hover,
        flip_yaxis=True,
    )


def _tsp_ragged_map_widget(ds: xr.Dataset, station_index_var: str | None, stn_id_var: str | None):
    """Static map showing all station locations from the ragged dataset."""
    if station_index_var is None:
        return pn.pane.Markdown("### Location\nNo multi-station location data available.")

    instance_dim = ds[station_index_var].attrs["instance_dimension"]

    # Find lat/lon on the instance dimension
    lat_var = next(
        (v for v in list(ds.data_vars) + list(ds.coords)
         if v in ds and ds[v].dims == (instance_dim,)
         and ds[v].attrs.get("standard_name", "").lower() in ("latitude",)),
        next((v for v in ("lat", "latitude", "LAT") if v in ds and ds[v].dims == (instance_dim,)), None),
    )
    lon_var = next(
        (v for v in list(ds.data_vars) + list(ds.coords)
         if v in ds and ds[v].dims == (instance_dim,)
         and ds[v].attrs.get("standard_name", "").lower() in ("longitude",)),
        next((v for v in ("lon", "longitude", "LON") if v in ds and ds[v].dims == (instance_dim,)), None),
    )

    if lat_var is None or lon_var is None:
        return pn.pane.Markdown("### Location\nNo location data found on station dimension.")

    location_df = pd.DataFrame(
        {"lat": ds[lat_var].values.astype(float), "lon": ds[lon_var].values.astype(float)}
    )
    if stn_id_var and stn_id_var in ds:
        location_df["station"] = utils._decode_bytes(ds[stn_id_var].values)

    hover_cols = [c for c in location_df.columns if c in ("lat", "lon", "station")]
    plot = location_df.hvplot.points(
        x="lon",
        y="lat",
        geo=True,
        tiles="OSM",
        size=80,
        color="red",
        hover_cols=hover_cols,
        height=400,
        width=500,
    )
    return pn.Column("### Station Locations", plot)


def tsp_ragged_widgets(ds: xr.Dataset, url: str):
    """Widgets for ragged-array timeSeriesProfile (CF H.5.3 or similar)."""
    row_size_var, station_index_var = utils.ragged_counting_vars(ds)
    obs_dim = ds[row_size_var].attrs["sample_dimension"]
    profile_dim = ds[row_size_var].dims[0]

    skip = {row_size_var, station_index_var}

    # Data variables are those on the obs dimension (excluding counting vars and QC flags)
    param_list = [
        f"{ds[v].attrs['long_name']}[{v}]" if "long_name" in ds[v].attrs else v
        for v in ds.data_vars
        if ds[v].dims == (obs_dim,) and v not in skip and not v.endswith("_qc")
    ]

    # Find depth variable on obs dimension
    depth_var = next(
        (v for v in list(ds.data_vars) + list(ds.coords)
         if v in ds and v not in skip
         and ds[v].dims == (obs_dim,)
         and ds[v].attrs.get("standard_name", "") in ("depth", "altitude", "height")),
        next(
            (v for v in ("depth", "DEPTH", "z", "altitude")
             if v in ds and ds[v].dims == (obs_dim,)),
            None,
        ),
    )

    # Find time variable on profile dimension (must be datetime-typed)
    time_var = next(
        (v for v in list(ds.data_vars) + list(ds.coords)
         if v in ds and ds[v].dims == (profile_dim,) and ds[v].dtype.kind == "M"),
        next(
            (v for v in ("time", "TIME") if v in ds and ds[v].dims == (profile_dim,)),
            None,
        ),
    )

    # Station identifier variable (cf_role = 'timeseries_id')
    stn_id_var = next(
        (v for v in list(ds.data_vars) + list(ds.coords)
         if v in ds and ds[v].attrs.get("cf_role") == "timeseries_id"),
        None,
    )

    variable_selector = pn.widgets.Select(
        name="Variable", options=param_list, value=param_list[0] if param_list else None
    )
    controls = [pn.Column("### Controls", variable_selector)]

    station_selector = None
    if station_index_var is not None and stn_id_var is not None and stn_id_var in ds:
        stations = [
            v.decode("latin-1") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in ds[stn_id_var].values
        ]
        station_selector = pn.widgets.Select(name="Station", options=stations, value=stations[0])
        controls.append(pn.Column("### Station", station_selector))

    plot = pn.bind(
        tsp_ragged_plot_widget,
        station=station_selector,
        variable_selector=variable_selector,
        url=url,
        dim_time=time_var,
        dim_depth=depth_var,
        stn_id_var=stn_id_var,
    )

    plot_box = pn.FlexBox(sizing_mode="stretch_width")
    plot_box.extend([*controls, data_links(url), plot])

    map_col = _tsp_ragged_map_widget(ds, station_index_var, stn_id_var)
    return map_col, plot_box

