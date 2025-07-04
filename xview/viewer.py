import hvplot.xarray
import hvplot.pandas
import panel as pn
import cf_xarray
import geoviews as gv
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

    if len(ds.dims) == 1 and time_dim_name(ds):
        app = pn.FlexBox()
        params = query_params(ds)
        map_col, plot_col = discrete_time_widgets(ds, url, params)
        app.extend(
            [title, pn.FlexBox(pn.Column(info, ds_pane, max_width=600), map_col, flex_direction="row"), plot_col]
        )
        return app

    return pn.FlexBox(
        pn.Column(
            title, info, ds_pane, data_links(url), "### Plotting is only supported for 1D timeSeries or trajectory."
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

    dim_name = time_dim_name(ds)
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
    if "time" in ds.cf:
        return ds.cf["time"].name
    if "T" in ds.cf.axes:
        return ds.cf.axes["T"][0]
    if "time" in ds.dims:
        return "time"
    if "TIME" in ds.dims:
        return "TIME"
    return ""


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

    dim_name = time_dim_name(ds)
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
