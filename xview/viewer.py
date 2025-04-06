import hvplot.xarray
import panel as pn
import cf_xarray
import xarray as xr
import urllib.parse
import pandas as pd
from datetime import datetime


from xview import utils
from xview.config import SETTINGS
from dataclasses import dataclass

DEFAULT_N_POINTS = 100000


def info(ds, url):

    info_txt = (
        f"## Welcome to xview!\n\n"
        f"xview allows you to visualize datasets and export them in various formats.\n\n"
        f"Also see the **[API docs](/docs)** for more information.\n\n"
        f"### Dataset Information\n"
        f"- **Title:** {ds.attrs.get('title', 'N/A')}\n"
        f"- **Summary:** {ds.attrs.get('summary', 'N/A')}\n"
        f"- **OPeNDAP URL:** [{url}]({url}.html)\n"
    )

    return pn.pane.Markdown(info_txt), pn.panel(ds)


def data_links(url, start=None, end=None, step=None):
    if isinstance(start, datetime):
        start = pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S")
        end = pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S")

    query_params = [f"{p}={v}" for p, v in [("start", start), ("end", end), ("step", step)] if v is not None]
    query_string = "&".join(query_params)
    html_url = f"{SETTINGS.server_url}/xview/data?f=html&url={urllib.parse.quote(url)}&{query_string}"
    json_url = f"{SETTINGS.server_url}/xview/data?f=json&url={urllib.parse.quote(url)}&{query_string}"
    csv_url = f"{SETTINGS.server_url}/xview/data?f=csv&url={urllib.parse.quote(url)}&{query_string}"

    html_url = f"[{html_url}]({html_url})"
    json_url = f"[{json_url}]({json_url})"
    csv_url = f"[{csv_url}]({csv_url})"
    txt = f"### Data Links\n" f"- **HTML:** {html_url}\n" f"- **JSON:** {json_url}\n" f"- **CSV:** {csv_url}\n"
    return pn.pane.Markdown(txt)


@dataclass
class Params:
    current_param: str
    param_list: list[str]
    start: datetime | int | None
    end: datetime | int | None
    step: str | int = 1

    def __post_init__(self):
        if isinstance(self.step, str) and self.step.isdigit():
            self.step = int(self.step)
        if isinstance(self.start, str):
            if self.start.isdigit():
                self.start = int(self.start)
            else:
                self.start = pd.to_datetime(self.start)
        if isinstance(self.end, str):
            if self.end.isdigit():
                self.end = int(self.end)
            else:
                self.end = pd.to_datetime(self.end)

        if type(self.start) != type(self.end):
            print(self.start, self.end)
            raise ValueError("Start and end must have the same type")


def query_params(ds):
    params = {
        p: pn.state.session_args[p][0].decode("utf-8")
        for p in ["parameter-name", "start", "end", "step"]
        if pn.state.session_args.get(p)
    }

    time_default = True

    if "start" in params and params["start"].isdigit():
        time_default = False

    if "end" in params and params["end"].isdigit():
        time_default = False

    if "start" not in params and "time" in ds.dims and len(ds.time) > DEFAULT_N_POINTS:
        # If the dataset is large and no start time is provided, set a default start time
        if time_default:
            params["start"] = pd.to_datetime(ds.time.values[-DEFAULT_N_POINTS])
        else:
            params["start"] = -DEFAULT_N_POINTS

    param_list = [v for v in ds.data_vars if "time" in ds[v].dims]
    if "parameter-name" not in params:
        params["parameter-name"] = param_list[0] if param_list else None

    return Params(
        current_param=params["parameter-name"],
        param_list=param_list,
        start=params.get("start", pd.to_datetime(ds.time.min().values) if time_default else 0),
        end=params.get("end", pd.to_datetime(ds.time.max().values) if time_default else len(ds.time)),
        step=pn.state.session_args.get("step", 1),
    )


def create_app():
    url = urllib.parse.unquote(pn.state.session_args.get("url", [None])[0].decode("utf-8"))
    ds = xr.open_dataset(url)

    column = pn.Column(*info(ds, url))

    params = query_params(ds)
    if not params.current_param:
        column.extend([data_links(url)])
    else:
        column.extend(time_plot(utils.bytes_to_str(ds), url, params))

    return column


@pn.cache
def sel(ds, start, end, step):

    if isinstance(start, datetime):
        return ds.sel(time=slice(start, end)).isel(time=slice(None, None, step))

    return ds.isel(time=slice(start, end, step))


@pn.cache
def update_plot(variable_selector, ds, start, end, step):
    return sel(ds[variable_selector], start, end, step).hvplot.scatter(x="time", size=2.5)


@pn.cache
def update_data_preview(ds, start, end, step):
    ds = sel(ds, start, end, step)
    return pn.widgets.Tabulator(
        ds.isel(time=slice(-100, None, None)).to_dataframe()[::-1],
        disabled=True,
        height=400,
        pagination="local",
        page_size=50,
    )


def time_plot(ds: xr.Dataset, url, params: Params):
    variable_selector = pn.widgets.Select(name="Variable", options=params.param_list, value=params.current_param)
    step_slider = pn.widgets.IntSlider(name="plot every n point", value=params.step, start=1, end=100, step=10)

    if isinstance(params.start, datetime):
        start_slider = pn.widgets.DatetimeSlider(
            name="Start Time", start=ds.time.min().values, end=ds.time.max().values, value=params.start
        )
        end_slider = pn.widgets.DatetimeSlider(
            name="End Time", start=ds.time.min().values, end=ds.time.max().values, value=params.end
        )
    else:
        start_slider = pn.widgets.IntSlider(name="Start Range", start=0, end=len(ds.time), step=1, value=params.start)
        end_slider = pn.widgets.IntSlider(name="End Range", start=0, end=len(ds.time), step=1, value=params.end)

    if pn.state.location:
        pn.state.location.sync(start_slider, {"value": "start"})
        pn.state.location.sync(end_slider, {"value": "end"})
        pn.state.location.sync(variable_selector, {"value": "parameter-name"})

    binding_plot = pn.bind(
        update_plot,
        variable_selector=variable_selector,
        ds=ds,
        start=start_slider,
        end=end_slider,
        step=step_slider,
    )
    download_binding = pn.bind(data_links, url=url, start=start_slider, end=end_slider, step=step_slider)
    binding_preview = pn.bind(update_data_preview, ds=ds, start=start_slider, end=end_slider, step=step_slider)

    controls = pn.Row(
        pn.Column("### Controls", variable_selector, step_slider),
        pn.Column("### Time Range", start_slider, end_slider),
    )

    column = pn.Column(download_binding)
    column.extend([controls, binding_plot])
    column.append(pn.pane.Markdown("### Data Preview (max 500 records)"))
    column.append(binding_preview)

    return column
