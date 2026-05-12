import logging
import uvicorn
import sys

# %%
import xarray as xr
import json
import panel as pn
import pandas as pd
from typing import Annotated
from bokeh.embed import server_document
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.templating import Jinja2Templates
from xview.viewer import create_app
from xview import utils
import panel as pn
from datetime import datetime
from pydantic import BeforeValidator
from typing import Union
import cf_xarray
import xarray as xr
from fastapi.responses import Response
from xview.config import SETTINGS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(module)s.%(funcName)s %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app = FastAPI(
    docs_url="/docs",
    title="xview",
    version="v1",
    root_path="/xview"
)

templates = Jinja2Templates(directory="templates")


def parse_slice_param(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, datetime)):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid value for time/index: {value}")

StartEndParam = Annotated[Union[int, datetime, None], BeforeValidator(parse_slice_param)]

@app.get("/panel")
async def xview(
    url: Annotated[str, Query(description="OPeNDAP URL")],
    request: Request,
    param_name: Annotated[str, Query(alias="parameter-name", description="Data variables to plot")] = "",
    start: Annotated[StartEndParam, Query(description="Start time in ISO 8601 or index")] = None,
    end: Annotated[StartEndParam, Query(description="End time in ISO 8601 or index")] = None,
    step: Annotated[int | None, Query(description="Select every step number of points between start and stop")] = None,
):
    """
    Render of a panel preview using xarray.

    The timeSeries and trajectory featureTypes are supported in simple form.

    Returns:
        TemplateResponse: A response object rendering a html template with the embedded Bokeh script.
    """

    query_params = request.query_params
    if start is not None and end is not None and type(start) != type(end):
        return Response(content="start and end must be of the same type if both are provided", status_code=400)

    script = server_document(f"{SETTINGS.bokeh_url}/bokeh/preview", arguments=query_params)
    return templates.TemplateResponse("base.html", {"request": request, "script": script})


@app.get("/data")
async def xdata_url(
    url: Annotated[str, Query(description="OPeNDAP URL")],
    param_name: Annotated[str, Query(alias="parameter-name", description="Data variables in dataset to return, comma separated string")] = None,
    start: Annotated[StartEndParam, Query(description="Start time in ISO 8601 or index")] = None,
    end: Annotated[StartEndParam, Query(description="End time in ISO 8601 or index")] = None,
    step: Annotated[int, Query(description="Step size")] = None,
    f: Annotated[str, Query(description="Output format", pattern="^(html|json|csv)$")] = "html",
    exclude_data: Annotated[bool, Query(alias="exclude-data", description="Exclude data from json output")] = False,
):
    """
    Convert a dataset into csv, json or html.
    The timeSeries, trajectory and timeSeriesProfile featureTypes are supported.

    For large datasets, the exclude-data parameter can be used to exclude the data from the json output.
    This is useful for fetching the size and requesting the data in smaller batches using the start and end index parameters,
    indexing is 0-based. Indexing from the end is supported using negative indices [e.g. -1 is the last point].

    Currently only the time dimension is supported for start, end and step.

    Returns:
        Data in the requested format (json, csv, html) based on the f parameter.
    """

    if start is not None and end is not None and type(start) != type(end):
        return Response(content="start and end must be of the same type if both are provided", status_code=400)

    ds = xr.open_dataset(url)

    if utils.is_ragged_tsp(ds):
        return _ragged_tsp_response(ds, param_name, start, end, f)

    ds = utils.subset(ds, param_name, start, end, step)
    ds = utils.to_json_types(ds, fill_nan=f == "json")
    if f == "json":
        return Response(
            content=json.dumps(ds.to_dict(data=False if exclude_data else "list")),
            media_type="application/json",
        )
    elif f == "csv":
        return Response(content=ds.to_dataframe().to_csv(), media_type="text/csv")
    else:
        return Response(content=ds.to_dataframe().to_html(), media_type="text/html")


def _ragged_tsp_response(ds: xr.Dataset, param_name, start, end, f: str) -> Response:
    """Expand a ragged-array timeSeriesProfile and return the requested format."""
    df = utils.expand_ragged_tsp(ds)

    # Filter columns to requested variables (keep all coordinate-like columns)
    if param_name:
        requested = {v.strip() for v in param_name.split(",")}
        coord_cols = {c for c in df.columns if c not in ds.data_vars or ds[c].attrs.get("cf_role")}
        keep = coord_cols | (requested & set(df.columns))
        df = df[[c for c in df.columns if c in keep]]

    # Filter by time range
    time_col = next((c for c in df.columns if c in ("time", "TIME")), None)
    if time_col and isinstance(start, datetime):
        df[time_col] = pd.to_datetime(df[time_col])
        if start is not None:
            df = df[df[time_col] >= start]
        if end is not None:
            df = df[df[time_col] <= end]

    # Convert datetime columns to ISO strings for JSON/HTML
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")

    if f == "json":
        return Response(
            content=df.to_json(orient="records", date_format="iso"),
            media_type="application/json",
        )
    elif f == "csv":
        return Response(content=df.to_csv(index=False), media_type="text/csv")
    else:
        return Response(content=df.to_html(index=False), media_type="text/html")
    
 
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    Returns:
        A JSON response with a status message.
    """
    return {"status": "ok"}

pn.extension(template="fast")
pn.serve(
    {"/preview": create_app},
    port=5000,
    allow_websocket_origin=[f"{SETTINGS.server_url.split('//')[1]}"],
    address="0.0.0.0",
    show=False,
    prefix="/bokeh",

)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
# poetry run uvicorn xview.main:app --reload
