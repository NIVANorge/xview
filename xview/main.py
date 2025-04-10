import logging
import uvicorn
import sys

# %%
import xarray as xr
import json
import panel as pn
from typing import Annotated
from bokeh.embed import server_document
from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from xview.viewer import create_app
from xview import utils
import panel as pn
from datetime import datetime

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


@app.get("/panel")
async def xview(
    url: Annotated[str, Query(description="OPeNDAP URL")],
    request: Request,
    param_name: Annotated[str, Query(alias="parameter-name", description="Data variables to plot")] = "",
    start: Annotated[datetime | int | None, Query(description="Start time in ISO 8601 or index")] = None,
    end: Annotated[datetime | int | None, Query(description="End time in ISO 8601 or index")] = None,
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
    start: Annotated[datetime | int, Query(description="Start time in ISO 8601 or index")] = None,
    end: Annotated[datetime | int, Query(description="End time in ISO 8601 or index")] = None,
    step: Annotated[int, Query(description="Step size")] = None,
    f: Annotated[str, Query(description="Output format", pattern="^(html|json|csv)$")] = "html",
    exclude_data: Annotated[bool, Query(alias="exclude-data", description="Exclude data from json output")] = False,
):
    """
    Convert a dataset into csv, json or html.
    The timeSeries and trajectory featureTypes are supported in simple"
    form."

    For large datasets, the exclude-data parameter can be used to exclude the data from the json output.
    This is useful for fetching the size and requesting the data in smaller batches using the start and end index parameters,
    indexing is 0-based. 

    Currently only the time dimension is supported for start, end and step.

    Returns:
        Data in the requested format (json, csv, html) based on the f parameter.
    """
    
    if start is not None and end is not None and type(start) != type(end):
        return Response(content="start and end must be of the same type if both are provided", status_code=400)
    ds = xr.open_dataset(url)
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
