# xview

`xview` is a FastAPI-based application for visualizing and interacting with datasets using `xarray` and `Panel`. It provides endpoints for rendering data previews and serving data in various formats.

## Features

- **Panel Integration**: Render interactive visualizations using `Panel` and `Bokeh`.
- **Data Subsetting**: Subset datasets based on parameters, time ranges, and step sizes.
- **Flexible Output Formats**: Serve data in JSON, CSV, or HTML formats.
- **FastAPI Endpoints**: Expose RESTful APIs for data visualization and retrieval.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/xview.git
    cd xview
    ```

2. Install dependencies:
    ```bash
    poetry install
    ```

3. Run the application:
    ```bash
    poetry run uvicorn xview.main:app --reload
    ```

## Docker

```bash
docker build . -t xview
docker run -p 8000:8000 -p 5000:5000 xview\
```

## Endpoints

### `/panel`
Renders a `Panel` preview using `xarray`.

#### Query Parameters:
- `url` (str): The OPeNDAP URL endpoint.
- `parameter-name` (str, optional): Data variables to plot.
- `start` (datetime | int, optional): Start time in ISO 8601 or index.
- `end` (datetime | int, optional): End time in ISO 8601 or index.
- `step` (int, optional): Step size for selecting points.


