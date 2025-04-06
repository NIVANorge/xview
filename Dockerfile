FROM python:3.11-slim AS base

RUN pip install poetry
RUN poetry config virtualenvs.create false

WORKDIR /app

COPY templates ./templates

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --no-root --without dev

COPY xview xview/

RUN poetry build 

FROM base AS runtime

COPY --from=base /app/dist/ /app/dist/
RUN pip install /app/dist/*.whl

FROM runtime AS test

RUN poetry install --no-root --with dev
COPY tests tests/

RUN pytest .

FROM runtime AS prod

# Fail if test stage fails
COPY --from=test test-res* test-result

ARG GIT_COMMIT_ID=unknown
LABEL git_commit_id=$GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

ENTRYPOINT ["uvicorn", "xview.main:app", "--host", "0.0.0.0"]