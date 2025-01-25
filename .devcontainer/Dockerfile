FROM python:3.11-slim

RUN apt -y update
RUN apt -y install curl git
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local POETRY_VERSION=1.8.5 python3 -

WORKDIR /workspace
COPY pyproject.toml /workspace/pyproject.toml
RUN poetry config virtualenvs.create false && poetry install
