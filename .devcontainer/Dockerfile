FROM python:3.11-slim

RUN apt -y update
RUN apt -y install curl git
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local python3 -

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY flax_trainer /workspace/flax_trainer
RUN poetry config virtualenvs.create false && poetry install
