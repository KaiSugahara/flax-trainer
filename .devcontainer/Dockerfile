FROM python:3.11-slim

RUN apt -y update
RUN apt -y install curl git
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local python3 -

RUN groupadd -f -g 1234 user    # dummy UID
RUN useradd -m -s /bin/bash -N -u 1234 -g 1234 -G sudo user # dummy UID & GID
USER user

WORKDIR /workspace
COPY pyproject.toml /workspace/pyproject.toml
RUN poetry install
