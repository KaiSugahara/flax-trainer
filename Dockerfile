FROM python:3.12-slim

RUN apt -y update
RUN apt -y install curl git
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local python3 -

ARG YOUR_UID
ARG YOUR_GID
RUN groupadd -f -g ${YOUR_GID} user
RUN useradd -m -s /bin/bash -N -u ${YOUR_UID} -g ${YOUR_GID} -G sudo user
USER user

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN poetry install