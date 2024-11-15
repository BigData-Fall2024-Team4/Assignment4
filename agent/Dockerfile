FROM public.ecr.aws/docker/library/python:3.12.0-slim-bullseye
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /asset

ENV POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies including DejaVu fonts
RUN apt-get update && \
    apt-get install -y fonts-dejavu && \
    rm -rf /var/lib/apt/lists/*

# Create directories and set permissions
RUN mkdir -p /asset/chat_outputs /asset/reports && \
    chmod -R 777 /asset/chat_outputs /asset/reports

# Install poetry
RUN pip install poetry

# Copy poetry files first for better caching
COPY pyproject.toml poetry.lock ./
COPY .env ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the application code
COPY ./ ./

# Ensure volume mount points have correct permissions
RUN chmod -R 777 /asset/chat_outputs /asset/reports

CMD ["poetry", "run", "demo"]