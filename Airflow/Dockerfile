FROM apache/airflow:2.10.2

USER root

# Install system dependencies
RUN apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false && \
    apt-get install -y \
    chromium \
    chromium-driver \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directory for Chrome
RUN mkdir -p /opt/chrome && \
    chmod -R 777 /opt/chrome

USER airflow

# Install Python packages
RUN pip install --no-cache-dir \
    pandas \
    boto3 \
    requests \
    webdriver_manager \
    cryptography \
    ntplib \
    docling==2.5.2 \
    python-dateutil \
    tqdm \
    pinecone-client \
    langchain \
    openai

# Set environment variables
ENV PYTHONPATH=/opt/airflow/dags \
    DISPLAY=:99 \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver \
    CHROME_BINARY_PATH=/usr/bin/chromium \
    AWS_DEFAULT_REGION=us-east-1

# Start Xvfb
RUN Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &

WORKDIR /opt/airflow
