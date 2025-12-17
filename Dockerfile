FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && apt-get clean

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl

# Install mini-swe-agent
# We assume the build context is the mini-swe-agent root
COPY . /app/mini-swe-agent
# Install in editable mode or just install
RUN pip install /app/mini-swe-agent

# Install runtime dependencies for the runner
RUN pip install requests litellm

# Copy the runner script
COPY mini_swe_runner.py /app/mini_swe_runner.py

# Env vars ensuring python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Entrypoint
CMD ["python", "/app/mini_swe_runner.py"]
