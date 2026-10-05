# Use a lightweight Python base image
FROM python:3.9-slim

# Install Docker CLI to allow the Orchestrator to spawn dynamic sibling containers
RUN apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency list and install (Create a requirements.txt with tensorflow, pandas, scipy, requests)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Synapse-AI project into the container
COPY . .

# Set the Orchestrator as the entry point
CMD ["python", "core/agent.py"]