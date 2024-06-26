# Use the official Ubuntu base image
FROM ubuntu:latest

# Set environment variables to avoid user interaction during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && apt-get clean

# Clone the GitHub repository
RUN git clone https://github.com/arnaybhatia/AI-Digital-Twin.git /AI-Digital-Twin

# Set the working directory
WORKDIR /AI-Digital-Twin

# Create a virtual environment and activate it
RUN python3 -m venv venv

# Install Python packages using pip within the virtual environment
RUN venv/bin/pip install llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface

# Set the default command to run when starting the container
CMD ["bash"]
