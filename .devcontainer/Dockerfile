FROM mcr.microsoft.com/devcontainers/python:3.10

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    rsync \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
