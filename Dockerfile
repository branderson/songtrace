# torch pip wheels bundle their own CUDA runtime libs; only the host NVIDIA
# driver is needed (exposed via the nvidia container runtime), so a plain
# Python base is sufficient — no CUDA base image required.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV KARAOKE_MAKER_CONFIG=/config/config.toml \
    TORCH_HOME=/data/models \
    XDG_CACHE_HOME=/data/models/cache

EXPOSE 8000
CMD ["songtrace", "serve"]
