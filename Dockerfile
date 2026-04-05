FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder


RUN apt-get update && apt-get install -y git curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --frozen || uv sync --no-install-project
COPY . .
RUN uv sync

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]