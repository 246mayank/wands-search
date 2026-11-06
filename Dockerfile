# Runs the demo API, or any `wands` command, on CPU. data/ is mounted, not baked in:
#   docker build -t wands-search .
#   docker run --rm -v "$PWD/data:/app/data" -p 8000:8000 wands-search
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY wands ./wands
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY prompts ./prompts
COPY cache ./cache
COPY web ./web
COPY splits.json ./

EXPOSE 8000
ENTRYPOINT ["wands"]
CMD ["serve", "--host", "0.0.0.0"]
