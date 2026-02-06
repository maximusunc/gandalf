FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package
COPY pyproject.toml setup.py ./
COPY gandalf/ gandalf/
RUN pip install --no-cache-dir -e .

# Port used by gandalf.main / uvicorn
EXPOSE 6429

# Graph data is expected to be mounted at /data
# Override GANDALF_GRAPH_PATH to match your mount point
ENV GANDALF_GRAPH_PATH=/data/gandalf_mmap
ENV GANDALF_GRAPH_FORMAT=auto

CMD ["uvicorn", "gandalf.server:APP", "--host", "0.0.0.0", "--port", "6429"]
