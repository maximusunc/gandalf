# Use RENCI python base image
FROM ghcr.io/translatorsri/renci-python-image:3.11.5

# Add image info
LABEL org.opencontainers.image.source https://github.com/BioPack-team/shepherd

ENV PYTHONHASHSEED=0

# Prevent numpy/BLAS from spawning excessive threads that compete with the event loop
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Graph loading configuration (mount mmap directory to this path)
ENV GANDALF_GRAPH_PATH=/app/graph
ENV GANDALF_GRAPH_FORMAT=auto

# set up requirements
WORKDIR /app

# make sure all is writeable for the nru USER later on
RUN chmod -R 777 .

# Install requirements
COPY ./shepherd_utils ./shepherd_utils
COPY ./pyproject.toml .
RUN pip install .

COPY ./workers/gandalf/requirements.txt .
RUN pip install -r requirements.txt

# switch to the non-root user (nru). defined in the base image
USER nru

# Copy in files
COPY ./workers/gandalf ./

# Variables that can be overridden
CMD ["python", "worker.py"]
