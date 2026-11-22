# Slim Python base instead of the full python image used in the original lab
FROM python:3.12-slim

LABEL org.opencontainers.image.title="wine-lab1" \
      org.opencontainers.image.description="Trains and evaluates a wine cultivar classifier"

# unbuffered output so training logs show up right away, no .pyc files, no pip cache
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# install dependencies first so this layer stays cached when only the code changes
COPY src/requirements.txt .
RUN pip install -r requirements.txt

# copy the training / prediction code
COPY src/ .

# don't run the job as root; only the artifacts folder needs to be writable
RUN useradd --create-home --uid 1000 mluser \
    && mkdir -p /app/artifacts \
    && chown mluser:mluser /app/artifacts
USER mluser

# default training settings, override with `docker run -e NAME=value`
ENV TEST_SIZE=0.2 \
    RANDOM_STATE=42 \
    CV_FOLDS=5 \
    ARTIFACT_DIR=/app/artifacts

# train the model when the container starts
CMD ["python", "train.py"]
