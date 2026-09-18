# Supply a reviewed Python image pinned by digest. No image was pulled or built here.
# Example shape: --build-arg BASE_IMAGE=python:3.12-slim@sha256:<real-digest>
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY src/substrate /app/src/substrate
# The closed IR worker uses only stdlib. Signature code imports cryptography lazily.
# Install source in site-packages so python -I does not depend on PYTHONPATH or cwd.
RUN python -c "import shutil,site; shutil.copytree('/app/src/substrate',site.getsitepackages()[0]+'/substrate')"
USER 65534:65534
CMD ["python", "-I", "-m", "substrate.sandbox"]
