FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

ARG PYTHON_VERSION=3.11

RUN apt-get update && apt-get install -y gcc wget curl git

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create Python virtual environment with uv
RUN /root/.local/bin/uv venv openr1 --python 3.11

COPY requirements.txt .
RUN . openr1/bin/activate && \
    GIT_LFS_SKIP_SMUDGE=1 \
    /root/.local/bin/uv pip install -r requirements.txt --upgrade pip --link-mode=copy --no-build-isolation

RUN . openr1/bin/activate && \
    GIT_LFS_SKIP_SMUDGE=1 \
    /root/.local/bin/uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# RUN . openr1/bin/activate && \
#     GIT_LFS_SKIP_SMUDGE=1 \
#     /root/.local/bin/uv pip install flash-attn --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu126


# Set environment variable to use the openr1 environment
ENV PATH="/app/openr1/bin:$PATH"

COPY train.py /app/
COPY polaris-antiviral-admet-2025_rdkit_properties.json /app/
COPY loss /app/loss
COPY dataset /app/dataset
COPY utils /app/utils
COPY vertex /app/vertex
WORKDIR /app/
ENV PYTHONPATH=/app/src

# Ensure the main environment is activated in ENTRYPOINT
SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["bash", "-c", "source activate main && python train.py \"$@\"", "--"]
# ENTRYPOINT ["bash", "-c", "source activate main && python cache.py \"$@\"", "--"]