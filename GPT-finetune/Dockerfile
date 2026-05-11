# PyTorch + CUDA 12.1 + cuDNN runtime — matches torch>=2.1 in requirements.txt
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

WORKDIR /app

# Install Python deps first so the layer caches across code edits.
# torch is already in the base image; pip will skip reinstall if the constraint is satisfied.
COPY requirements.txt .
# transformers >=5 needs torch >=2.5; this image ships torch 2.4, so cap at 4.x.
# datasets 4.x and accelerate 1.x are fine.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt "transformers>=4.40,<5"

# Copy source last — keeps the deps layer cached
COPY . .

# Default to training; override with `docker run ... python inference.py "..."`
CMD ["python", "finetune.py"]
