# ComputerXRobot-Bench on a Linux GPU server (e.g. an A100): headless MuJoCo rendering through
# EGL on the NVIDIA driver, PyTorch with CUDA.
#
#   docker build -t computerxrobot .
#   docker run --gpus all --rm -v "$PWD/data:/app/data" -v "$PWD/checkpoints:/app/checkpoints" \
#       -v "$PWD/results:/app/results" -e WORKERS=32 computerxrobot sh scripts/run_gpu.sh
#
# Physics, rendering for evaluation and data collection run on the CPU cores (one process per
# worker); training runs on the GPU.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    SDL_VIDEODRIVER=dummy \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgl1 libglvnd0 libgles2 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "benchmark", "list"]
