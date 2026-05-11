# Containerizing GPT-finetune — Architecture & GPU Passthrough

This doc explains **what was added**, **how the host GPU reaches the container**, and **why this is the same model SageMaker uses on EC2**.

---

## 1. What changed

Before containerization, training ran directly on the host Python interpreter — it touched the OS pip env, the host CUDA toolkit, and wrote checkpoints into the project tree.

After containerization, everything except the GPU driver lives inside an immutable image. The host now provides three things only: the kernel, the NVIDIA driver, and a directory to mount for checkpoints.

| Concern               | Before                    | After                                                    |
| --------------------- | ------------------------- | -------------------------------------------------------- |
| Python + torch + CUDA | Installed on host         | Baked into `pytorch/pytorch:2.4.0-cuda12.1-cudnn9` image |
| Dependencies          | `pip install` on host     | Resolved once at `docker build`                          |
| GPU access            | Direct (host process)     | Injected via NVIDIA Container Toolkit                    |
| Checkpoints (`output/`) | Written in project tree | Bind-mounted to host directory                           |
| HF model cache        | `~/.cache/huggingface`    | Named Docker volume `hf-cache` (survives rebuilds)       |
| Reproducibility       | Whatever's on the machine | Image digest = exact env, anywhere                       |

### Files added
- `Dockerfile` — image recipe (base + deps + source).
- `.dockerignore` — keeps `.git/`, `output/`, `__pycache__/` out of the build context.
- `docker-compose.yml` — declares the GPU reservation, volumes, and run command.

### One pin that matters
The base image ships **torch 2.4.0**. The default `transformers` on pip is now 5.x and registers
a `torch.library.custom_op` whose tensor signature 2.4 can't parse. The Dockerfile pins `transformers>=4.40,<5` to dodge this. If you upgrade the base image to torch 2.5+, you can drop the pin.

---

## 2. How CUDA reaches the container

The container does **not** ship a GPU driver. It ships the CUDA *runtime* (`libcudart`, cuDNN, the PyTorch wheels). The *driver* (`libcuda.so`, kernel module) stays on the host. The NVIDIA Container Toolkit splices the two together at container start.

### Layered view

```
+----------------------------------------------------------+
|  Container (gpt-finetune image)                          |
|    - python 3.11, torch 2.4 (CUDA 12.1 runtime), HF      |
|    - finetune.py / inference.py                          |
|    - calls torch.cuda.is_available()  ──┐                |
+-----------------------------------------│----------------+
                                          │ dlopen("libcuda.so")
                                          ▼
+----------------------------------------------------------+
|  /usr/lib/x86_64-linux-gnu/libcuda.so                    |
|  ── BIND-MOUNTED from host at container start ──         |
+-----------------------------------------│----------------+
                                          │ ioctl(/dev/nvidia*)
                                          ▼
+----------------------------------------------------------+
|  WSL2 kernel  → /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm  |
|  (exposed to WSL by the host driver, no kernel module in WSL) |
+-----------------------------------------│----------------+
                                          │
                                          ▼
+----------------------------------------------------------+
|  Windows NVIDIA driver 581.83  ──►  RTX 4060 (physical)  |
+----------------------------------------------------------+
```

### The handshake, step by step

1. **You ask for GPUs.** In `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```
   (Same as `docker run --gpus all` on the CLI.)

2. **Docker Desktop sees that request and invokes `nvidia-container-runtime`** instead of plain `runc`. The runtime is shipped with Docker Desktop on Windows — no separate install.

3. **Before the container's `init` runs**, the runtime executes an OCI **prestart hook** (`nvidia-container-cli`). It does three things:
   - Bind-mounts the host's `libcuda.so` and matching userspace libs into the container.
   - Bind-mounts `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` so the process can `ioctl()` the driver.
   - Adjusts cgroups so the container actually has permission to use those devices.

4. **PyTorch starts.** `import torch; torch.cuda.is_available()` succeeds because:
   - `libcudart.so.12` (runtime, in the image) finds `libcuda.so` (driver, mounted from host).
   - The driver version (581.83 → CUDA 13.0 capable) is ≥ the runtime's required version (12.1). **This direction matters** — host driver must be ≥ container's CUDA runtime. Newer host, older container = fine.

5. **`finetune.py:27 get_device()` returns `"cuda"`** and training runs on the host GPU. We confirmed this end-to-end: training ran to completion in 21:30 on the RTX 4060, with ~4 GB of GPU memory in use on the host.

### Why the container can be "thin"
The CUDA runtime is ~1 GB; the driver is hundreds of MB. By keeping the driver on the host and only mounting it in, the toolkit avoids:
- Shipping driver binaries per-image (would need re-release every driver update).
- Driver/kernel-module version skew (an old container would refuse a new kernel).

This is the same reason you can run a CUDA 11 container on a CUDA 13 host driver, but **not** the reverse.

---

## 3. The full compute stack — from Python call to VRAM

The layered view in section 2 stops at the device nodes. Here's what *actually* happens all the way down to the silicon when `model.to("cuda")` or `loss.backward()` runs. This applies whether you're inside a container or on bare metal — the NVIDIA Container Toolkit hook just bridges the container's userspace to the host-side parts of this same stack.

```
┌──────────────────────────────────────────────────────────────────────┐
│  USERSPACE  (runs on CPU, inside the container process)              │
├──────────────────────────────────────────────────────────────────────┤
│  1. Python / PyTorch application                                     │
│     loss = model(x);  loss.backward()                                │
│                              │                                       │
│                              ▼                                       │
│  2. PyTorch C++ dispatcher                                           │
│     selects the CUDA backend  →  torch._C._cuda_*                    │
│                              │                                       │
│                              ▼                                       │
│  3. CUDA Runtime API  (libcudart.so, libcublas.so, libcudnn.so)      │
│     cudaMalloc, cudaMemcpyAsync, cuBLAS gemm, cuDNN conv2d           │
│     ─── shipped INSIDE the image ───                                 │
│                              │                                       │
│                              ▼                                       │
│  4. CUDA Driver API   (libcuda.so)                                   │
│     cuMemAlloc, cuLaunchKernel, cuStreamSynchronize                  │
│     ─── BIND-MOUNTED from the host driver install ───                │
└──────────────────────────────│───────────────────────────────────────┘
                               │  syscall:  ioctl(/dev/nvidia0, ...)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  KERNEL SPACE  (host Linux / WSL2 kernel — NEVER inside container)   │
├──────────────────────────────────────────────────────────────────────┤
│  5. NVIDIA kernel modules                                            │
│     - nvidia.ko       → command submission, context management      │
│     - nvidia-uvm.ko   → Unified Virtual Memory, page migration      │
│     - nvidia-drm.ko   → display (irrelevant for pure compute)       │
│                                                                      │
│     Translates ioctl() requests into:                                │
│       • DMA buffer setup in pinned host RAM                          │
│       • Command-ring writes targeting the GPU's host channel         │
│       • IRQ handling when the GPU signals kernel completion          │
└──────────────────────────────│───────────────────────────────────────┘
                               │  PCIe (MMIO writes + DMA reads/writes)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  GPU HARDWARE  (RTX 4060 in our case)                                │
├──────────────────────────────────────────────────────────────────────┤
│  6. Host interface + command processor (front-end)                   │
│     Pulls commands from the ring buffer, dispatches to the SMs.      │
│                              │                                       │
│                              ▼                                       │
│  7. Streaming Multiprocessors (SMs) — the actual compute cores       │
│     • Warps of 32 threads execute in lock-step (SIMT)                │
│     • Tensor Cores: fused mat-mul-accumulate (FP16/BF16/FP8/INT8)    │
│     • Register file + L1 cache + shared memory per SM                │
│                              │                                       │
│                              ▼                                       │
│  8. L2 cache (shared across all SMs)                                 │
│                              │                                       │
│                              ▼                                       │
│  9. VRAM   (GDDR6 — 8 GB on RTX 4060, ~272 GB/s bandwidth)           │
│     Holds: model weights, optimizer state, activations, gradients.   │
│     Allocated by cuMemAlloc, freed by cuMemFree.                     │
│     When you see `torch.cuda.OutOfMemoryError`, this is full.        │
└──────────────────────────────────────────────────────────────────────┘
```

### A round-trip in plain English

When `loss.backward()` runs on a tensor that lives on the GPU:

1. **PyTorch (on the CPU)** walks the autograd graph and queues a sequence of kernel launches — one per op (matmul, bias-add, softmax, …).
2. **`libcudart`** (inside the image) translates each `cudaLaunchKernel` into the lower-level `cuLaunchKernel` exposed by **`libcuda.so`**.
3. **`libcuda.so`** (from the host, bind-mounted in) issues an `ioctl()` on `/dev/nvidia0`. This is the userspace → kernel boundary; it's the last line of code that runs inside the container.
4. **`nvidia.ko`** (host kernel module) writes a command packet into a ring buffer in pinned host RAM and rings a doorbell register on the GPU over PCIe.
5. **GPU front-end** sees the doorbell, fetches the command, dispatches the kernel to one or more SMs.
6. **SMs** read inputs from VRAM (via L2 → L1 → registers), compute, write outputs back to VRAM. Tensor Cores accelerate the matmul.
7. When the kernel finishes, the GPU raises an interrupt; **`nvidia.ko`** signals completion back to the userspace stream.
8. **PyTorch's stream sync** (or the next op that depends on the result) sees the event and the Python code continues.

### What lives where

| Layer | Source | In container? | In host kernel? |
|---|---|:---:|:---:|
| PyTorch, `torch.cuda.*` | `pip install torch` | ✅ | — |
| `libcudart`, `libcublas`, `libcudnn` | base image | ✅ | — |
| `libcuda.so` (driver API userspace) | host driver install | ✅ (bind-mount) | — |
| `/dev/nvidia*` device nodes | created by host driver | ✅ (bind-mount) | — |
| `nvidia.ko`, `nvidia-uvm.ko` | host driver install | ❌ | ✅ |
| Command rings, pinned DMA buffers | allocated by kernel module | — | ✅ |
| VRAM allocations | `cuMemAlloc` over PCIe | — (on the GPU) | — |

The container only ever holds the **top half** of this stack. Everything from the `ioctl()` boundary downward is the host's responsibility. That's why:

- A single host driver can serve many containers simultaneously.
- Containers never become "too old for the hardware" — they carry zero GPU-specific kernel code, so a new GPU plus a new host driver just works with the same old image.
- You can't run a CUDA container on a host without an NVIDIA driver, no matter how new the image is — the kernel-space pieces are non-negotiable.

---

## 4. The SageMaker analogy

SageMaker training jobs are not a different paradigm — they are the same Docker + NVIDIA Container Toolkit stack, just with AWS managing the host.

### Mapping

| Local setup (what we just built)              | SageMaker equivalent                                                                |
| --------------------------------------------- | ----------------------------------------------------------------------------------- |
| Your Windows 11 box                           | An EC2 instance (e.g. `ml.g5.xlarge` with A10G, `ml.p4d.24xlarge` with A100)        |
| WSL2 + Docker Desktop                         | Amazon Linux 2 + containerd (managed by SageMaker)                                  |
| NVIDIA driver 581.83 on the host              | NVIDIA driver pre-installed on the AMI by AWS                                       |
| NVIDIA Container Toolkit (Docker Desktop)     | Same toolkit, pre-wired on SageMaker hosts                                          |
| `pytorch/pytorch:...-cuda12.1-cudnn9-runtime` | SageMaker DLC (Deep Learning Container) — AWS publishes the same kind of image     |
| `docker compose run --rm finetune`            | SageMaker `Estimator.fit()` → starts a job, pulls your image, runs `train` cmd     |
| `./output:/app/output` bind mount             | `/opt/ml/model` inside container, auto-uploaded to S3 at job end                    |
| `./data/train_reasoning.txt`                  | An S3 "input channel" mounted at `/opt/ml/input/data/training/`                     |
| `hf-cache` named volume                       | Optional FSx for Lustre or S3-mounted cache for shared model weights                |
| `docker build` (your machine)                 | `docker build && docker push` to ECR (Elastic Container Registry)                   |
| CUDA exposure: `--gpus all` + toolkit hook    | **Identical** — SageMaker invokes `nvidia-container-runtime` with the same hooks   |

### What's actually identical at the GPU layer

When SageMaker spins up an `ml.g5.2xlarge` and launches your training container, the **exact same sequence** runs:

1. AWS provisions an EC2 host with an NVIDIA driver installed in the AMI.
2. The SageMaker agent on the host pulls your image from ECR.
3. It runs `docker run --gpus all <your-image>` (effectively).
4. `nvidia-container-runtime` fires its prestart hook.
5. `libcuda.so` from the EC2 host is bind-mounted into your container.
6. `/dev/nvidia*` devices appear inside the container.
7. Your `torch.cuda.is_available()` returns True.

Steps 4–7 are byte-for-byte the same code path as what just happened on your laptop. The only differences are:

- **Where the host lives** (your machine vs. AWS's data center).
- **Who manages the driver** (you/Windows Update vs. AWS via the AMI).
- **Where the image is pulled from** (local Docker daemon vs. ECR).
- **Lifecycle** (you run `docker run`; SageMaker tears the host down when the job finishes).

### What SageMaker adds on top

- **Image distribution** (ECR push/pull).
- **Auto-scaling** (one container per node, multi-node distributed training via NCCL).
- **S3 ↔ container filesystem plumbing** (`/opt/ml/input/data/...`, `/opt/ml/model`, `/opt/ml/checkpoints`).
- **Job lifecycle** (provision → run → de-provision, billed per second).
- **Spot instance handling, IAM, VPC, CloudWatch logs.**

None of those touch the GPU passthrough mechanism. If you wanted to lift this exact project to SageMaker, the steps would be:

1. Replace `config.DATA_FILE` reads with reads from `/opt/ml/input/data/training/`.
2. Have `finetune.py` save to `/opt/ml/model` instead of `output/`.
3. `docker build` and `docker push` the image to ECR.
4. Define a SageMaker `Estimator(image_uri=..., instance_type="ml.g5.xlarge", ...)` and call `.fit({"training": "s3://bucket/data/"})`.

The Dockerfile barely changes. The GPU layer doesn't change at all.

---

## 5. Source code is baked into the image, not uploaded at run time

A common point of confusion: when you run

```powershell
docker compose run --rm finetune python inference.py "What is deep learning?"
```

…it **looks** like you're passing `inference.py` *into* the container. You aren't. The script already lives inside the image at `/app/inference.py` — it was copied there during `docker build` by this line in the Dockerfile:

```dockerfile
COPY . .
```

What `docker compose run` does is take everything after the service name (`python inference.py "..."`) and use it to **override the image's default `CMD`**, which is `["python", "finetune.py"]`. Only the command *string* crosses the host→container boundary. The script itself is the frozen copy inside the image.

### Visual

```
┌──────────────────── Host ────────────────────┐
│  GPT-finetune/                               │
│   ├── inference.py   ← edited on host        │
│   ├── finetune.py                            │
│   └── docker-compose.yml                     │
└──────────────────────────────────────────────┘
                    │
                    │  docker compose run --rm finetune \
                    │       python inference.py "..."
                    ▼
              (only the *command string* crosses)
                    │
                    ▼
┌─────────── Container (gpt-finetune image) ───┐
│  /app/                                       │
│   ├── inference.py   ← frozen at build time  │
│   ├── finetune.py                            │
│   ├── config.py                              │
│   └── data/                                  │
│                                              │
│  $ python inference.py "..."   ◄── runs THIS │
└──────────────────────────────────────────────┘
```

### The consequence

If you **edit `inference.py` or `config.py` on the host**, the running container won't see the change. Two ways to handle it:

1. **Rebuild** — fast because the pip layer is cached:
   ```powershell
   docker compose -f GPT-finetune/docker-compose.yml build
   ```

2. **Bind-mount the source over `/app`** — edits go live with no rebuild:
   ```powershell
   docker run --rm --gpus all `
     -v ${PWD}/GPT-finetune:/app `
     -v ${PWD}/GPT-finetune/output:/app/output `
     gpt-finetune python inference.py "..."
   ```
   The bind shadows the baked-in copy with the host directory.

Day-to-day prompt iteration: the baked-in version is fine. Iterating on Python code: bind-mount is faster.

### Why "baked in" is the default

This is the same trade-off SageMaker enforces: an image is a **versioned, hash-identified artifact**. You pin code into the image at build time so that a job that runs today is the same job that runs in six months from the same `image_uri`. Bind-mounting host files is a dev-loop convenience; production training jobs should run from a built image so the code is reproducible alongside the dependencies.

---

## 6. tl;dr

- The container is **just a packaged Python+CUDA-runtime environment**.
- The host driver is the only thing that talks to the GPU.
- A request for `--gpus all` triggers a runtime hook that mounts the host driver into the container.
- SageMaker is "this same thing, but the host is an EC2 box AWS rented to you for the duration of the job."
