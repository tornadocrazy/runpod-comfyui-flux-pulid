# syntax=docker/dockerfile:1.6
# build v4 — swap t5xxl_fp16 (9.1 GB) to t5xxl_fp8 (4.5 GB) to stay under 30 min build limit
FROM runpod/worker-comfyui:5.4.1-base

# ─────────────────────────────────────────────────────────────────────────────
# System packages
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip git && \
    rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Python dependencies
# ────────────────────────────────────────────────────────────────────────────
# Install insightface pre-built wheel (avoids C compilation issues) + GPU onnxruntime
# transformers ≥ 4.49 is required by Florence2 (fixes missing is_flash_attn import)
RUN pip install --no-cache-dir \
    https://huggingface.co/iwr-redmond/linux-wheels/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl \
    onnxruntime-gpu==1.20.0 \
    "transformers>=4.49.0" \
    facexlib \
    timm \
    einops

# ─────────────────────────────────────────────────────────────────────────────
# Custom nodes (all cloned in one layer)
# ─────────────────────────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/lldacing/ComfyUI_PuLID_Flux_ll \
        /comfyui/custom_nodes/ComfyUI-PuLID_Flux_II && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes \
        /comfyui/custom_nodes/ComfyUI-KJNodes && \
    cd /comfyui/custom_nodes/ComfyUI-KJNodes && \
    git fetch --depth 1 origin e64b67b8f4aa3a555cec61cf18ee7d1cfbb3e5f0 && \
    git checkout FETCH_HEAD && \
    cd /comfyui && \
    git clone --depth 1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts \
        /comfyui/custom_nodes/ComfyUI-Custom-Scripts && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-Florence2 \
        /comfyui/custom_nodes/ComfyUI-Florence2 && \
    pip install --no-cache-dir \
        -r /comfyui/custom_nodes/ComfyUI-PuLID_Flux_II/requirements.txt \
        -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt \
        -r /comfyui/custom_nodes/ComfyUI-Florence2/requirements.txt \
        facenet-pytorch --no-deps && \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.20.0

# ReActor (face swap) + RMBG (background removal)
RUN comfy-node-install comfyui-reactor comfyui-rmbg

# Bypass ReActor NSFW filter (downloads large classifier model otherwise)
COPY reactor_sfw.py /comfyui/custom_nodes/comfyui-reactor/scripts/reactor_sfw.py

# ─────────────────────────────────────────────────────────────────────────────
# Models — split into separate layers so Docker caches each independently
# ─────────────────────────────────────────────────────────────────────────────

# Small models (~250 MB + ~300 MB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors \
        --relative-path models/clip --filename clip_l.safetensors && \
    comfy model download \
        --url https://huggingface.co/lovis93/testllm/resolve/ed9cf1af7465cebca4649157f118e331cf2a084f/ae.safetensors \
        --relative-path models/vae --filename ae.safetensors && \
    comfy model download \
        --url https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors \
        --relative-path models/pulid --filename pulid_flux_v0.9.1.safetensors && \
    comfy model download \
        --url https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
        --relative-path models/insightface/models --filename buffalo_l.zip && \
    comfy model download \
        --url https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip \
        --relative-path models/insightface/models --filename antelopev2.zip && \
    unzip /comfyui/models/insightface/models/buffalo_l.zip \
          -d /comfyui/models/insightface/models/buffalo_l && \
    unzip /comfyui/models/insightface/models/antelopev2.zip \
          -d /comfyui/models/insightface/models && \
    rm /comfyui/models/insightface/models/buffalo_l.zip \
       /comfyui/models/insightface/models/antelopev2.zip

# t5xxl_fp8 (~4.5 GB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors \
        --relative-path models/clip --filename t5xxl_fp8_e4m3fn.safetensors

# EVA-CLIP (~817 MB .pt) — downloaded then converted to visual-only safetensors at build time
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt \
        --relative-path models/clip --filename EVA02_CLIP_L_336_psz14_s6B.pt

# P1: Convert EVA-CLIP .pt (pickle) -> visual-only .safetensors (10-15s faster cold start)
COPY convert_eva_clip.py /tmp/convert_eva_clip.py
RUN python3 /tmp/convert_eva_clip.py \
        /comfyui/models/clip/EVA02_CLIP_L_336_psz14_s6B.pt \
        /comfyui/models/clip/EVA02_CLIP_L_336_visual.safetensors && \
    rm /comfyui/models/clip/EVA02_CLIP_L_336_psz14_s6B.pt && \
    rm /tmp/convert_eva_clip.py

# facexlib weights
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
        --relative-path models/facexlib --filename detection_Resnet50_Final.pth && \
    comfy model download \
        --url https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth \
        --relative-path models/facexlib --filename parsing_bisenet.pth

# ReActor: inswapper face swap model + face restoration
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
        --relative-path models/insightface --filename inswapper_128.onnx && \
    comfy model download \
        --url https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth \
        --relative-path models/facerestore_models --filename GFPGANv1.4.pth

# BiRefNet RMBG (background removal)
# Small text files use wget (comfy model download fails on chunked transfers)
RUN mkdir -p /comfyui/models/RMBG/BiRefNet && \
    wget -q -O /comfyui/models/RMBG/BiRefNet/birefnet.py \
        "https://huggingface.co/1038lab/BiRefNet/raw/main/birefnet.py" && \
    wget -q -O /comfyui/models/RMBG/BiRefNet/BiRefNet_config.py \
        "https://huggingface.co/1038lab/BiRefNet/raw/main/BiRefNet_config.py" && \
    wget -q -O /comfyui/models/RMBG/BiRefNet/config.json \
        "https://huggingface.co/1038lab/BiRefNet/resolve/main/config.json"
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/1038lab/BiRefNet/resolve/main/BiRefNet-HR.safetensors \
        --relative-path models/RMBG/BiRefNet --filename BiRefNet-general.safetensors

# flux1-dev-fp8 (~11 GB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors \
        --relative-path models/diffusion_models --filename flux1-dev-fp8.safetensors

# ─────────────────────────────────────────────────────────────────────────────
# Patches — P1/P2/P3/P5: optimize PuLID cold-start loading
# ─────────────────────────────────────────────────────────────────────────────
COPY patch_pulid.sh /tmp/patch_pulid.sh
RUN chmod +x /tmp/patch_pulid.sh && /tmp/patch_pulid.sh && rm /tmp/patch_pulid.sh

COPY warmup_models.py /warmup_models.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

# OPTION B — full bf16  (uncomment to activate, comment out fp8 download above)
# ARG HF_TOKEN
# RUN mkdir -p /comfyui/models/diffusion_models && \
#     wget -q --show-progress \
#         --header="Authorization: Bearer ${HF_TOKEN}" \
#         "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
#         -O /comfyui/models/diffusion_models/flux1-dev.safetensors
