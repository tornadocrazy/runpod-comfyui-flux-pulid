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
    transformers>=4.49.0 \
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
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack \
        /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    cd /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    git submodule update --init --recursive && \
    python install.py || echo "Impact-Pack install.py failed, continuing to pip requirements" && \
    pip install --no-cache-dir \
        -r /comfyui/custom_nodes/ComfyUI-PuLID_Flux_II/requirements.txt \
        -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt \
        -r /comfyui/custom_nodes/ComfyUI-Florence2/requirements.txt \
        -r /comfyui/custom_nodes/ComfyUI-Impact-Pack/requirements.txt \
        piexif \
        facenet-pytorch --no-deps

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
    unzip /comfyui/models/insightface/models/buffalo_l.zip \
          -d /comfyui/models/insightface/models/buffalo_l && \
    rm /comfyui/models/insightface/models/buffalo_l.zip

# t5xxl_fp8 (~4.5 GB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors \
        --relative-path models/clip --filename t5xxl_fp8_e4m3fn.safetensors

# EVA-CLIP (~4 GB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_E_psz14_s4B.pt \
        --relative-path models/clip --filename EVA02_CLIP_E_psz14_s4B.pt

# flux1-dev-fp8 (~11 GB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors \
        --relative-path models/diffusion_models --filename flux1-dev-fp8.safetensors

# OPTION B — full bf16  (uncomment to activate, comment out fp8 download above)
# ARG HF_TOKEN
# RUN mkdir -p /comfyui/models/diffusion_models && \
#     wget -q --show-progress \
#         --header="Authorization: Bearer ${HF_TOKEN}" \
#         "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
#         -O /comfyui/models/diffusion_models/flux1-dev.safetensors
