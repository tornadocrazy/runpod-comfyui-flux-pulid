# build v3
# syntax=docker/dockerfile:1.6
FROM runpod/worker-comfyui:5.4.1-base

# ─────────────────────────────────────────────────────────────────────────────
# System packages
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip git && \
    rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
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
RUN git clone --depth 1 https://github.com/cubiq/PuLID_ComfyUI \
        /comfyui/custom_nodes/ComfyUI-PuLID_Flux_II && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes \
        /comfyui/custom_nodes/ComfyUI-KJNodes && \
    git clone --depth 1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts \
        /comfyui/custom_nodes/ComfyUI-Custom-Scripts && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-Florence2 \
        /comfyui/custom_nodes/ComfyUI-Florence2 && \
    pip install --no-cache-dir \
        -r /comfyui/custom_nodes/ComfyUI-Florence2/requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Models — all downloads in one layer with HF cache mount
# clip_l, t5xxl, EVA-CLIP, flux1-dev-fp8, ae (VAE), pulid, buffalo_l
# ─────────────────────────────────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors \
        --relative-path models/clip --filename clip_l.safetensors && \
    comfy model download \
        --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors \
        --relative-path models/clip --filename t5xxl_fp16.safetensors && \
    comfy model download \
        --url https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_E_psz14_s4B.pt \
        --relative-path models/clip --filename EVA02_CLIP_E_psz14_s4B.pt && \
    comfy model download \
        --url https://huggingface.co/kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors \
        --relative-path models/diffusion_models --filename flux1-dev-fp8.safetensors && \
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

# OPTION B — full bf16  (uncomment to activate, comment out fp8 download above)
# ARG HF_TOKEN
# RUN mkdir -p /comfyui/models/diffusion_models && \
#     wget -q --show-progress \
#         --header="Authorization: Bearer ${HF_TOKEN}" \
#         "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
#         -O /comfyui/models/diffusion_models/flux1-dev.safetensors
