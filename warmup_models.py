"""
Warmup script: load all heavy models into VRAM in parallel before handler accepts jobs.
Runs as a background process during container boot.

This bypasses ComfyUI's sequential node execution — loads PuLID, InsightFace,
EVA-CLIP, Flux UNET, T5, and VAE concurrently using threads.
"""
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='[warmup] %(message)s')
log = logging.getLogger(__name__)

MODELS_DIR = "/comfyui/models"


def load_flux_unet():
    """Load Flux Dev FP8 diffusion model (~11GB) into GPU"""
    t0 = time.time()
    import torch
    from safetensors.torch import load_file
    path = os.path.join(MODELS_DIR, "diffusion_models/flux1-dev-fp8.safetensors")
    sd = load_file(path, device="cuda")
    log.info(f"Flux UNET: {len(sd)} keys loaded in {time.time()-t0:.1f}s")
    del sd
    torch.cuda.empty_cache()


def load_t5_clip():
    """Load T5-XXL FP8 + CLIP-L text encoders"""
    t0 = time.time()
    from safetensors.torch import load_file
    t5_path = os.path.join(MODELS_DIR, "clip/t5xxl_fp8_e4m3fn.safetensors")
    clip_path = os.path.join(MODELS_DIR, "clip/clip_l.safetensors")
    sd1 = load_file(t5_path, device="cuda")
    sd2 = load_file(clip_path, device="cuda")
    log.info(f"T5+CLIP-L: {len(sd1)+len(sd2)} keys loaded in {time.time()-t0:.1f}s")
    del sd1, sd2
    import torch; torch.cuda.empty_cache()


def load_vae():
    """Load VAE (~300MB)"""
    t0 = time.time()
    from safetensors.torch import load_file
    path = os.path.join(MODELS_DIR, "vae/ae.safetensors")
    sd = load_file(path, device="cuda")
    log.info(f"VAE: {len(sd)} keys loaded in {time.time()-t0:.1f}s")
    del sd
    import torch; torch.cuda.empty_cache()


def load_eva_clip_visual():
    """Load pre-converted EVA-CLIP visual safetensors"""
    t0 = time.time()
    from safetensors.torch import load_file
    path = os.path.join(MODELS_DIR, "clip/EVA02_CLIP_L_336_visual.safetensors")
    if os.path.exists(path):
        sd = load_file(path, device="cuda")
        log.info(f"EVA-CLIP visual: {len(sd)} keys loaded in {time.time()-t0:.1f}s")
        del sd
    else:
        log.warning(f"EVA-CLIP visual safetensors not found at {path}, skipping")
    import torch; torch.cuda.empty_cache()


def load_insightface():
    """Initialize InsightFace with only detection + recognition"""
    t0 = time.time()
    from insightface.app import FaceAnalysis
    insightface_dir = os.path.join(MODELS_DIR, "insightface")
    model = FaceAnalysis(
        name="antelopev2",
        root=insightface_dir,
        allowed_modules=['detection', 'recognition'],
        providers=['CUDAExecutionProvider']
    )
    model.prepare(ctx_id=0, det_size=(640, 640))
    log.info(f"InsightFace: loaded in {time.time()-t0:.1f}s")
    del model


def load_pulid():
    """Load PuLID safetensors"""
    t0 = time.time()
    from safetensors.torch import load_file
    path = os.path.join(MODELS_DIR, "pulid/pulid_flux_v0.9.1.safetensors")
    sd = load_file(path, device="cuda")
    log.info(f"PuLID: {len(sd)} keys loaded in {time.time()-t0:.1f}s")
    del sd
    import torch; torch.cuda.empty_cache()


def main():
    t_start = time.time()
    log.info("Starting parallel model warmup...")

    # All loaders are independent — run them concurrently
    loaders = [
        ("flux_unet", load_flux_unet),
        ("t5_clip", load_t5_clip),
        ("vae", load_vae),
        ("eva_clip", load_eva_clip_visual),
        ("insightface", load_insightface),
        ("pulid", load_pulid),
    ]

    with ThreadPoolExecutor(max_workers=len(loaders)) as pool:
        futures = {pool.submit(fn): name for name, fn in loaders}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                log.error(f"{name} failed: {e}")

    log.info(f"All models warmed up in {time.time()-t_start:.1f}s (parallel)")


if __name__ == "__main__":
    main()
