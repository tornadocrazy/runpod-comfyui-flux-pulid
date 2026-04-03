#!/usr/bin/env bash
# Patch ComfyUI-PuLID_Flux_II for faster cold-start loading
set -e

PULID_DIR="/comfyui/custom_nodes/ComfyUI-PuLID_Flux_II"

# ─────────────────────────────────────────────────────────────────────────────
# P1: Patch factory.py — load safetensors instead of .pt pickle
# ─────────────────────────────────────────────────────────────────────────────
# Add safetensors import at top of factory.py
sed -i '9a\from safetensors.torch import load_file as safetensors_load_file' "$PULID_DIR/eva_clip/factory.py"

# Replace the load_state_dict function to try safetensors first
# We patch the torch.load line to check for .safetensors file first
sed -i 's|checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)|# Try safetensors first (much faster than pickle)\n        safetensors_path = checkpoint_path.rsplit(".", 1)[0] + ".safetensors" if not checkpoint_path.endswith(".safetensors") else checkpoint_path\n        import os as _os\n        if _os.path.exists(safetensors_path) and safetensors_path != checkpoint_path:\n            logging.info(f"Loading safetensors (fast path): {safetensors_path}")\n            return safetensors_load_file(safetensors_path, device=map_location)\n        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)|' "$PULID_DIR/eva_clip/factory.py"

# ─────────────────────────────────────────────────────────────────────────────
# P1+P3: Patch pulidflux.py — load visual-only safetensors directly,
#         skip building full CLIP model with text tower
# ─────────────────────────────────────────────────────────────────────────────
# Replace the load_eva_clip method to use visual-only safetensors when available
python3 -c "
import re

with open('$PULID_DIR/pulidflux.py', 'r') as f:
    content = f.read()

# Replace the load_eva_clip method
old_method = '''    def load_eva_clip(self):
        from .eva_clip.factory import create_model_and_transforms

        clip_file_path = folder_paths.get_full_path(\"text_encoders\", 'EVA02_CLIP_L_336_psz14_s6B.pt')
        if clip_file_path is None:
            clip_dir = os.path.join(folder_paths.models_dir, \"clip\")
        else:
            clip_dir = os.path.dirname(clip_file_path)
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True, local_dir=clip_dir)

        model = model.visual'''

new_method = '''    def load_eva_clip(self):
        from .eva_clip.factory import create_model_and_transforms

        clip_file_path = folder_paths.get_full_path(\"text_encoders\", 'EVA02_CLIP_L_336_psz14_s6B.pt')
        if clip_file_path is None:
            clip_dir = os.path.join(folder_paths.models_dir, \"clip\")
        else:
            clip_dir = os.path.dirname(clip_file_path)

        # Fast path: load visual-only safetensors directly (skips text tower + pickle)
        visual_safetensors = os.path.join(clip_dir, 'EVA02_CLIP_L_336_visual.safetensors')
        if os.path.exists(visual_safetensors):
            logging.info(f'Loading pre-converted EVA-CLIP visual weights (fast path)')
            from safetensors.torch import load_file as _st_load
            from .eva_clip.factory import get_model_config
            import json

            model_cfg = get_model_config('EVA02-CLIP-L-14-336')
            from .eva_clip.model import _build_vision_tower
            model = _build_vision_tower(model_cfg['embed_dim'], model_cfg['vision_cfg'])
            state_dict = _st_load(visual_safetensors, device='cpu')

            # resize_eva_pos_embed expects model.visual but we ARE the visual tower
            class _W: pass
            _w = _W()
            _w.visual = model
            from .eva_clip.utils import resize_eva_pos_embed
            resize_eva_pos_embed(state_dict, _w)

            model.load_state_dict(state_dict, strict=False)

            # Set image_mean/image_std (normally set by create_model_and_transforms)
            from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
            model.image_mean = OPENAI_DATASET_MEAN
            model.image_std = OPENAI_DATASET_STD
        else:
            logging.info(f'No pre-converted safetensors found, using standard (slow) path')
            model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True, local_dir=clip_dir)
            model = model.visual'''

content = content.replace(old_method, new_method)

with open('$PULID_DIR/pulidflux.py', 'w') as f:
    f.write(content)
print('Patched load_eva_clip method')
"

# ─────────────────────────────────────────────────────────────────────────────
# P2: Patch InsightFace — load only detection + recognition models
# ─────────────────────────────────────────────────────────────────────────────
sed -i "s|model = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=\[provider + 'ExecutionProvider', \])|model = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, allowed_modules=['detection', 'recognition'], providers=[provider + 'ExecutionProvider'])|" "$PULID_DIR/pulidflux.py"

# ─────────────────────────────────────────────────────────────────────────────
# P5: Defer facenet_pytorch import (loaded at module level but may be unused)
# ─────────────────────────────────────────────────────────────────────────────
sed -i 's|^from facenet_pytorch import MTCNN, InceptionResnetV1|# Deferred: from facenet_pytorch import MTCNN, InceptionResnetV1  # moved to lazy import|' "$PULID_DIR/pulidflux.py"

# ─────────────────────────────────────────────────────────────────────────────
# P6: Parallel prefetch — first loader to execute triggers all 3 in threads
# ─────────────────────────────────────────────────────────────────────────────
# Appends code to pulidflux.py that wraps the 3 loader methods.
# When ComfyUI calls any loader, it kicks off all 3 loads in parallel threads,
# then waits only on its own. Total load time = max(individual) not sum.
cat >> "$PULID_DIR/pulidflux.py" << 'PARALLEL_PATCH'

# ── P6: Parallel model prefetch ──────────────────────────────────────────────
import threading as _p6_threading
from concurrent.futures import ThreadPoolExecutor as _p6_TPE

_p6_lock = _p6_threading.Lock()
_p6_futures = {}
_p6_cache = {}
_p6_triggered = False
_p6_pool = None

# Save the already-patched originals
_p6_orig_load_model = PulidFluxModelLoader.load_model
_p6_orig_load_insightface = PulidFluxInsightFaceLoader.load_insightface
_p6_orig_load_eva_clip = PulidFluxEvaClipLoader.load_eva_clip

def _p6_get_pool():
    global _p6_pool
    if _p6_pool is None:
        _p6_pool = _p6_TPE(max_workers=3)
    return _p6_pool

def _p6_trigger_all(pulid_file='pulid_flux_v0.9.1.safetensors', provider='CUDA'):
    """Kick off all 3 loads in parallel. Safe to call multiple times — only runs once."""
    global _p6_triggered
    with _p6_lock:
        if _p6_triggered:
            return
        _p6_triggered = True
    logging.info('P6: Starting parallel prefetch of all 3 PuLID models')
    pool = _p6_get_pool()
    _p6_futures['pulid'] = pool.submit(
        _p6_orig_load_model, PulidFluxModelLoader(), pulid_file)
    _p6_futures['insightface'] = pool.submit(
        _p6_orig_load_insightface, PulidFluxInsightFaceLoader(), provider)
    _p6_futures['eva_clip'] = pool.submit(
        _p6_orig_load_eva_clip, PulidFluxEvaClipLoader())

def _p6_wait(key):
    """Wait for a specific prefetch to complete and cache the result."""
    with _p6_lock:
        if key in _p6_cache:
            logging.info(f'P6: {key} from cache (instant)')
            return _p6_cache[key]
    result = _p6_futures[key].result()
    with _p6_lock:
        _p6_cache[key] = result
    logging.info(f'P6: {key} ready')
    return result

def _p6_parallel_load_model(self, pulid_file):
    _p6_trigger_all(pulid_file=pulid_file)
    return _p6_wait('pulid')

def _p6_parallel_load_insightface(self, provider):
    _p6_trigger_all(provider=provider)
    return _p6_wait('insightface')

def _p6_parallel_load_eva_clip(self):
    _p6_trigger_all()
    return _p6_wait('eva_clip')

PulidFluxModelLoader.load_model = _p6_parallel_load_model
PulidFluxInsightFaceLoader.load_insightface = _p6_parallel_load_insightface
PulidFluxEvaClipLoader.load_eva_clip = _p6_parallel_load_eva_clip
logging.info('P6: Parallel prefetch monkey-patches applied')
# ── End P6 ────────────────────────────────────────────────────────────────────
PARALLEL_PATCH

echo "[patch_pulid] All patches applied successfully (including P6 parallel prefetch)"
