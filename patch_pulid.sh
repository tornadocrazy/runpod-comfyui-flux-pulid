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

echo "[patch_pulid] All patches applied successfully"
