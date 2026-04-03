"""
Build-time script: convert EVA02_CLIP_L_336 .pt (pickle) to visual-only .safetensors.

Saves ~10-15s on cold start by:
1. Switching from slow pickle deserialization to fast safetensors mmap loading
2. Stripping the text tower weights (unused by PuLID) — smaller file, less to load
"""
import sys
import torch
from safetensors.torch import save_file

pt_path = sys.argv[1]  # e.g. /comfyui/models/clip/EVA02_CLIP_L_336_psz14_s6B.pt
out_path = sys.argv[2]  # e.g. /comfyui/models/clip/EVA02_CLIP_L_336_visual.safetensors

print(f"[convert] Loading {pt_path} ...")
checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

# The .pt file may wrap state_dict under 'model', 'module', or 'state_dict' keys
state_dict = checkpoint
for key in ('model', 'module', 'state_dict'):
    if isinstance(state_dict, dict) and key in state_dict:
        state_dict = state_dict[key]
        break

# Extract only visual.* keys and strip the prefix
# Extract only visual.* keys, strip prefix, and clone to break shared memory
# (RoPE freqs_cos/freqs_sin are shared across blocks — safetensors requires unique storage)
visual_sd = {}
for k, v in state_dict.items():
    if k.startswith('visual.'):
        visual_sd[k[7:]] = v.clone()  # strip 'visual.' prefix + clone

print(f"[convert] Extracted {len(visual_sd)} visual keys (discarded {len(state_dict) - len(visual_sd)} text/other keys)")
save_file(visual_sd, out_path)
print(f"[convert] Saved {out_path}")
