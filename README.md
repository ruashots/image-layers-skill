# image-layers

Split any image into transparent RGBA layers — subjects, background plate, even
lighting — using [Qwen-Image-Layered](https://github.com/QwenLM/Qwen-Image-Layered)
running on your own ComfyUI. One script, no custom nodes.

```bash
python3 scripts/split_layers.py photo.png \
  --prompt "a kid juggling a soccer ball on a street, banners overhead" \
  --layers 5 --out layers/
```

You get: numbered PNGs (composite → inpainted background plate → element layers
with alpha), a `manifest.json`, and a checkerboard `sheet.png` to eyeball the cut.

Use it for: background removal, clean plates ("the scene without the subject"),
2.5D parallax assets, text-behind-subject composites, sticker cutouts.

## Setup
1. A running ComfyUI (tested on 0.3.x with native Qwen-Image-Layered support).
2. Download into `ComfyUI/models/`:
   - [`text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors)
   - [`diffusion_models/qwen_image_layered_fp8mixed.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_layered_fp8mixed.safetensors)
   - [`vae/qwen_image_layered_vae.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/resolve/main/split_files/vae/qwen_image_layered_vae.safetensors)
3. `pip install pillow` (optional, for the QC sheet).

~24GB VRAM recommended for fp8mixed at 640px. Python stdlib otherwise.

## Knobs
| flag | default | meaning |
|---|---|---|
| `--prompt` | generic | describe the scene — better descriptions, better splits |
| `--layers` | 4 | how many layers to request (1–16; extras pad empty) |
| `--size` | 640 | largest dimension; 1024 for sharper edges, slower |
| `--steps` / `--cfg` | 20 / 2.5 | sampling |
| `--seed` | fixed | deterministic by default |
| `--out` / `--prefix` | `layers/` | where results land |
| `--comfy-url` | localhost:8188 | your ComfyUI |
| `--no-sheet` | off | skip the QC sheet |

## Agents
Claude Code: drop this folder in `~/.claude/skills/` — `SKILL.md` wires it up.
Other agents: see [`HANDOFF.md`](HANDOFF.md).

## License
MIT. Model weights are Apache 2.0 (Alibaba Qwen team).
