---
name: image-layers
description: Decompose any image into transparent RGBA layers + an inpainted background plate using Qwen-Image-Layered via local ComfyUI. Use when asked to split an image into layers, remove a background (take the subject layer), extract a clean plate ("the scene without X"), prepare 2.5D parallax assets for video, or put text/graphics behind a subject. Works on photos and posters/design images.
---

# image-layers — image → RGBA layer stack (local, ComfyUI)

One command splits an image into N transparent layers plus an inpainted
background plate:

```bash
python3 ~/.claude/skills/image-layers/scripts/split_layers.py input.png \
  --prompt "describe the scene: subjects, background, any text" \
  --layers 4 --out layers/
```

Output (in `--out`): numbered PNGs + `manifest.json` + `sheet.png` (checkerboard
QC contact sheet). Typical stack order: **composite reconstruction → inpainted
background plate → element layers (RGBA)**. Read sheet.png to verify the split
before using layers.

## The three verbs it covers
- **split** — the full stack (this is the native operation)
- **remove background** — use the subject's element layer, discard the rest
- **clean plate** — use layer 2 (background with subjects removed and infilled)

## Craft notes (tested 2026-06-11 on RTX 5090, fp8mixed)
- **Describe the scene in `--prompt`** — a real description of subjects/background/
  text measurably improves separation vs the generic default.
- **Layer grouping is semantic and greedy**: at low counts it bundles (eagle+smoke+text
  came out as one layer at `--layers 4`). Ask for more layers to split finer;
  extra capacity pads with near-empty layers (harmless — discard them).
- It can separate **lighting from objects** (a ball's glow/shadow got its own layer).
- Works on photographic scenes too, not just posters; finest detail (hair edges)
  is the weak spot at 640px — raise `--size` (e.g. 1024) + `--steps` for production
  cuts, at ~2-4x the runtime (640px ≈ 40-60s warm; first run loads ~28GB of models).
- Deterministic by default (fixed seed); same input + args = same layers.
- The plate is a *reconstruction* — inspect it; occasionally elements survive in it.

## Requirements
Running ComfyUI (default `http://127.0.0.1:8188`) with:
- `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `diffusion_models/qwen_image_layered_fp8mixed.safetensors` (bf16 variant needs >32GB)
- `vae/qwen_image_layered_vae.safetensors`
(all from `Comfy-Org/Qwen-Image-Layered_ComfyUI` on HuggingFace; encoder from
`Comfy-Org/HunyuanVideo_1.5_repackaged`). Pillow optional (QC sheet).

## Pipeline position
ideogram4 (compose) → **image-layers (split)** → hyperframes (2.5D parallax,
text-between-layers) or recompose → ltx-director. Also standalone for thumbnails,
web assets, and any "cut this out / remove that" ask.
