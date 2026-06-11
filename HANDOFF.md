# image-layers — agent handoff

You are an agent with shell access and image-viewing capability. This tool
decomposes one image into transparent RGBA layers using a local ComfyUI server
running Qwen-Image-Layered.

## Contract
1. Run:
   `python3 scripts/split_layers.py <image> --prompt "<scene description>" --layers <N> --out <dir>`
2. Read `<dir>/sheet.png` (checkerboard composite of every output) and judge the
   split: layer 1 = composite, layer 2 = inpainted background plate, rest =
   element layers. Empty padding layers are normal.
3. Use `<dir>/manifest.json` for file-to-role mapping in downstream steps.

## Rules
- Always pass a real scene description in `--prompt`; it improves separation.
- Need finer separation? Re-run with more `--layers`. Need sharper edges?
  `--size 1024 --steps 30` (slower).
- Deterministic: identical args reproduce identical layers (fixed `--seed`).
- If submission fails mentioning models/safetensors: the Qwen-Image-Layered
  models are not installed in ComfyUI — surface that to the operator; do not retry.
- The background plate is generated content — verify it before presenting it
  as "the scene without X".

## Failure modes you must handle
- ComfyUI unreachable → report, don't loop.
- Render timeout (default 900s) → report queue state, don't resubmit blindly.
- Subject split across layers → re-run with adjusted prompt naming the subject explicitly.
