#!/usr/bin/env python3
"""image-layers: decompose an image into RGBA layers via Qwen-Image-Layered (ComfyUI).

Outputs N transparent PNG layers + an inpainted background plate + a composite,
downloads them locally, writes a manifest, and builds a checkerboard QC sheet.

Requires a running ComfyUI with the Qwen-Image-Layered models installed:
  models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
  models/diffusion_models/qwen_image_layered_fp8mixed.safetensors   (or bf16)
  models/vae/qwen_image_layered_vae.safetensors
(Comfy-Org/Qwen-Image-Layered_ComfyUI on HuggingFace)
"""
import argparse, json, mimetypes, os, sys, time, urllib.request, urllib.error, uuid

DEFAULT_UNET = "qwen_image_layered_fp8mixed.safetensors"
DEFAULT_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
DEFAULT_VAE = "qwen_image_layered_vae.safetensors"
DEFAULT_PROMPT = ("Decompose this image into clean, semantically coherent layers: "
                  "each distinct subject or element on its own transparent layer, "
                  "with a fully reconstructed background plate.")

def fail(msg, code=1):
    print(f"error: {msg}", file=sys.stderr); sys.exit(code)

def http_json(url, payload=None, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or
                                 ({"Content-Type": "application/json"} if data else {}))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def upload_image(comfy, path):
    fn = os.path.basename(path)
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{fn}"\r\n'
            f'Content-Type: {mimetypes.guess_type(fn)[0] or "image/png"}\r\n\r\n').encode()
    body += open(path, "rb").read()
    body += f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(f"{comfy}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req, timeout=120))["name"]

def build_graph(img, prompt, layers, size, steps, cfg, seed, unet, clip, vae, prefix):
    return {
     "1": {"class_type":"LoadImage","inputs":{"image":img,"upload":"image"}},
     "2": {"class_type":"ImageScaleToMaxDimension","inputs":{"image":["1",0],"upscale_method":"lanczos","largest_size":size}},
     "10":{"class_type":"UNETLoader","inputs":{"unet_name":unet,"weight_dtype":"default"}},
     "11":{"class_type":"CLIPLoader","inputs":{"clip_name":clip,"type":"qwen_image","device":"default"}},
     "12":{"class_type":"VAELoader","inputs":{"vae_name":vae}},
     "13":{"class_type":"ModelSamplingAuraFlow","inputs":{"model":["10",0],"shift":1}},
     "20":{"class_type":"CLIPTextEncode","inputs":{"clip":["11",0],"text":prompt}},
     "21":{"class_type":"CLIPTextEncode","inputs":{"clip":["11",0],"text":""}},
     "30":{"class_type":"VAEEncode","inputs":{"pixels":["2",0],"vae":["12",0]}},
     "31":{"class_type":"ReferenceLatent","inputs":{"conditioning":["20",0],"latent":["30",0]}},
     "32":{"class_type":"ReferenceLatent","inputs":{"conditioning":["21",0],"latent":["30",0]}},
     "40":{"class_type":"GetImageSize","inputs":{"image":["2",0]}},
     "41":{"class_type":"EmptyQwenImageLayeredLatentImage","inputs":{"width":["40",0],"height":["40",1],"layers":layers,"batch_size":1}},
     "50":{"class_type":"KSampler","inputs":{"model":["13",0],"positive":["31",0],"negative":["32",0],
           "latent_image":["41",0],"seed":seed,"control_after_generate":"fixed","steps":steps,"cfg":cfg,
           "sampler_name":"euler","scheduler":"simple","denoise":1}},
     "60":{"class_type":"LatentCutToBatch","inputs":{"samples":["50",0],"dim":"t","slice_size":1}},
     "61":{"class_type":"VAEDecode","inputs":{"samples":["60",0],"vae":["12",0]}},
     "70":{"class_type":"SaveImage","inputs":{"images":["61",0],"filename_prefix":prefix}},
    }

def make_sheet(paths, out_path):
    try:
        from PIL import Image
    except ImportError:
        print("note: Pillow not installed — skipping QC sheet (pip install pillow)"); return None
    cells = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        cb = Image.new("RGBA", im.size, (90,90,90,255)); px = cb.load(); S = max(12, im.size[0]//30)
        for y in range(0, im.size[1], S):
            for x in range(0, im.size[0], S):
                if (x//S + y//S) % 2 == 0:
                    for yy in range(y, min(y+S, im.size[1])):
                        for xx in range(x, min(x+S, im.size[0])): px[xx, yy] = (142,142,142,255)
        comp = Image.alpha_composite(cb, im).convert("RGB")
        comp.thumbnail((300, 9999)); cells.append(comp)
    W = sum(c.width for c in cells); H = max(c.height for c in cells)
    sheet = Image.new("RGB", (W, H), (18,18,22)); x = 0
    for c in cells: sheet.paste(c, (x, 0)); x += c.width
    sheet.save(out_path); return out_path

def main():
    ap = argparse.ArgumentParser(description="Split an image into RGBA layers (Qwen-Image-Layered via ComfyUI)")
    ap.add_argument("image", help="input image path")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="describe the scene for better splits (subjects, background, text)")
    ap.add_argument("--layers", type=int, default=4, help="layer count to request (default 4)")
    ap.add_argument("--size", type=int, default=640, help="largest dimension (default 640; higher = slower/sharper)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=231023)
    ap.add_argument("--out", default="layers", help="output directory (default ./layers)")
    ap.add_argument("--prefix", default="layers", help="output filename prefix")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--no-sheet", action="store_true", help="skip the checkerboard QC sheet")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the render")
    a = ap.parse_args()

    if not os.path.isfile(a.image): fail(f"input not found: {a.image}")
    if not 1 <= a.layers <= 16: fail("--layers must be 1..16")
    comfy = a.comfy_url.rstrip("/")
    try:
        http_json(f"{comfy}/system_stats")
    except Exception as e:
        fail(f"ComfyUI not reachable at {comfy} ({e})")

    img = upload_image(comfy, a.image)
    run_prefix = f"{a.prefix}_{uuid.uuid4().hex[:6]}"
    g = build_graph(img, a.prompt, a.layers, a.size, a.steps, a.cfg, a.seed,
                    DEFAULT_UNET, DEFAULT_CLIP, DEFAULT_VAE, run_prefix)
    try:
        r = http_json(f"{comfy}/prompt", {"prompt": g, "client_id": "image-layers"})
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:800]
        if "safetensors" in detail or "not in" in detail:
            fail("ComfyUI rejected the graph — are the Qwen-Image-Layered models installed?\n" + detail)
        fail("submit failed: " + detail)
    pid = r["prompt_id"]
    print(f"submitted {pid} — sampling {a.layers} layers at {a.size}px / {a.steps} steps")

    t0 = time.time()
    while True:
        time.sleep(4)
        if time.time() - t0 > a.timeout: fail("timed out waiting for render")
        h = http_json(f"{comfy}/history/{pid}")
        if pid not in h: continue
        st = h[pid].get("status", {})
        if st.get("status_str") == "error":
            msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            fail("execution error: " + json.dumps(msgs)[:600])
        if st.get("completed"): break
    outs = []
    for nid, o in h[pid].get("outputs", {}).items():
        outs += [im for im in o.get("images", [])]
    if not outs: fail("render completed but produced no images")

    os.makedirs(a.out, exist_ok=True)
    local = []
    for i, im in enumerate(outs, 1):
        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder",""), "type": im.get("type","output")})
        dst = os.path.join(a.out, f"{a.prefix}_{i:02d}.png")
        with urllib.request.urlopen(f"{comfy}/view?{q}", timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
        local.append(dst)
    roles = {1: "composite (reconstruction)", 2: "background plate (inpainted)"}
    manifest = {"input": os.path.abspath(a.image), "prompt": a.prompt, "layers_requested": a.layers,
                "size": a.size, "steps": a.steps, "cfg": a.cfg, "seed": a.seed,
                "outputs": [{"file": p, "role": roles.get(i+1, f"layer {i-1} (RGBA)")} for i, p in enumerate(local)]}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"{len(local)} images -> {a.out}/ (typical order: composite, background plate, then element layers)")
    if not a.no_sheet:
        s = make_sheet(local, os.path.join(a.out, "sheet.png"))
        if s: print(f"QC sheet -> {s}")

if __name__ == "__main__":
    import urllib.parse
    main()
