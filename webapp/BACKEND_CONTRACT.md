# Backend contract — what `app.py` must return

The frontend is tolerant (it accepts several key spellings), but this is the shape it is built for.

## Files to copy into your Flask project

```
templates/index.html      ← replaces your current one
static/style.css          ← replaces your current one
static/app.js             ← replaces your current one
```

Your `app.py` stays as it is, as long as it declares:

```python
app = Flask(__name__, template_folder='templates', static_folder='static')
```

---

## `GET /api/health`

```json
{ "model_loaded": true, "device": "cpu", "classes": ["Organic", "Recyclable"] }
```

Drives the badge in the top-right of the nav. `loaded` or `ok` also accepted instead of `model_loaded`.

---

## `GET /api/stats`

Anything JSON-serialisable. Shown raw in the API console.

```json
{
  "dataset": { "total": 25077, "organic": 13966, "recyclable": 11111 },
  "test": { "accuracy": 0.8925, "f1": 0.8915, "n": 800 }
}
```

---

## `GET /api/research-gaps`

Anything JSON-serialisable. Shown raw in the API console. The five gaps are already written
into the page itself, so this endpoint is for the API demo only.

---

## `POST /api/predict` — the important one

**Request** — `multipart/form-data`, field `image` (a file). The page also sends `image_b64`
(a `data:` URL string) as a fallback when the item came from a paste or the camera, so accept both.

**Response**

```json
{
  "label": "Organic",
  "class_index": 0,
  "confidence": 0.9421,
  "probabilities": { "Organic": 0.9421, "Recyclable": 0.0579 },
  "inference_ms": 18.3,
  "gradcam": "data:image/png;base64,iVBORw0KGgo..."
}
```

- `probabilities` is what drives the two bars. If you omit it, the page falls back to
  `label` + `confidence`.
- `inference_ms` fills the amber timing badge. Omit it and the page uses round-trip time.
- `gradcam` is the heat map, either a full `data:` URL or a bare base64 string — both work.
  Omit it and the Grad-CAM panel shows "Heat map unavailable" instead of breaking.

### Reference implementation

```python
import io, time, base64
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from flask import request, jsonify

def _read_upload():
    """Accept either a file field or a base64 data URL."""
    if 'image' in request.files:
        return Image.open(request.files['image'].stream).convert('RGB')
    b64 = request.form.get('image_b64', '')
    if ',' in b64:
        b64 = b64.split(',', 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')


def _gradcam_png(model, x, class_idx, pil_img):
    """Grad-CAM on the last conv block, returned as a base64 PNG the size of the input."""
    acts, grads = {}, {}
    layer = model.features[-4]          # last Conv2d in block 5
    h1 = layer.register_forward_hook(lambda m, i, o: acts.setdefault('v', o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.setdefault('v', go[0]))

    model.zero_grad()
    out = model(x)
    out[0, class_idx].backward()
    h1.remove(); h2.remove()

    w = grads['v'].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts['v']).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(pil_img.height, pil_img.width),
                        mode='bilinear', align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # jet-ish colour ramp -> RGB
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    rgb = (np.dstack([r, g, b]) * 255).astype('uint8')

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        img = _read_upload()
    except Exception:
        return jsonify(error='Could not read the image.'), 400

    x = transform(img).unsqueeze(0).to(DEVICE)   # your existing 128x128 transform

    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    elapsed = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())
    try:
        cam = _gradcam_png(model, x, idx, img)
    except Exception:
        cam = None                                # page degrades gracefully

    return jsonify(
        label=CLASSES[idx],
        class_index=idx,
        confidence=float(probs[idx]),
        probabilities={CLASSES[0]: float(probs[0]), CLASSES[1]: float(probs[1])},
        inference_ms=round(elapsed, 1),
        gradcam=cam,
    )
```

`model.features[-4]` targets the last `Conv2d` in block 5 of `WasteClassifierCNN`
(the block is `Conv2d, BatchNorm2d, ReLU, MaxPool2d`, so index `-4` from the end of
`features` is that conv). If you swap in MobileNetV2, use `model.features[-1]` instead.

---

## What degrades gracefully

| Missing | What happens |
|---|---|
| `/api/predict` returns non-200 | Toast explains the failure, pane returns to idle |
| `gradcam` absent | Grad-CAM panel shows "Heat map unavailable" |
| `inference_ms` absent | Timing badge uses measured round-trip |
| `/api/health` unreachable | Nav badge reads "server unreachable" in amber |
| No camera permission | Toast explains, upload still works |
