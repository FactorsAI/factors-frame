# Factors speaker framing worker (v6) - FACE-NORMALIZED
# POST /frame {"url","aspect":"wide|story|square","side":"left|right|center","grade":"yes|no"}
# Detects the face, scales so face height is a fixed fraction of the plate, and places
# the face at a fixed vertical position. This makes ANY headshot frame identically,
# matching the approved reference. Feed transparent PNGs.
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageFilter, ImageChops
import io, requests, numpy as np, cv2

app = Flask(__name__)

# plate size per format (aspect must match the template slot)
CANVAS = {"wide": (760, 900), "story": (640, 720), "square": (600, 600)}
# face height as fraction of plate height (calibrated to the approved reference)
FACE_FILL = {"wide": 0.36, "story": 0.62, "square": 0.44}
# face CENTER vertical position as fraction of plate height
# square 0.58 (v13): matches where short/width-capped speakers settle after the drop-to-bottom,
# so tall slim speakers (who never trigger the drop) line up at the same face height.
FACE_CY   = {"wide": 0.63, "story": 0.42, "square": 0.58}
# person width must not exceed this fraction of plate width (keeps inner shoulder in frame)
WIDTH_MARGIN = {"wide": 0.98, "story": 0.98, "square": 0.96}
# formats where the person is anchored flush to the plate bottom (no floating gap)
BOTTOM_ANCHOR = set()
# formats where a short-torso crop is extended (shirt stretched) to reach the plate bottom
# square REMOVED (v12): no stretch — face-positioned so the shortest torso reaches the
# bottom on its own; longer torsos bleed below the frame. story keeps its fill for now.
BOTTOM_FILL = {"story"}
# formats where a floating person is slid straight DOWN until the shirt touches the plate
# bottom (never stretched, never lifted). Longer torsos already exceed the bottom -> unchanged.
BOTTOM_FLOOR = {"square"}
# horizontal bleed toward the outer side (fraction of plate width)
SIDE_SHIFT = 0.10
PURPLE = (150, 60, 210)
ALPHA_THRESH = 40
import os
_CASC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
_casc = cv2.CascadeClassifier(_CASC_PATH)

def person_bbox(img):
    a = img.split()[3].point(lambda v: 255 if v >= ALPHA_THRESH else 0)
    return a.getbbox() or img.split()[3].getbbox()

def detect_face(rgba):
    # returns (fx,fy,fw,fh) in image coords, or None
    rgb = np.array(rgba.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _casc.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        faces = _casc.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
    if len(faces) == 0:
        return None
    # largest face
    return max(faces, key=lambda f: f[2] * f[3])

def grade(img):
    img = img.convert("RGBA"); W, H = img.size; a = img.split()[3]
    g = Image.new("L", (W, H), 0); gp = g.load()
    for y in range(H):
        for x in range(W):
            gp[x, y] = max(0, min(255, int(((1 - x / W) * .6 + (y / H) * .6) * 210)))
    g = g.filter(ImageFilter.GaussianBlur(W * .04))
    t = Image.new("RGBA", (W, H), PURPLE + (0,)); t.putalpha(g)
    t = Image.composite(t, Image.new("RGBA", (W, H), (0, 0, 0, 0)), a)
    sc = ImageChops.screen(img.convert("RGB"), t.convert("RGB")).convert("RGBA")
    sc.putalpha(t.split()[3])
    out = Image.alpha_composite(img, sc); out.putalpha(a); return out

def frame(person, fmt, side, zoom=1.0):
    cw, ch = CANVAS.get(fmt, CANVAS["wide"])
    face = detect_face(person)
    if face is not None:
        fx, fy, fw, fh = face
        target_fh = FACE_FILL.get(fmt, 0.26) * ch
        scale = target_fh / fh
    else:
        # fallback: scale person to fill plate height like before
        bb = person_bbox(person); ph = bb[3] - bb[1]
        scale = (0.96 * ch) / ph
        fx = fy = fw = fh = 0
    pw0, ph0 = person.size
    # WIDTH CAP: never let shoulders touch the plate edges (inner shoulder must stay in frame)
    max_w = cw * WIDTH_MARGIN.get(fmt, 0.88)
    if pw0 * scale > max_w:
        scale = max_w / pw0
    # per-speaker size nudge (default 1.0), applied AFTER the width cap so an explicit zoom
    # can deliberately grow a speaker past the cap. Face stays at FACE_CY, so this changes
    # SIZE only, not vertical alignment.
    scale *= zoom
    nw, nh = max(1, int(pw0 * scale)), max(1, int(ph0 * scale))
    person = person.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    if face is not None:
        # place face center at (side-based x, FACE_CY*ch)
        fcx, fcy = (fx + fw / 2) * scale, (fy + fh / 2) * scale
        target_cy = FACE_CY.get(fmt, 0.40) * ch
        if side == "left":
            target_cx = cw * (0.5 - SIDE_SHIFT)
        elif side == "right":
            target_cx = cw * (0.5 + SIDE_SHIFT)
        else:
            target_cx = cw * 0.5
        x = int(target_cx - fcx)
        y = int(target_cy - fcy)
        if fmt in BOTTOM_ANCHOR:
            y = ch - nh          # sit flush on the plate bottom
        elif fmt in BOTTOM_FLOOR:
            # drop-to-bottom: if the shirt floats above the plate bottom, slide the whole
            # image straight down until the opaque bottom touches it. Never lift, never stretch.
            ob = person_bbox(person)            # opaque bbox in resized coords
            if ob is not None and y + ob[3] < ch:
                y = ch - ob[3]
    else:
        x = (cw - nw) // 2
        y = ch - nh
    canvas.paste(person, (x, y), person)
    # BOTTOM FILL: extend the shirt's bottom edge straight down (per-column clamp) — seamless on fabric
    person_bottom = y + nh
    gap = ch - person_bottom
    if fmt in BOTTOM_FILL and gap > 0:
        import numpy as _np
        arr = _np.array(canvas)                       # RGBA of what's placed so far
        # for each column, find the lowest opaque row within the person and repeat it downward
        for cx in range(x, min(x + nw, cw)):
            col_alpha = arr[:person_bottom, cx, 3]
            rows = _np.where(col_alpha > 40)[0]
            if len(rows):
                src = arr[rows[-1], cx, :].copy()      # bottom-most opaque pixel of this column
                arr[person_bottom:ch, cx, :] = src     # clamp/extend to plate bottom
        canvas = Image.fromarray(arr, "RGBA")
    return canvas

@app.get("/")
def health(): return jsonify(ok=True, service="factors-frame", version=14)

@app.post("/frame")
def go():
    d = request.get_json(force=True)
    fmt = d.get("aspect", "wide"); side = d.get("side", "center")
    try: zoom = float(d.get("zoom", 1.0))
    except (TypeError, ValueError): zoom = 1.0
    r = requests.get(d["url"], timeout=30); r.raise_for_status()
    src = Image.open(io.BytesIO(r.content)).convert("RGBA")
    do_grade = str(d.get("grade", "yes")).lower() in ("yes","true","1","on")
    img = frame(grade(src) if do_grade else src, fmt, side, zoom)
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
