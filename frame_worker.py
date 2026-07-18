# Factors speaker framing worker (v3)
# POST /frame {"url","aspect":"wide|story|square","grade":"yes|no"}
# Strategy: crop to the real person, then place them in a slot whose height is a
# FIXED multiple of the person's WIDTH (shoulders width ~ consistent across headshots),
# head pinned near the top. This keeps face size + vertical position consistent even
# when crops include different amounts of shoulder. Feed transparent PNGs.
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageFilter, ImageChops
import io, requests

app = Flask(__name__)
ASPECT = {"wide": 0.872, "story": 0.60, "square": 0.78}   # slot width/height
# slot height as a multiple of the person's cropped WIDTH  (bigger = person appears smaller)
HEIGHT_X_WIDTH = {"wide": 1.85, "story": 2.9, "square": 2.1}
# person height must not exceed this fraction of slot height (caps big-in-frame crops)
PERSON_MAXFILL = {"wide": 0.82, "story": 0.72, "square": 0.80}
# gap above the head as a fraction of slot height
TOP_PAD = {"wide": 0.06, "story": 0.05, "square": 0.06}
PURPLE = (150, 60, 210)
ALPHA_THRESH = 40

def person_bbox(img):
    a = img.split()[3].point(lambda v: 255 if v >= ALPHA_THRESH else 0)
    return a.getbbox() or img.split()[3].getbbox()

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

def frame(person, fmt):
    person = person.crop(person_bbox(person))
    pw, ph = person.size
    aspect = ASPECT.get(fmt, 0.872)
    # slot height keyed off person WIDTH -> consistent face scale
    # scale slot from person WIDTH, but cap so tall/wide crops don't balloon
    ch = int(pw * HEIGHT_X_WIDTH.get(fmt, 1.85))
    ch = max(ch, int(ph / PERSON_MAXFILL.get(fmt, 0.9)))   # ensure headroom for tall crops
    if ch < ph + 2:
        ch = ph + 2
    cw = int(ch * aspect)
    if cw < pw: cw = pw                    # never clip horizontally
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    x = (cw - pw) // 2
    y = int(ch * TOP_PAD.get(fmt, 0.06))   # head pinned near the top
    canvas.paste(person, (x, y), person)
    return canvas

@app.get("/")
def health(): return jsonify(ok=True, service="factors-frame", version=4)

@app.post("/frame")
def go():
    d = request.get_json(force=True)
    fmt = d.get("aspect", "wide")
    r = requests.get(d["url"], timeout=30); r.raise_for_status()
    src = Image.open(io.BytesIO(r.content)).convert("RGBA")
    do_grade = str(d.get("grade", "yes")).lower() in ("yes","true","1","on")
    img = frame(grade(src) if do_grade else src, fmt)
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
