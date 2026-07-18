# Factors speaker framing worker
# POST /frame  {"url":"<transparent-or-raw png>","aspect":"wide|story|square","grade":"yes|no"}
# Returns a PNG: person scaled to a consistent size, bottom-aligned into a fixed slot,
# optional locked purple grade. No background removal (feed transparent PNGs).
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageFilter, ImageChops
import io, requests

app = Flask(__name__)

ASPECT = {"wide": 0.872, "story": 0.60, "square": 0.78}   # person-frame width/height
PURPLE = (150, 60, 210)

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
    out = Image.alpha_composite(img, sc); out.putalpha(a)
    return out

def frame(person, aspect):
    bb = person.split()[3].getbbox()
    if bb: person = person.crop(bb)
    pw, ph = person.size
    cw = pw; ch = int(cw / aspect)
    if ch < ph:
        ch = ph; cw = int(ch * aspect)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    canvas.paste(person, ((cw - pw) // 2, ch - ph), person)
    return canvas

@app.get("/")
def health():
    return jsonify(ok=True, service="factors-frame")

@app.post("/frame")
def go():
    d = request.get_json(force=True)
    r = requests.get(d["url"], timeout=30); r.raise_for_status()
    src = Image.open(io.BytesIO(r.content)).convert("RGBA")
    aspect = ASPECT.get(d.get("aspect", "wide"), 0.872)
    do_grade = str(d.get("grade", "yes")).lower() in ("yes", "true", "1", "on")
    img = frame(grade(src) if do_grade else src, aspect)
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
