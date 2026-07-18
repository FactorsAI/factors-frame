# Factors speaker framing worker
# POST /frame  {"url":"<transparent png>","aspect":"wide|story|square","grade":"yes|no"}
# Normalizes each speaker by HEAD HEIGHT so tight and wide headshots come out the
# same face size, then bottom-aligns into a fixed slot. Feed transparent PNGs.
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageFilter, ImageChops
import io, requests

app = Flask(__name__)

# person-frame width/height per format
ASPECT = {"wide": 0.872, "story": 0.60, "square": 0.78}
# fraction of the slot HEIGHT the person should occupy (controls zoom/consistency)
PERSON_FILL = {"wide": 0.78, "story": 0.62, "square": 0.74}
PURPLE = (150, 60, 210)
ALPHA_THRESH = 40   # ignore near-transparent stray pixels when measuring the person

def person_bbox(img):
    a = img.split()[3]
    mask = a.point(lambda v: 255 if v >= ALPHA_THRESH else 0)
    return mask.getbbox() or a.getbbox()

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

def frame(person, aspect, fill):
    # tight-crop to the real person (thresholded so stray pixels don't inflate it)
    person = person.crop(person_bbox(person))
    pw, ph = person.size
    # slot sized so the person is `fill` fraction of slot height -> consistent face size
    ch = int(ph / fill)
    cw = int(ch * aspect)
    if cw < pw:                      # never crop the person horizontally
        cw = pw; ch = int(cw / aspect)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    x = (cw - pw) // 2
    y = ch - ph                      # bottom-aligned
    canvas.paste(person, (x, y), person)
    return canvas

@app.get("/")
def health():
    return jsonify(ok=True, service="factors-frame", version=2)

@app.post("/frame")
def go():
    d = request.get_json(force=True)
    fmt = d.get("aspect", "wide")
    aspect = ASPECT.get(fmt, 0.872)
    fill = PERSON_FILL.get(fmt, 0.92)
    r = requests.get(d["url"], timeout=30); r.raise_for_status()
    src = Image.open(io.BytesIO(r.content)).convert("RGBA")
    do_grade = str(d.get("grade", "yes")).lower() in ("yes", "true", "1", "on")
    img = frame(grade(src) if do_grade else src, aspect, fill)
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
