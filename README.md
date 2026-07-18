# factors-frame

Framing worker for Factors webinar speaker images.

## What it does
Takes a person PNG (ideally transparent), scales the person to a consistent size,
bottom-aligns them into a fixed slot, and (optionally) applies the locked purple grade.
No background removal — feed transparent PNGs.

## Deploy (Railway)
1. Push this folder to a GitHub repo.
2. Railway → New Project → Deploy from GitHub repo → pick it.
3. Service → Settings → Networking → Generate Domain.
4. Worker URL = <domain>/frame

## Test
curl -X POST https://YOUR-DOMAIN/frame \
  -H "Content-Type: application/json" \
  -d '{"url":"<transparent-png-url>","aspect":"wide","grade":"yes"}' \
  --output test.png

## Request body
- url    : image URL (required)
- aspect : "wide" | "story" | "square"  (default "wide")
- grade  : "yes" | "no"  (default "yes" — set "no" if purple is already baked in)

## n8n
Point the "Cut + grade" node URL at  https://YOUR-DOMAIN/frame
