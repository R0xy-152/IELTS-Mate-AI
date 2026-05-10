# IELTS-Mate AI Deployment

This app is a FastAPI backend that also serves the static frontend in `static/`.
It is suitable for a portfolio demo when deployed as a backend service, not as a
pure static website.

## Recommended Demo Shape

Use your personal website as the showcase page, then link to the deployed app:

```text
Personal website -> Live Demo link -> IELTS-Mate AI FastAPI service
```

If your personal website calls the API from its own frontend, set `CORS_ORIGINS`
to your site domain.

## Required Environment Variables

Set these in the deployment platform dashboard:

| Name | Value |
| --- | --- |
| `GEMINI_API_KEY` | Your real Gemini key, stored only in the platform dashboard |
| `GEMINI_MODEL` | `gemini-2.5-flash` |
| `GEMINI_IMAGE_MODEL` | `gemini-2.5-flash-image` |
| `IMAGE_DAILY_LIMIT_PER_IP` | `1` |
| `RATE_LIMIT_SALT` | A long random private string |
| `CORS_ORIGINS` | Your personal website origin, such as `https://your-domain.example` |

Never commit `.env` or a real API key.

## Local Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[test]
cp .env.example .env
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Render

This repo includes `render.yaml`.

1. Push the repo to GitHub.
2. In Render, create a new Blueprint from the repo.
3. Fill `GEMINI_API_KEY`.
4. Fill `CORS_ORIGINS` with your personal website origin if needed.
5. Deploy.

Render free instances may sleep. The first request after sleep can be slow.

## Railway / Fly.io / VPS

Use this start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

For a VPS where `$PORT` is not provided, choose a fixed port and put Nginx or
Caddy in front of it.

## Current Demo Limits

- AI image generation is limited to one successful generation per IP per day by
  default.
- The app stores image quota records as hashed IPs.
- SQLite is acceptable for a small portfolio demo.
- Generated images are stored under `static/generated_images/`; on some free
  platforms, local files can disappear after redeploys or restarts.

For a more durable public product, use Postgres plus object storage such as
Cloudflare R2 or S3.

## Validate Before Publishing

```bash
python -m pytest
git status --short --ignored
# Also run a secret scan before pushing. At minimum, search for real API key
# prefixes and non-empty secret/token/password assignments.
git status --short --branch
```

The secret scan should not show real credentials. It may show documentation
examples only if they do not contain real values.
