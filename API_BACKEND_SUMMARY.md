# EcoFont FastAPI Backend Summary

## Purpose

This backend exposes the existing `eco_research_hangul` guided font-generation
pipeline as an HTTP API.

The first API version is intentionally a thin wrapper around the current
research code:

```text
HTTP request
  -> API job config
  -> eco_research_hangul guided-infer
  -> manifest parsing
  -> PNG/JSON asset URLs
```

It supports Hangul and Cherokee requests through the existing guided configs.
It also supports uploaded Cherokee TTF batch generation, where one uploaded
font is converted into multiple complete Cherokee candidate TTFs plus previews,
metrics, and a zip artifact.

## Main Files

- `ecofont/api.py`
  - FastAPI application entry point.
  - Keeps the older `/api/fonts/convert` endpoint.
  - Mounts the new `/v1/*` generation API.
- `ecofont/server/api.py`
  - FastAPI router for profiles, generation jobs, results, and assets.
- `ecofont/server/schemas.py`
  - Request and response models.
- `ecofont/server/profiles.py`
  - Maps API profile names to existing `eco_research_hangul/configs/guided_*.yaml`.
- `ecofont/server/jobs.py`
  - In-memory job queue using a single worker thread.
- `ecofont/server/research_runner.py`
  - Creates per-request config files.
  - Runs `eco_research_hangul.guided.run_guided_inference_from_config`.
  - Runs report generation.
  - Parses manifests into API JSON.
- `ecofont/server/font_generation_runner.py`
  - Converts an uploaded Cherokee TTF into candidate complete Cherokee TTFs.
- `ecofont/server/font_generation_jobs.py`
  - In-memory job queue for uploaded TTF batch generation.
- `ecofont/server/script_sets.py`
  - Defines script/codepoint sets such as `cherokee_full`.
- `ecofont/server/ttf_export.py`
  - Converts generated glyph bitmaps into TrueType outlines.

## Run

```bash
cd /home/woojye2020/decs_jupyter_lab/eco-font
.venv/bin/python -m uvicorn ecofont.api:app --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

```text
GET  /health
GET  /v1/health
GET  /v1/profiles
GET  /v1/font-generation/methods
POST /v1/generate
POST /v1/generate-sync
POST /v1/font-generation/jobs
POST /v1/font-generation/jobs-sync
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/result
GET  /v1/font-generation/jobs/{job_id}
GET  /v1/font-generation/jobs/{job_id}/result
GET  /v1/assets/{job_id}/{asset_path}
```

`POST /v1/generate` is the recommended endpoint. It returns immediately with a
`job_id`, then the client polls `/v1/jobs/{job_id}`.

`POST /v1/generate-sync` is useful only for short local tests because it blocks
until generation finishes.

`POST /v1/font-generation/jobs` accepts `multipart/form-data`:

- `font`: uploaded `.ttf` or glyf-based `.otf`
- `spec`: JSON string with `script`, `method`, `candidate_count`,
  `codepoint_set`, `return_format`, and `evaluation`

Current uploaded TTF batch support:

```text
script: cherokee
method: eco_research_guided
codepoint_set: cherokee_full | uploaded_cherokee
candidate_count: 1..20
```

## Example Requests

Hangul:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "가나다한글",
    "script": "hangul",
    "profile": "hangul-jua",
    "target_saving": 0.42,
    "diffusion_candidates": 2,
    "sample_steps": 48
  }'
```

Cherokee:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "ᎠᎡᎢᎣᎤ",
    "script": "cherokee",
    "profile": "cherokee-noto",
    "target_saving": 0.42
  }'
```

Uploaded Cherokee TTF batch:

```bash
curl -X POST http://127.0.0.1:8000/v1/font-generation/jobs \
  -F 'font=@/usr/share/fonts/truetype/noto/NotoSansCherokee-Regular.ttf' \
  -F 'spec={
    "script": "cherokee",
    "method": "eco_research_guided",
    "candidate_count": 20,
    "codepoint_set": "cherokee_full",
    "return_format": "zip",
    "evaluation": {
      "ocr": true,
      "ink": true,
      "ocr_eval_mode": "sample",
      "eval_sample_size": 32,
      "ocr_lang": "chr",
      "ocr_psm": [8, 6, 10]
    }
  }'
```

## Profiles

- `hangul-jua`
- `hangul-gothic`
- `hangul-myeongjo`
- `hangul-barunpen`
- `cherokee-noto`

If `profile` is omitted, the API selects the default profile for the requested
script.

## Result Shape

The result contains:

- `contact_sheet`
  - Source vs selected generated glyph overview.
- `manifests.inference`
  - Raw selected-glyph manifest.
- `manifests.candidates`
  - Candidate-level OCR, ink, style, and objective scores.
- `characters[]`
  - One item per generated character.
  - Includes source image, selected generated image, selected metrics, and saved
    intermediate candidate image URLs.

Uploaded TTF batch results contain:

- `outputs.zip_url`
  - Zip containing candidate TTFs, preview PNGs, and `manifest.json`.
- `coverage`
  - Requested, covered, visible, and missing Cherokee glyph counts.
- `candidates[]`
  - One item per generated TTF candidate.
  - Includes `style_id`, `ttf_url`, `preview_url`, coverage, and average metrics.

Candidate image URLs are served by:

```text
/v1/assets/{job_id}/guided/candidates/...
```

## Runtime Output

Runtime files are intentionally excluded from git:

```text
eco_research_hangul/configs/api_job_*.yaml
eco_research_hangul/outputs/api_jobs/{job_id}/
```

## Current Limits

- Job state is in memory. Restarting the server loses job status, though output
  files remain on disk.
- Only one worker runs by default to avoid concurrent GPU/CPU memory pressure.
- Mixed-script text is rejected. Send Hangul and Cherokee as separate requests.
- Uploaded TTF batch generation currently supports Cherokee only.
- Uploaded font outlines must be TrueType `glyf`; CFF/OTF outlines are rejected.
- Uploaded TTF batch mode uses fast eco style recipes, not full diffusion over
  every glyph.
- This is not yet containerized.
- Authentication and rate limits are not implemented.

## Next Backend Steps

- Add persistent job storage.
- Move long jobs to Celery/RQ if multiple users will call the API.
- Add artifact cleanup and retention policy.
- Add authentication before exposing the server outside localhost.
- Add a frontend client that polls job status and displays contact sheets and
  candidate grids.
