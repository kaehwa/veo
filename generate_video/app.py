# app.py
import os
import re
import time
import json
import random
import hashlib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv

# Google GenAI SDK (Gemini / Veo)
from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Minimal settings (only what's necessary)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_ASPECT = "16:9"         # Veo 3 (Preview) is most stable with 16:9
ALLOWED_ASPECTS = {DEFAULT_ASPECT}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-pro")
VEO_MODEL = os.getenv("VEO_MODEL", "veo-3.0-generate-preview")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# Consistency controls (env-tunable)
PROMPT_TEMP_FAST = float(os.getenv("PROMPT_TEMP_FAST", "0.95"))
PROMPT_TEMP_CONSISTENT = float(os.getenv("PROMPT_TEMP_CONSISTENT", "0.78"))
PROMPT_TOP_P = float(os.getenv("PROMPT_TOP_P", "0.95"))
PROMPT_CANDIDATES = int(os.getenv("PROMPT_CANDIDATES", "2"))  # 2~3 추천
PROMPT_MIN_CHARS = int(os.getenv("PROMPT_MIN_CHARS", "120"))
PROMPT_MAX_CHARS = int(os.getenv("PROMPT_MAX_CHARS", "420"))

app = FastAPI(title="Veo3 One-Click (Free Mode · Consistent Quality)")

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
class PromptResponse(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    aspect_ratio: Optional[str] = DEFAULT_ASPECT
    person_generation: Optional[str] = None  # don't default to allow_all (unsupported)

class GenerateResponse(BaseModel):
    operation_name: str
    saved_path: Optional[str] = None

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def get_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Missing GEMINI_API_KEY")
    return genai.Client(api_key=GEMINI_API_KEY)

def _poll_op_until_done(client: genai.Client, op, poll_seconds: int = 8, timeout_s: int = 360):
    """
    Poll Veo operation until it completes. Returns (generated_video, operation_name).
    Tries to be tolerant of either passing the operation object or its name.
    """
    def _get(o):
        try:
            # newer SDKs accept object or name directly
            return client.operations.get(o)
        except TypeError:
            name = getattr(o, "name", None) or (o if isinstance(o, str) else None)
            if not name:
                raise HTTPException(502, "Operation has no name.")
            return client.operations.get(name)

    start = time.time()
    cur = op
    while True:
        cur = _get(cur)
        if getattr(cur, "done", False):
            break
        if time.time() - start > timeout_s:
            raise HTTPException(504, "Video generation timed out.")
        time.sleep(int(poll_seconds))

    if getattr(cur, "error", None):
        raise HTTPException(400, detail=str(cur.error))

    if getattr(cur, "response", None) and getattr(cur.response, "generated_videos", None):
        vids = cur.response.generated_videos
        if not vids:
            raise HTTPException(502, "Empty generated_videos.")
        return vids[0], cur.name

    raise HTTPException(502, "Veo finished without a video or explicit error.")

# ──────────────────────────────────────────────────────────────────────────────
# Free-mode system prompt (LLM has full creative freedom)
# + Strong instruction for dynamic camera movement
# ──────────────────────────────────────────────────────────────────────────────
FREE_PROMPTER_SYS = """
You are writing a single cinematic prompt for Google Veo 3.
Look at the user's optional text and/or attached photo and compose the best possible prompt in English.
Write 1–4 sentences, present tense, vivid and specific. Use your creativity fully.
ALWAYS include energetic, dynamic camera movement cues (e.g., fast dolly-in, whip pan, orbit, crane up, tracking, tilt, push-in/pullback).
Return ONLY the prompt as plain text (no labels, no JSON, no quotes).
"""

# If LLM forgot camera motion, inject a dynamic camera line.
CAMERA_KEYWORDS = (
    "dolly", "track", "tracking", "pan", "tilt", "orbit", "push-in", "push in",
    "pullback", "crane", "gimbal", "handheld", "steadicam", "whip pan", "whip-pan",
    "zoom", "rack focus", "parallax"
)
LIGHTING_WORDS = (
    "golden", "backlit", "neon", "noir", "overcast", "sunlit", "silhouette",
    "flare", "glow", "moody", "shadow", "rim light"
)
AUDIO_WORDS = (
    "ambience", "hush", "wind", "waves", "crowd", "music", "footsteps",
    "rustle", "buzz", "echo"
)
DYNAMIC_VERBS = (
    "surges", "whips", "pivots", "orbits", "cranes", "sweeps", "dives",
    "races", "glides", "rushes", "tilts", "tracks", "pushes", "pulls"
)

def _ensure_dynamic_camera(prompt_text: str) -> str:
    t = prompt_text.lower()
    if any(k in t for k in CAMERA_KEYWORDS):
        return prompt_text
    addon = " The camera executes a fast dolly-in, then a whip-pan into a wide orbit before craning up for an energetic reveal."
    if not re.search(r"[.!?]$", prompt_text.strip()):
        prompt_text = prompt_text.strip() + "."
    return prompt_text + addon

def _score_prompt(p: str) -> float:
    """Heuristic score for cinematic-ness & structure."""
    t = p.lower()
    score = 0.0
    # camera cues (up to +6)
    score += min(6, sum(2 for k in CAMERA_KEYWORDS if k in t))
    # lighting cues (+0..3)
    score += min(3, sum(1 for k in LIGHTING_WORDS if k in t))
    # audio cues (+0..2)
    score += min(2, sum(1 for k in AUDIO_WORDS if k in t))
    # dynamic verbs (+0..4)
    score += min(4, sum(1 for k in DYNAMIC_VERBS if k in t))
    # sentence count: 1~4 ideal (+2), else 0
    sents = max(1, p.count(".") + p.count("!") + p.count("?"))
    if 1 <= sents <= 4:
        score += 2
    # length: ideal window (+2), soft penalty otherwise
    n = len(p.strip())
    if PROMPT_MIN_CHARS <= n <= PROMPT_MAX_CHARS:
        score += 2
    else:
        score -= abs(n - max(min(n, PROMPT_MAX_CHARS), PROMPT_MIN_CHARS)) / 100.0
    return score

def _pick_best(prompts: List[str]) -> str:
    prompts = [s.strip() for s in prompts if s and s.strip()]
    if not prompts:
        raise HTTPException(500, "No prompt candidates.")
    scored = [(p, _score_prompt(p)) for p in prompts]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]

def _generate_prompt(client: genai.Client, parts: List[Any], temperature: float, top_p: float) -> str:
    resp = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=parts if parts else [genai_types.Part(text="")],
        config=genai_types.GenerateContentConfig(
            system_instruction=FREE_PROMPTER_SYS,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=500,
            response_mime_type="text/plain",
        ),
    )
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise HTTPException(500, "Empty prompt from LLM.")
    return text

def _generate_prompt_candidates(client: genai.Client, parts: List[Any], n: int, temperature: float, top_p: float) -> List[str]:
    n = max(1, min(5, n))
    cands: List[str] = []
    for _ in range(n):
        cands.append(_generate_prompt(client, parts, temperature, top_p))
    return cands

def _hash_inputs(user_text: Optional[str], img_bytes: Optional[bytes]) -> str:
    h = hashlib.sha256()
    h.update((user_text or "").encode("utf-8"))
    if img_bytes:
        h.update(img_bytes[:262144])
    return h.hexdigest()

# Simple in-proc cache for prompts
PROMPT_CACHE: Dict[str, str] = {}

# ──────────────────────────────────────────────────────────────────────────────
# Health / Root
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "see": "/docs", "text_model": GEMINI_TEXT_MODEL, "veo_model": VEO_MODEL}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "text_model": GEMINI_TEXT_MODEL,
        "veo_model": VEO_MODEL,
        "allowed_aspects": sorted(ALLOWED_ASPECTS),
        "prompt_temp_fast": PROMPT_TEMP_FAST,
        "prompt_temp_consistent": PROMPT_TEMP_CONSISTENT,
        "prompt_candidates": PROMPT_CANDIDATES,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 1) Auto-prompt: single or best-of-N prompt (no rules, no negatives)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/auto-prompt", response_model=PromptResponse)
async def auto_prompt(
    user_text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    aspect_ratio: Optional[str] = Form(DEFAULT_ASPECT),
    quality: Optional[str] = Form("fast"),   # "fast" | "consistent"
):
    client = get_client()

    parts: List[Any] = []
    img_bytes = None
    if user_text:
        parts.append(genai_types.Part(text=user_text.strip()))
    if image:
        img_bytes = await image.read()
        mime = image.content_type or "image/png"
        parts.append(genai_types.Part(inline_data=genai_types.Blob(mime_type=mime, data=img_bytes)))

    # cache
    cache_key = f"{quality}:{_hash_inputs(user_text, img_bytes)}"
    if cache_key in PROMPT_CACHE:
        cached = PROMPT_CACHE[cache_key]
        return {"prompt": cached, "negative_prompt": None, "aspect_ratio": aspect_ratio or DEFAULT_ASPECT, "person_generation": None}

    # fast: 1-shot (high creativity) | consistent: best-of-N (lower temp)
    if (quality or "fast").lower() == "consistent":
        cands = _generate_prompt_candidates(client, parts, PROMPT_CANDIDATES, PROMPT_TEMP_CONSISTENT, PROMPT_TOP_P)
        best = _ensure_dynamic_camera(_pick_best(cands))
    else:
        best = _ensure_dynamic_camera(_generate_prompt(client, parts, PROMPT_TEMP_FAST, PROMPT_TOP_P))

    PROMPT_CACHE[cache_key] = best
    return {
        "prompt": best,
        "negative_prompt": None,
        "aspect_ratio": aspect_ratio or DEFAULT_ASPECT,
        "person_generation": None,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 2) One-click: creative prompt → Veo call → MP4 saved (supports consistent mode)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/oneclick", response_model=GenerateResponse)
async def oneclick(
    user_text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    outfile: Optional[str] = Form("veo_output.mp4"),
    aspect_ratio: Optional[str] = Form(DEFAULT_ASPECT),
    poll_seconds: Optional[int] = Form(8),
    quality: Optional[str] = Form("fast"),   # "fast" | "consistent"
    reuse_cached: Optional[bool] = Form(True),
):
    if not user_text and not image:
        raise HTTPException(400, "Provide at least one of: user_text or image.")
    if aspect_ratio not in ALLOWED_ASPECTS:
        raise HTTPException(400, f"Unsupported aspect_ratio '{aspect_ratio}'. Allowed: {sorted(ALLOWED_ASPECTS)}")

    client = get_client()

    # (1) Prompt build (fast vs consistent)
    parts: List[Any] = []
    img_bytes = None
    mime = None
    if user_text:
        parts.append(genai_types.Part(text=user_text.strip()))
    if image:
        img_bytes = await image.read()
        mime = image.content_type or "image/png"
        parts.append(genai_types.Part(inline_data=genai_types.Blob(mime_type=mime, data=img_bytes)))

    cache_key = f"{quality}:{_hash_inputs(user_text, img_bytes)}"
    if reuse_cached and cache_key in PROMPT_CACHE:
        prompt_text = PROMPT_CACHE[cache_key]
    else:
        if (quality or "fast").lower() == "consistent":
            cands = _generate_prompt_candidates(client, parts, PROMPT_CANDIDATES, PROMPT_TEMP_CONSISTENT, PROMPT_TOP_P)
            prompt_text = _ensure_dynamic_camera(_pick_best(cands))
        else:
            prompt_text = _ensure_dynamic_camera(_generate_prompt(client, parts, PROMPT_TEMP_FAST, PROMPT_TOP_P))
        PROMPT_CACHE[cache_key] = prompt_text

    # (2) Veo call (no negative prompt; no extra rules)
    image_obj = None
    if img_bytes:
        image_obj = genai_types.Image(image_bytes=img_bytes, mime_type=mime)

    # I2V only allow_adult; text→video omit person_generation
    if image_obj is not None:
        cfg = genai_types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            negative_prompt=None,
            person_generation="allow_adult",
        )
    else:
        cfg = genai_types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            negative_prompt=None
        )

    try:
        op = client.models.generate_videos(
            model=VEO_MODEL,
            prompt=prompt_text,
            image=image_obj,  # I2V if image provided; text→video otherwise
            config=cfg,
        )
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vid, op_name = _poll_op_until_done(client, op, poll_seconds=int(poll_seconds), timeout_s=360)
     # === 저장 경로: ./outputs/파일명 ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.basename(outfile or "veo_output.mp4")
    out_path = os.path.join(OUTPUT_DIR, filename)

    client.files.download(file=vid.video)
    vid.video.save(out_path)

    return {"operation_name": op_name, "saved_path": out_path}

# ──────────────────────────────────────────────────────────────────────────────
# Optional local run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("generate_video.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=True)
