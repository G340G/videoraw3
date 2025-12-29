#!/usr/bin/env python3
"""
generate.py — "Found VHS Job Training" analogue horror PPT generator.

- Always-different theme keyword chosen from random online sources.
- Stable ARG template every run (90s PPT vibe).
- Web scraping: Wikipedia text + Wikimedia Commons images.
- UPGRADE: Scrapes random "creepy" textures (liminal spaces, static, etc).
- UPGRADE: "Ken Burns" effect (pan/zoom) on all images.
- UPGRADE: Full-screen generated data visualization (infographics).
- FIXED: Replaced imageio with direct ffmpeg piping.

Usage:
  python generate.py --config config.yaml --out out.mp4
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import math
import os
import random
import re
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav

UA = "pptex-vhs-generator/4.0 (+github-actions; educational/art project)"

DEFAULTS: Dict[str, Any] = {
    "seed": 1337,
    "theme_source": "random_online",
    "theme_key": "",
    "workdir": ".work",
    "local_images_dir": "assets/images",
    "web": {
        "enable": True,
        "timeout_s": 15,
        "text_paragraphs": 12,
        "image_limit": 20,
        "random_source": "mix",
        "random_attempts": 5,
        "min_keyword_len": 3,
        "max_keyword_len": 48,
        "query_expand": ["{k}", "{k} close up", "{k} structure", "{k} archive"],
        "creepy_terms": ["static noise", "glitch art", "liminal space", "abandoned office", "security camera", "mold", "rust"],
    },
    "story": {
        "slide_count": 14,
        "normal_ratio": 0.45,
        "include_intro_outro": True,
        "include_infographic": True,
        "include_jane_doe": True,
        "include_fatal": True,
        "fatal_probability": 0.15,
        "jane_doe_probability": 0.18,
        "easter_egg_probability": 0.33,
        "entity_mentions_min": 2,
        "entity_mentions_max": 5,
    },
    "render": {
        "width": 640,
        "height": 480,
        "fps": 15,
        "slide_seconds": 3.8,
        "max_popups": 3,
        "popup_seconds": 0.5,
        "micro_popup_probability": 0.07,
        "vhs_strength": 1.25,
        "redaction_strength": 1.15,
        "flashes": 10,
        "censor_probability": 0.35,
        "entity_overlay_probability": 0.10,
    },
    "audio": {
        "sr": 44100,
        "music": True,
        "tts": True,
        "tts_speed": 155,
        "tts_pitch": 32,
        "tts_amp": 170,
        "voices": ["en-us", "en", "en-uk-rp", "en-uk-north", "en-sc"],
        "stinger_count": 22,
        "jingle_strength": 1.0,
    },
    "transmission": {
        "enable": True,
        "error_probability": 0.38,
        "freeze_probability": 0.62,
        "early_end_probability": 0.40,
        "freeze_seconds_min": 0.8,
        "freeze_seconds_max": 2.2,
    },
}

# ----------------------------
# FFmpeg Writer (No imageio)
# ----------------------------

class FFmpegWriter:
    def __init__(self, filename: str, width: int, height: int, fps: int):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None

    def start(self):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            self.filename
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def write(self, frame: np.ndarray):
        if self.proc is None:
            self.start()
        # Ensure correct size
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = np.array(Image.fromarray(frame).resize((self.width, self.height)))
        
        if self.proc.stdin:
            try:
                self.proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass

    def close(self):
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait()
            self.proc = None

# ----------------------------
# Config & Utils
# ----------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def resolve_env_value(v: Any) -> Any:
    if isinstance(v, str):
        m = re.match(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$", v.strip())
        if m:
            return os.environ.get(m.group(1), "")
    return v

def coerce_type(v: Any, default: Any, target_type: type) -> Any:
    v = resolve_env_value(v)
    if v is None: return default
    try:
        if target_type is bool:
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes", "y", "on")
            return bool(v)
        return target_type(v)
    except (ValueError, TypeError):
        return default

def load_config(path: Path) -> Dict[str, Any]:
    raw = {}
    if path and path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"Warning: Could not parse config file {path}: {e}")
    cfg = deep_merge(DEFAULTS, raw)
    # Type coercion for critical fields
    cfg["seed"] = coerce_type(cfg.get("seed"), 1337, int)
    cfg["theme_key"] = str(resolve_env_value(cfg.get("theme_key", "")) or "").strip()
    cfg["workdir"] = str(resolve_env_value(cfg.get("workdir", ".work")) or ".work")
    cfg["local_images_dir"] = str(resolve_env_value(cfg.get("local_images_dir", "assets/images")) or "assets/images")
    return cfg

def _http_get(url: str, timeout: int) -> requests.Response:
    return requests.get(url, headers={"User-Agent": UA}, timeout=timeout)

def _clean_keyword(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    s = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", s).strip()
    return s

def _font_try(names: List[str], size: int) -> ImageFont.ImageFont:
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()

def _clamp(x, a, b):
    return max(a, min(b, x))

# ----------------------------
# Web Scraping
# ----------------------------

def wikipedia_random_title(timeout_s: int) -> Optional[str]:
    url = "https://en.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=1"
    try:
        js = _http_get(url, timeout_s).json()
        it = (js.get("query", {}).get("random") or [{}])[0]
        return _clean_keyword(it.get("title", "")) or None
    except Exception:
        return None

def choose_theme_key(rng: random.Random, cfg: Dict[str, Any]) -> str:
    wcfg = cfg["web"]
    fallback = ["employee handbook", "warning label", "paper clip", "office chair", "door hinge"]
    for _ in range(int(wcfg.get("random_attempts", 3))):
        pick = wikipedia_random_title(int(wcfg.get("timeout_s", 15)))
        if pick and 3 < len(pick) < 50:
            return pick
    return rng.choice(fallback)

def wiki_extract(query: str, max_paragraphs: int, timeout_s: int) -> str:
    if not query: return ""
    try:
        api = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json"}
        d1 = _http_get(api + "?" + requests.compat.urlencode(params), timeout_s).json()
        hits = (d1.get("query", {}).get("search") or [])
        if not hits: return ""
        title = hits[0].get("title", query)
        params2 = {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json"}
        d2 = _http_get(api + "?" + requests.compat.urlencode(params2), timeout_s).json()
        pages = d2.get("query", {}).get("pages") or {}
        page = next(iter(pages.values()), {})
        txt = page.get("extract", "") or ""
        paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        return "\n\n".join(paras[: max(1, int(max_paragraphs))])
    except Exception:
        return ""

def commons_images(query: str, limit: int, timeout_s: int) -> List[str]:
    if not query: return []
    api = "https://commons.wikimedia.org/w/api.php"
    try:
        params = {"action": "query", "list": "search", "srsearch": query, "srnamespace": 6, "format": "json"}
        d = _http_get(api + "?" + requests.compat.urlencode(params), timeout_s).json()
        hits = (d.get("query", {}).get("search") or [])[: max(10, limit * 2)]
        titles = [h.get("title") for h in hits if h.get("title")]
        urls: List[str] = []
        for t in titles:
            if len(urls) >= limit: break
            p2 = {"action": "query", "titles": t, "prop": "imageinfo", "iiprop": "url", "format": "json"}
            d2 = _http_get(api + "?" + requests.compat.urlencode(p2), timeout_s).json()
            pages = d2.get("query", {}).get("pages", {})
            p = next(iter(pages.values()), {})
            ii = (p.get("imageinfo") or [])
            if ii:
                u = ii[0].get("url", "")
                if re.search(r"\.(jpg|jpeg|png|webp)$", u, re.I) and u not in urls:
                    urls.append(u)
        return urls
    except Exception:
        return []

def download_image(url: str, timeout_s: int) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout_s)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content))
        im.load()
        return im.convert("RGB")
    except Exception:
        return None

def extract_related_terms(rng: random.Random, theme_key: str, scraped_text: str, max_terms: int = 8) -> List[str]:
    txt = (scraped_text or "")
    tokens = re.findall(r"\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,}){0,2}\b", txt)
    tokens = [t.strip() for t in tokens if t.strip()]
    seed_terms = ["manual", "safety", "archive", "camera", "badge", "protocol"]
    pool = list(dict.fromkeys(tokens))
    rng.shuffle(pool)
    out = [theme_key]
    for t in pool[: max_terms]:
        if t.lower() not in theme_key.lower():
            out.append(t)
    out.extend(rng.sample(seed_terms, k=min(len(seed_terms), 3)))
    cleaned: List[str] = []
    for x in out:
        x = _clean_keyword(str(x))
        if 3 <= len(x) <= 60 and x not in cleaned:
            cleaned.append(x)
    return cleaned[: max_terms]

def extract_numeric_facts(scraped_text: str, max_items: int = 8) -> List[Tuple[str, float, str]]:
    txt = scraped_text or ""
    facts: List[Tuple[str, float, str]] = []
    # Percentages
    for m in re.finditer(r"\b(\d{1,3}(?:\.\d+)?)\s*%\b", txt):
        v = float(m.group(1))
        if 0 <= v <= 100: facts.append(("RATE", v, "%"))
    # Years
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", txt):
        y = float(m.group(1))
        facts.append(("YEAR", y, ""))
    # Counts
    unit_pat = r"\b(\d{1,6}(?:,\d{3})*(?:\.\d+)?)\s*(people|items|meters|feet|tons|cases|files)\b"
    for m in re.finditer(unit_pat, txt, flags=re.I):
        try:
            v = float(m.group(1).replace(",", ""))
            unit = m.group(2).lower()[:10].upper()
            facts.append((unit, v, ""))
        except: pass
    return facts[:max_items]

# ----------------------------
# Slide Generation
# ----------------------------

@dataclass
class Slide:
    kind: str
    title: str
    body: str
    bg_imgs: List[Image.Image]
    face_imgs: List[Image.Image]
    seconds: float
    zoom_style: str = "static" # static, slow_in, slow_out, pan

def redact_text(rng: random.Random, s: str, strength: float) -> str:
    if not s: return s
    strength = _clamp(strength, 0.0, 3.0)
    prob = 0.12 * strength
    words = s.split()
    out = []
    for w in words:
        if rng.random() < prob and len(w) > 3:
            out.append("[REDACTED]" if rng.random() < 0.6 else "█" * len(w))
        else:
            out.append(w)
    return " ".join(out)

def _snip_sentences(txt: str) -> List[str]:
    txt = re.sub(r"\s+", " ", (txt or "")).strip()
    if not txt: return []
    frags = re.split(r"(?<=[\.\!\?])\s+", txt)
    out = []
    for f in frags:
        f = f.strip()
        if 26 <= len(f) <= 170: out.append(f)
    return out

def build_template_slides(rng: random.Random, theme_key: str, scraped_text: str,
                         topic_imgs: List[Image.Image], creepy_imgs: List[Image.Image], local_imgs: List[Image.Image],
                         cfg: Dict[str, Any], tape_no: int) -> List[Slide]:
    
    story = cfg["story"]
    render = cfg["render"]
    slide_seconds = float(render["slide_seconds"])
    
    # Combine pools
    all_normal = topic_imgs + local_imgs
    all_creepy = creepy_imgs + local_imgs # Fallback to local if scraping fails
    
    # Text lines
    lines = _snip_sentences(scraped_text)
    if not lines:
        lines = [
            "This module is designed to keep your workday stable.",
            "If you notice something that feels incorrect, you are already involved.",
            "Recognition is the failure mode.",
            "Touch an anchor object. Leave. Do not describe what you saw.",
        ]

    # Helper to pick images with fallback
    def pick(pool: List[Image.Image], k: int) -> List[Image.Image]:
        if not pool: return []
        return rng.sample(pool, k=min(k, len(pool)))

    slides: List[Slide] = []

    # 1. Tape Slate
    tape_slate = "\n".join([
        f"TAPE NUMBER: {tape_no:02d}-{rng.randint(100,999)}",
        f"YEAR: {rng.randint(1988, 1997)}   FORMAT: VHS-SP   CH: {rng.randint(1,12):02d}",
        f"SOURCE KEYWORD: {theme_key}",
        f"ARCHIVE HASH: {hashlib.sha1(f'{theme_key}-{tape_no}'.encode()).hexdigest()[:12].upper()}",
        "PLAYBACK NOTE: DO NOT PAUSE / DO NOT REWIND",
    ])
    slides.append(Slide("slate_tape", "TAPE INFO", tape_slate, [], [], 2.8))

    # 2. Intro
    if story.get("include_intro_outro", True):
        tech = [
            f"TOPIC: {theme_key.upper()}",
            "NOTE: DO NOT DUPLICATE",
            "NOTE: DO NOT DESCRIBE ANOMALIES",
        ]
        slides.append(Slide("intro", "TRAINING ARCHIVE PLAYBACK", "\n".join(tech), pick(all_normal, 1), [], 2.4, "slow_in"))

    # 3. Agenda
    agenda = "\n".join([
        "AGENDA",
        "• Wellness module (standard)",
        "• Equipment & etiquette (standard)",
        "• Memory safety bulletin (mandatory)",
        "• Incident response (mandatory)",
        "• Assessment (redacted)",
    ])
    slides.append(Slide("agenda", "TODAY'S TRAINING", 
                        redact_text(rng, agenda, render["redaction_strength"]*0.35),
                        pick(all_normal, 1), [], 2.6, "static"))

    # Content generation
    normal_titles = ["WORKPLACE WELLNESS", "HAPPINESS HABITS", "PRODUCTIVITY TIP", "TEAM CULTURE", "OFFICE ETIQUETTE"]
    normal_bullets = ["Hydrate every hour.", "Keep notes simple.", "Smile (optional).", "Reduce distractions.", "Keep your desk tidy."]
    protocol_titles = ["MEMORY SAFETY BULLETIN", "RECOGNITION HAZARD", "ENTITY AVOIDANCE", "INCIDENT RESPONSE"]
    protocol_lines = [
        "If a face looks unfamiliar: look away.",
        "Do not describe it. Description teaches it.",
        "Touch an anchor object you can name.",
        "If the room repeats, change the subject.",
        "If you see bright eyes in the dark: do not verify.",
    ]

    count = int(story.get("slide_count", 12))
    normal_n = max(3, int(count * 0.45))
    scary_n = max(3, count - normal_n)

    # Generate Content Slides
    content_slides = []
    
    # Normal Slides
    for _ in range(normal_n):
        t = rng.choice(normal_titles)
        bs = rng.sample(normal_bullets, k=rng.randint(3, 4))
        if lines and rng.random() < 0.7: bs.append(rng.choice(lines))
        body = "• " + "\n• ".join(bs)
        body = redact_text(rng, body, render["redaction_strength"]*0.45)
        content_slides.append(Slide("normal", t, body, pick(all_normal, 1), [], slide_seconds, rng.choice(["slow_in", "pan"])))

    # Scary Slides
    for _ in range(scary_n):
        t = rng.choice(protocol_titles)
        ls = rng.sample(protocol_lines, k=rng.randint(3, 4))
        if lines and rng.random() < 0.8: ls.append("Note: " + rng.choice(lines))
        body = "\n".join([f"{i+1}) {l}" for i, l in enumerate(ls)])
        body = redact_text(rng, body, render["redaction_strength"]*1.2)
        # Use creepy images here!
        bg = pick(all_creepy, 1)
        content_slides.append(Slide("protocol", t, body, bg, [], slide_seconds, "slow_in"))

    rng.shuffle(content_slides)
    slides.extend(content_slides)

    # Insert Intermissions
    if story.get("include_jane_doe", True) and rng.random() < float(story.get("jane_doe_probability", 0.18)):
        body = "\n".join([
            "SUBJECT: JANE DOE",
            "IDENTITY: [REDACTED]",
            f"LAST STABLE MEMORY: {rng.randint(0,23):02d}:{rng.randint(0,59):02d}",
            "OBSERVED: BRIGHT EYES IN SHADOW",
            "NEXT INSTRUCTION: DO NOT DESCRIBE.",
        ])
        slides.insert(rng.randint(2, len(slides)), 
                      Slide("intermission", "JANE DOE INTERMISSION", body, pick(all_creepy, 1), pick(all_normal, 1), 3.0, "static"))

    if story.get("include_fatal", True) and rng.random() < float(story.get("fatal_probability", 0.15)):
        body = "FATAL ERROR: TRAINING_PLAYER.EXE\nSYSTEM: MEMORY MAP UNSTABLE\nADVICE: DO NOT RESTART"
        slides.insert(rng.randint(3, len(slides)),
                      Slide("fatal", "TRACKING LOST", body, [], [], 3.0, "static"))

    # Insert Infographic & Forensic
    if story.get("include_infographic", True):
        # 1. Charts
        facts = extract_numeric_facts(scraped_text, max_items=5)
        # Pad facts if missing
        while len(facts) < 4:
            facts.append(("INDEX", rng.randint(10, 90), ""))
        
        chart_body = f"METRICS: {theme_key.upper()}\n"
        for l, v, u in facts:
            chart_body += f"• {l}: {v}{u}\n"
        
        slides.insert(rng.randint(3, len(slides)), 
                      Slide("infographic", "COMPLIANCE DATA", chart_body, pick(all_normal, 1), [], 4.0, "static"))

        # 2. Forensic (Visual Zoom)
        forensic_body = "FORENSIC IMAGE REVIEW\n• Inspect highlights.\n• Compare artifacts.\n• If you recognize the shape: stop."
        slides.insert(rng.randint(3, len(slides)),
                      Slide("forensic", "PHOTO INSPECTION", forensic_body, pick(all_normal, 1), [], 4.5, "static"))

    # Outro
    if story.get("include_intro_outro", True):
        outro = "END OF MODULE\nThank you.\nDo not replay this tape.\nIf you remember the bright eyes, you are already late."
        slides.append(Slide("outro", "END OF TRANSMISSION", redact_text(rng, outro, render["redaction_strength"]*0.6), pick(all_normal, 1), [], 2.5, "slow_out"))

    return slides[:24] # Safety cap

# ----------------------------
# Visual Engine
# ----------------------------

@dataclass
class RenderContext:
    W: int; H: int; FPS: int
    vhs_strength: float
    redaction_strength: float
    censor_prob: float

def cover_resize(im: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = im.size
    s = max(w/iw, h/ih)
    nw, nh = int(iw*s), int(ih*s)
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    x0, y0 = (nw - w)//2, (nh - h)//2
    return im2.crop((x0, y0, x0+w, y0+h))

def apply_ken_burns(im: Image.Image, w: int, h: int, t: float, style: str, rng: random.Random) -> np.ndarray:
    """
    Apply zoom/pan effect based on time t (0.0 to 1.0).
    style: 'slow_in', 'slow_out', 'pan', 'static'
    """
    iw, ih = im.size
    # We need internal high-res to zoom without pixellation if possible, 
    # but for VHS style, pixellation is fine.
    
    # Calculate base scale to fill screen
    base_scale = max(w/iw, h/ih)
    
    # Define Zoom range
    z1, z2 = 1.0, 1.0
    if style == "slow_in": z1, z2 = 1.0, 1.15
    elif style == "slow_out": z1, z2 = 1.15, 1.0
    elif style == "pan": z1 = z2 = 1.10
    
    curr_z = z1 + (z2 - z1) * t
    sc = base_scale * curr_z
    nw, nh = int(iw * sc), int(ih * sc)
    
    # Calculate crop position
    max_x, max_y = max(0, nw - w), max(0, nh - h)
    
    if style == "pan":
        # Pan across center Y
        x = max_x * (0.1 + 0.8 * t) # 10% to 90%
        y = max_y / 2
    else:
        # Center zoom
        x = max_x / 2
        y = max_y / 2
        
    # Resize and Crop
    im_res = im.resize((nw, nh), Image.Resampling.BILINEAR)
    out = im_res.crop((int(x), int(y), int(x)+w, int(y)+h))
    return np.array(out)

def vibrant_bg(rng: random.Random, W: int, H: int) -> np.ndarray:
    c1 = np.array([rng.randint(30,100), rng.randint(30,100), rng.randint(60,150)])
    c2 = np.array([rng.randint(10,50), rng.randint(10,50), rng.randint(20,80)])
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        r = y/H
        arr[y,:] = c1*(1-r) + c2*r
    return arr

def draw_infographic(ctx: RenderContext, rng: random.Random, frame: np.ndarray, text: str) -> np.ndarray:
    """Full screen prominent bar chart overlay."""
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    font = _font_try(["DejaVuSans-Bold.ttf", "Arial.ttf"], 14)
    
    # Parse data
    data = []
    for line in text.split('\n'):
        m = re.search(r"• ([A-Z0-9\s]+): ([\d\.]+)", line)
        if m:
            val = float(m.group(2))
            # Normalize large numbers for visualization
            if val > 100: val = (val % 100) + 20
            data.append((m.group(1).strip()[:8], val))
            
    if not data:
        data = [("ERR", 0), ("NULL", 0)]

    # Draw Chart
    margin = 60
    chart_w = ctx.W - 2*margin
    chart_h = 240
    chart_y = 140
    
    # BG
    d.rectangle((margin-10, chart_y-10, margin+chart_w+10, chart_y+chart_h+10), fill=(0,0,0,200), outline=(0,255,0,255))
    
    count = len(data)
    bar_width = (chart_w // count) - 20
    max_val = max(d[1] for d in data) + 1
    
    for i, (label, val) in enumerate(data):
        h = int((val / max_val) * chart_h)
        x = margin + 10 + i * (bar_width + 20)
        y = chart_y + (chart_h - h)
        
        # Bar
        d.rectangle((x, y, x+bar_width, chart_y+chart_h), fill=(0, 200, 255, 220))
        # Text
        d.text((x, chart_y+chart_h+5), label, fill="white", font=font)
        d.text((x, y-15), str(int(val)), fill="white", font=font)
        
    return np.array(im)

def make_ui_layer(ctx: RenderContext, slide: Slide) -> np.ndarray:
    im = Image.new("RGBA", (ctx.W, ctx.H), (0,0,0,0))
    d = ImageDraw.Draw(im)
    fontT = _font_try(["DejaVuSans-Bold.ttf", "Arial.ttf"], 28)
    fontB = _font_try(["DejaVuSans.ttf", "Arial.ttf"], 18)
    
    # Color scheme
    color = (60, 200, 255, 230)
    if slide.kind == "fatal": color = (255, 50, 50, 230)
    elif slide.kind in ("protocol", "intermission"): color = (255, 200, 50, 230)
    
    # Header
    d.rectangle((0, 40, ctx.W, 90), fill=color)
    d.text((20, 50), slide.title.upper(), font=fontT, fill=(0,0,0,255))
    
    # Body Box
    d.rectangle((20, 110, ctx.W-20, ctx.H-50), fill=(20, 20, 20, 180), outline=color, width=2)
    
    # Text
    y = 130
    for line in slide.body.split('\n'):
        # Wrap
        wrapped = textwrap.wrap(line, width=48)
        for w in wrapped:
            d.text((35, y), w, font=fontB, fill=(240, 240, 240, 255))
            y += 24
        y += 8 # paragraph gap
        
    return np.array(im)

def vhs_effects(ctx: RenderContext, rng: random.Random, frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    
    # 1. Color Bleed
    out[:, :, 0] = np.roll(out[:, :, 0], 3, axis=1)
    
    # 2. Scanlines
    scan = np.ones((ctx.H, 1), dtype=np.float32)
    scan[::2] = 0.75
    out = (out * scan).astype(np.uint8)
    
    # 3. Noise
    if rng.random() < 0.5:
        noise = np.random.randint(-20, 20, out.shape, dtype=np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return out

def timecode_overlay(ctx: RenderContext, frame: np.ndarray, frame_idx: int, ch: int, tape_no: int) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    font = _font_try(["DejaVuSansMono.ttf", "Courier New.ttf"], 18)
    
    secs = frame_idx / ctx.FPS
    ts = time.strftime("%H:%M:%S", time.gmtime(secs))
    ff = int((secs % 1) * 100)
    
    txt = f"PLAY  {ts}:{ff:02d}  TAPE-{tape_no}  CH{ch:02d}"
    d.text((42, ctx.H-42), txt, font=font, fill=(0,0,0)) # shadow
    d.text((40, ctx.H-44), txt, font=font, fill=(255,255,255))
    
    return np.array(im)

def stamp_popup(ctx: RenderContext, rng: random.Random, frame: np.ndarray, popup_im: Image.Image) -> np.ndarray:
    out = frame.copy()
    pw = rng.randint(150, 300)
    ph = int(pw * (popup_im.height / popup_im.width))
    pm = popup_im.resize((pw, ph))
    if rng.random() < 0.5: pm = ImageOps.invert(pm.convert("RGB"))
    
    parr = np.array(pm)
    x = rng.randint(0, ctx.W - pw)
    y = rng.randint(0, ctx.H - ph)
    
    out[y:y+ph, x:x+pw] = parr[:min(ph, out.shape[0]-y), :min(pw, out.shape[1]-x)]
    return out

# ----------------------------
# Render Video Loop
# ----------------------------

def render_video(rng: random.Random, cfg: Dict[str, Any], slides: List[Slide], out_mp4: Path, tape_no: int) -> Tuple[Path, float, int, List[float]]:
    r = cfg["render"]
    W, H, FPS = int(r["width"]), int(r["height"]), int(r["fps"])
    ctx = RenderContext(W, H, FPS, float(r["vhs_strength"]), float(r["redaction_strength"]), float(r["censor_probability"]))
    
    out_silent = out_mp4.with_name(out_mp4.stem + "_silent.mp4")
    writer = FFmpegWriter(str(out_silent), W, H, FPS)
    
    frame_idx = 0
    event_times_s = []
    
    # Collect pool for popups
    all_bgs = [img for s in slides for img in s.bg_imgs]
    
    for slide in slides:
        num_frames = int(slide.seconds * FPS)
        
        # Pre-render UI (mostly static)
        ui_arr = None
        if slide.kind not in ("slate_tape", "intro", "outro"):
            ui_arr = make_ui_layer(ctx, slide)
            
        # Forensic setup
        forensic_roi = None
        if slide.kind == "forensic" and slide.bg_imgs:
            # Pick a target area to zoom INTO
            # target center
            tx = rng.randint(int(W*0.3), int(W*0.7))
            ty = rng.randint(int(H*0.3), int(H*0.7))
            forensic_roi = (tx, ty)

        for i in range(num_frames):
            t = i / num_frames
            
            # 1. Background (Ken Burns)
            if slide.bg_imgs:
                bg_img = slide.bg_imgs[0]
                
                # Special Forensic Zoom Logic
                if slide.kind == "forensic":
                    # Zoom aggressively towards ROI
                    base = np.array(cover_resize(bg_img, W, H).convert("RGB"))
                    # We simulate zoom by cropping smaller and scaling up
                    zoom_fac = 1.0 + (1.5 * t) # 1x to 2.5x
                    cw, ch = int(W/zoom_fac), int(H/zoom_fac)
                    cx, cy = forensic_roi if forensic_roi else (W//2, H//2)
                    cx = min(max(cx, cw//2), W - cw//2)
                    cy = min(max(cy, ch//2), H - ch//2)
                    
                    pil_base = Image.fromarray(base)
                    crop = pil_base.crop((cx - cw//2, cy - ch//2, cx + cw//2, cy + ch//2))
                    base = np.array(crop.resize((W, H), Image.Resampling.BILINEAR))
                else:
                    base = apply_ken_burns(bg_img, W, H, t, slide.zoom_style, rng)
            else:
                base = vibrant_bg(rng, W, H)
            
            # 2. UI Overlay
            if ui_arr is not None:
                base = alpha_over(base, ui_arr)
            elif slide.kind == "slate_tape":
                # Manual drawing for slate
                pil = Image.fromarray(base)
                d = ImageDraw.Draw(pil)
                d.rectangle((0,0,W,H), fill="black")
                d.text((50, 100), slide.body, fill="white", font=_font_try(["Courier New.ttf"], 20))
                base = np.array(pil)
            elif slide.kind == "intro":
                pil = Image.fromarray(base)
                d = ImageDraw.Draw(pil)
                d.text((50, 200), slide.title, fill="white", font=_font_try(["Arial.ttf"], 40))
                d.text((50, 250), slide.body, fill="white", font=_font_try(["Arial.ttf"], 20))
                base = np.array(pil)

            # 3. Infographic special draw
            if slide.kind == "infographic":
                base = draw_infographic(ctx, rng, base, slide.body)
            
            # 4. Popups
            if rng.random() < 0.015 and all_bgs: 
                pop = rng.choice(all_bgs)
                base = stamp_popup(ctx, rng, base, pop)
                event_times_s.append(frame_idx/FPS)

            # 5. VHS
            final = vhs_effects(ctx, rng, base)
            
            # 6. Timecode
            final = timecode_overlay(ctx, final, frame_idx, ch=1, tape_no=tape_no)
            
            writer.write(final)
            frame_idx += 1
            
    writer.close()
    return out_silent, frame_idx / FPS, frame_idx, event_times_s

# ----------------------------
# Audio (Simple)
# ----------------------------

def gen_audio_track(dur_s: float, sr: int, events: List[float]) -> np.ndarray:
    n = int(dur_s * sr)
    # Background drone
    t = np.linspace(0, dur_s, n, False)
    drone = 0.05 * np.sin(2*np.pi*60*t) + 0.03 * np.sin(2*np.pi*120*t)
    noise = np.random.uniform(-0.05, 0.05, n)
    audio = (drone + noise).astype(np.float32)
    
    # Events
    for et in events:
        idx = int(et * sr)
        if idx < n:
            l = int(0.4 * sr)
            end = min(n, idx+l)
            # Static burst
            audio[idx:end] += np.random.uniform(-0.3, 0.3, end-idx)
            
    return audio

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="out.mp4")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found.")
        return

    cfg = load_config(Path(args.config))
    Path(cfg["workdir"]).mkdir(exist_ok=True, parents=True)
    
    seed = int(cfg["seed"])
    if seed == 1337: seed = random.randint(0, 99999999)
    rng = random.Random(seed)
    print(f"SEED: {seed}")

    # 1. Scrape Topic
    theme_key = cfg.get("theme_key")
    if not theme_key:
        theme_key = choose_theme_key(rng, cfg)
    print(f"THEME: {theme_key}")
    
    # 2. Scrape Text
    text = wiki_extract(theme_key, int(cfg["web"]["text_paragraphs"]), 15)
    
    # 3. Scrape Images (Topic + Creepy)
    print("Scraping images...")
    topic_urls = commons_images(theme_key, 12, 15)
    
    # Creepy pool
    creepy_terms = cfg["web"]["creepy_terms"]
    creepy_urls = commons_images(rng.choice(creepy_terms), 10, 15)
    
    topic_imgs = [download_image(u, 15) for u in topic_urls]
    creepy_imgs = [download_image(u, 15) for u in creepy_urls]
    
    # Filter Nones
    topic_imgs = [x for x in topic_imgs if x]
    creepy_imgs = [x for x in creepy_imgs if x]
    
    # 4. Build Slides
    slides = build_slides(rng, theme_key, text, topic_imgs, creepy_imgs, [], cfg, tape_no=rng.randint(10,99))
    
    # 5. Render
    print("Rendering video...")
    out_silent, dur_s, n_frames, events = render_video(rng, cfg, slides, Path(args.out), tape_no=rng.randint(10,99))
    
    # 6. Audio
    print("Generating audio...")
    sr = 44100
    audio_data = gen_audio_track(dur_s, sr, events)
    wav_path = Path(cfg["workdir"]) / "temp.wav"
    write_wav(str(wav_path), sr, (audio_data * 32767).astype(np.int16))
    
    # 7. Mux
    print("Muxing...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(out_silent), "-i", str(wav_path),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(args.out)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"Done! Saved to {args.out}")

if __name__ == "__main__":
    main()
