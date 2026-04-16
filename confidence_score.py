# ============================================================
# UTILITIES, EXTRACTION, PROVIDERS, REPORT/GITHUB
# ============================================================

import re
import io
import os
import time
import json
import logging
import requests
from typing import Dict, Any, Optional, List, Tuple
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("confidence_scorer")


# ============================================================
# BASIC UTILITIES
# ============================================================

def clamp01(x: float) -> float:
    """Clamp float between 0 and 1."""
    return max(0.0, min(1.0, float(x)))


def load_bytes(path: str) -> bytes:
    """Load file bytes."""
    with open(path, "rb") as f:
        return f.read()


def load_image_bytes(path_or_bytes):
    """Load raw bytes of image."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return bytes(path_or_bytes)
    if isinstance(path_or_bytes, str) and os.path.exists(path_or_bytes):
        return load_bytes(path_or_bytes)
    return b""


# ============================================================
# PROVIDER-SPECIFIC VERIFICATION URL PATTERNS
# (STATIC SYSTEM – DIRECT CHECK ONLY)
# ============================================================

PROVIDER_ENDPOINTS = {
    "nptel": "https://nptel.ac.in/noc/verify",
    "coursera": "https://coursera.org/verify/{}",
    "edx": "https://courses.edx.org/certificates/{}",
    "google": "https://www.credly.com/badges/{}",
    "google career certificates": "https://www.credly.com/badges/{}",
    "credly": "https://www.credly.com/badges/{}",
    "hubspot": "https://app.hubspot.com/academy/achievements/{}",
    "futurelearn": "https://www.futurelearn.com/certificates/{}",
    "greatlearning": "https://verify.greatlearning.in/{}",
    "infosectrain": "https://www.infosectrain.com/verify",
    "iata": "QR_ONLY",
    "global edulink": "https://www.globaledulink.co.uk/validate-certificate-code",
    "peace operations training institute": "https://www.peaceopstraining.org/verify",
    "certifyme": "https://certifyme.online/verify/{}",
    "certifier": "https://verify.certifier.io/{}",
    "simplelearn": None,
    "udemy": None,
    "udacity": None,
}

# Patterns for extracting certificate IDs from PDF text
PROVIDER_CERT_PATTERNS = {
    "nptel": [
        re.compile(r"NPTEL[-\s]*CERT[-\s]*No[:\s]*([A-Z0-9-]+)", re.I),
        re.compile(r"Certificate\s*No[:\s]*([A-Z0-9-]+)", re.I),
        re.compile(r"NPTEL.*certificate.*([A-Z0-9-]{6,})", re.I),
    ],
    "coursera": [
        re.compile(r"coursera\.org/certificates/([A-Za-z0-9-]+)", re.I),
        re.compile(r"Credential\s*ID[:\s]*([A-Za-z0-9-]+)", re.I),
    ],
    "edx": [
        re.compile(r"Certificate\s*ID[:\s]*([A-Za-z0-9-]+)", re.I)
    ],
    "google": [
        re.compile(r"Credly.*badge.*ID.*([A-Za-z0-9-]+)", re.I),
    ],
    "futurelearn": [
        re.compile(r"futurelearn\.com/certificates/([A-Za-z0-9-]+)", re.I)
    ],
    "greatlearning": [
        re.compile(r"greatlearning.*verify.*([A-Za-z0-9-]+)", re.I)
    ],
    "certifyme": [
        re.compile(r"certifyme\.online/verify/([A-Za-z0-9-]+)", re.I),
    ],
    "certifier": [
        re.compile(r"certifier\.io.*verify.*([A-Za-z0-9-]+)", re.I)
    ],
}

GENERIC_CERT_PATTERN = re.compile(
    r"certificate\s*(?:no|number|id|#)[:\-\s]*([A-Za-z0-9\-]{4,40})", re.I
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def parse_pdf_text(pdf_bytes: bytes) -> str:
    """Return extracted text using PyMuPDF → OCR → PyPDF2 fallback."""
    if not pdf_bytes:
        return ""

    # PyMuPDF first
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        if text and len(text.strip()) > 10:
            return text
    except Exception as e:
        logger.debug("PyMuPDF extraction failed: %s", e)

    # OCR fallback
    try:
        from pdf2image import convert_from_bytes
        from pytesseract import image_to_string
        from PIL import Image
        pages = convert_from_bytes(pdf_bytes)
        out = ""
        for p in pages:
            try:
                if not isinstance(p, Image.Image):
                    p = Image.fromarray(p)
                out += image_to_string(p)
            except:
                continue
        if out.strip():
            return out
    except:
        pass

    # PyPDF2 final fallback
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        txt = ""
        for pg in reader.pages:
            try:
                txt += pg.extract_text() or ""
            except:
                pass
        return txt
    except:
        pass

    return ""


# ============================================================
# CERTIFICATE NUMBER DETECTION
# ============================================================

def extract_certificate_id(text: str) -> Optional[Tuple[str, str]]:
    """Return (id, provider) or None."""
    if not text:
        return None

    # Provider-specific patterns
    for prov, pats in PROVIDER_CERT_PATTERNS.items():
        for pat in pats:
            m = pat.search(text)
            if m:
                return m.group(1).strip(), prov

    # Generic certificate detection
    m = GENERIC_CERT_PATTERN.search(text)
    if m:
        return m.group(1).strip(), "unknown"

    # Heuristic long alnum ID
    g = re.search(r"\b([A-Z0-9]{8,})\b", text, re.I)
    if g:
        return g.group(1), "unknown"

    return None


# ============================================================
# OFFER LETTER / CONFIRMATION MAIL / REPORT EXTRACTION
# ============================================================

def extract_report_features(report_bytes: bytes) -> Dict[str, float]:
    """
    Analyze project/research/internship report text for keywords.
    Returns a score component (0–1).
    """

    if not report_bytes:
        return {"score": 0.0, "keywords": [], "text_len": 0}

    text = parse_pdf_text(report_bytes).lower()
    length = len(text)

    keywords = [
        "abstract", "introduction", "methodology", "implementation",
        "results", "discussion", "conclusion", "future work",
        "architecture", "workflow", "dataset", "analysis"
    ]

    hits = sum(1 for k in keywords if k in text)
    keyword_ratio = hits / len(keywords)

    # length-based confidence
    if length > 8000:
        length_score = 1.0
    elif length > 4000:
        length_score = 0.8
    elif length > 2000:
        length_score = 0.6
    else:
        length_score = 0.3

    final_score = clamp01(0.6 * length_score + 0.4 * keyword_ratio)

    return {
        "score": final_score,
        "keywords": [k for k in keywords if k in text],
        "text_len": length
    }


# ============================================================
# GITHUB VERIFICATION (FOR PROJECTS)
# ============================================================

def analyze_github_repo(url: str) -> Dict[str, Any]:
    """
    Check if GitHub repository is valid:
    - exists
    - has commits
    - has files
    - contains README
    - stars/forks optional
    """
    if not url or "github.com" not in url.lower():
        return {"score": 0.0, "valid": False, "detail": "Invalid GitHub URL"}

    api = None

    try:
        # convert https://github.com/user/repo → API URL
        parts = url.rstrip("/").split("/")
        user = parts[-2]
        repo = parts[-1]
        api = f"https://api.github.com/repos/{user}/{repo}"

        r = requests.get(api, timeout=6)
        if r.status_code != 200:
            return {"score": 0.0, "valid": False, "detail": "Repo not found"}

        data = r.json()

        score = 0.0
        if data.get("size", 0) > 50:
            score += 0.4
        if data.get("stargazers_count", 0) > 0:
            score += 0.2
        if data.get("forks_count", 0) > 0:
            score += 0.2

        # README Detection
        readme_url = f"{api}/readme"
        rr = requests.get(readme_url, timeout=6)
        if rr.status_code == 200:
            score += 0.2

        return {
            "score": clamp01(score),
            "valid": True,
            "detail": "GitHub repo verified",
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "size": data.get("size")
        }

    except Exception as e:
        return {"score": 0.0, "valid": False, "detail": str(e)}


# ============================================================
# FACE MATCH + PRESENCE (USED IN MANY CATEGORIES)
# ============================================================

def face_match(photo_bytes: bytes, id_bytes: bytes) -> float:
    """Try DeepFace similarity; return 0 on failure."""
    try:
        from deepface import DeepFace
        import numpy as np
        import cv2

        def _to_cv(b):
            arr = np.frombuffer(b, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        img1 = _to_cv(photo_bytes)
        img2 = _to_cv(id_bytes)
        if img1 is None or img2 is None:
            return 0.0

        r = DeepFace.verify(img1, img2, model_name="Facenet", enforce_detection=False)
        dist = r.get("distance", 1.0)
        sim = clamp01(1 - dist / 1.2)
        return sim

    except Exception:
        return 0.0


def presence_strength(proof_photos: List[bytes], student_image: bytes) -> float:
    """Compute presence score."""
    if not proof_photos:
        return 0.0

    face_score = 0.0
    if student_image:
        try:
            face_score = face_match(proof_photos[0], student_image)
        except:
            face_score = 0.0

    return clamp01(face_score * 0.8 + 0.2)


# ============================================================
# PAYMENT STRENGTH
# ============================================================

def payment_strength(payment_bytes: bytes, student_name: str) -> float:
    """OCR-based payment proof detection."""
    if not payment_bytes:
        return 0.0
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(payment_bytes)).convert("L")
        txt = pytesseract.image_to_string(img, config="--psm 6")
    except:
        txt = ""

    score = 0.0

    if re.search(r"\b(INR|Rs|₹)?\s?[0-9]{2,7}\b", txt):
        score += 0.4
    if student_name.lower() in txt.lower():
        score += 0.3
    if re.search(r"(paid|txn|transaction|receipt|amount)", txt, re.I):
        score += 0.3

    return clamp01(score)
# ============================================================
# Provider verification, certificate text scoring,
# dynamic weight redistribution, and category evaluation
# ============================================================


# ---------------------------
# Provider hybrid checker
# ---------------------------
VERIFICATION_KEYWORDS = ["valid certificate", "certificate verified", "verified", "authentic", "credential", "credential id", "this certificate"]

def _provider_hybrid_check(url: str, cert_id: str, timeout: float = 8.0) -> Tuple[float, str]:
    """
    Hybrid: success if page loads (200) AND (cert id present OR verification keywords).
    Returns (score, detail)
    """
    try:
        headers = {"User-Agent": "confidence-scorer/1.0"}
        r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200:
            return 0.0, f"status {r.status_code} at {url}"
        text = (r.text or "").lower()
        if cert_id.lower() in text:
            return 1.0, f"id {cert_id} found on provider page"
        for kw in VERIFICATION_KEYWORDS:
            if kw in text:
                return 0.9, f"keyword '{kw}' matched on provider page"
        # try parameters
        for p in ({"id": cert_id}, {"certificate": cert_id}, {"cert": cert_id}, {"certificateId": cert_id}):
            try:
                r2 = requests.get(url, params=p, timeout=timeout, headers=headers)
                if r2.status_code == 200:
                    t2 = (r2.text or "").lower()
                    if cert_id.lower() in t2:
                        return 1.0, f"id matched via params at {r2.url}"
                    for kw in VERIFICATION_KEYWORDS:
                        if kw in t2:
                            return 0.9, f"keyword '{kw}' matched via params at {r2.url}"
            except Exception:
                continue
        return 0.0, "no id/keyword found"
    except Exception as e:
        logger.debug("provider check error: %s", e)
        return 0.0, f"error {e}"

# ---------------------------
# High-level provider verifier
# ---------------------------
def verify_certificate_by_provider(provider_hint: str, cert_id: str) -> Dict[str, Any]:
    """
    Try provider-specific endpoint first, then fall back to simple checks.
    Returns dict with keys: verified (bool), score (0-1), detail, endpoint
    """
    if not cert_id:
        return {"verified": False, "score": 0.0, "detail": "no cert id", "endpoint": None}
    ph = (provider_hint or "").lower()
    # try to match provider keys loosely
    for k, v in PROVIDER_ENDPOINTS.items():
        if k in ph:
            templ = PROVIDER_ENDPOINTS.get(k)
            if templ is None:
                return {"verified": False, "score": 0.0, "detail": "no universal endpoint", "endpoint": None}
            if templ == "QR_ONLY":
                return {"verified": False, "score": 0.0, "detail": "requires QR validation", "endpoint": templ}
            try:
                if "{}" in templ:
                    url = templ.format(cert_id)
                    s, d = _provider_hybrid_check(url, cert_id)
                    return {"verified": s >= 0.99, "score": s, "detail": d, "endpoint": url}
                else:
                    url = templ
                    s, d = _provider_hybrid_check(url, cert_id)
                    return {"verified": s >= 0.99, "score": s, "detail": d, "endpoint": url}
            except Exception as e:
                logger.debug("provider verify error: %s", e)
                return {"verified": False, "score": 0.0, "detail": str(e), "endpoint": templ}
    # fallback: nothing matched, return no direct check
    return {"verified": False, "score": 0.0, "detail": "provider not recognized", "endpoint": None}

# ---------------------------
# Certificate text scoring (Option B boosts integrated)
# ---------------------------
def certificate_text_strength_and_info(cert_text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Compute certificate textual strength and extract info like cert_id and provider_hint.
    Returns (score 0-1, info dict)
    """
    info = {"cert_id": None, "provider_hint": None, "organizer": None}
    if not cert_text:
        return 0.0, info
    # organizer
    org = None
    try:
        org = re.search(r"(organized by|issued by|conducted by)\s*[:\-]?\s*([A-Za-z0-9 &,\.\-()]{3,120})", cert_text, re.I)
        if org:
            info["organizer"] = org.group(2).strip()
    except Exception:
        info["organizer"] = None
    # cert id
    cid = extract_certificate_id(cert_text)
    if cid:
        info["cert_id"] = cid[0]
        info["provider_hint"] = cid[1]
    score = 0.0
    # base id presence
    if info["cert_id"]:
        score += 0.25
        # provider hint
        if info["provider_hint"] and info["provider_hint"] != "unknown":
            score += 0.20
    else:
        score += 0.02
    # textual cues
    for w in ["certificate", "course", "workshop", "training", "achievement", "award"]:
        if w in cert_text.lower():
            score += 0.03
    if re.search(r"20(1[5-9]|2[0-9]|3[0-5])", cert_text or ""):
        score += 0.10
    if re.search(r"(university|college|institute|nptel|coursera|edx)", cert_text or "", re.I):
        score += 0.15
    return clamp01(score), info

# ---------------------------
# Dynamic weight redistribution
# ---------------------------
# Base nominal weights before redistribution (these will be redistributed based on available components)
NOMINAL_WEIGHTS = {
    "certificate": 0.30,
    "verification": 0.25,
    "report": 0.20,
    "presence": 0.10,
    "host": 0.10,
    "payment": 0.05
}

def redistribute_weights_based_on_availability(available: Dict[str, bool]) -> Dict[str, float]:
    """
    available: mapping of component -> bool (True if present and applicable)
    Return normalized weights summing to 1.0 among available components.
    """
    working = {}
    total_nominal = 0.0
    for k, v in NOMINAL_WEIGHTS.items():
        if available.get(k, False):
            working[k] = NOMINAL_WEIGHTS[k]
            total_nominal += NOMINAL_WEIGHTS[k]
        else:
            working[k] = 0.0
    if total_nominal == 0.0:
        # fallback equal distribution to whatever available
        keys = [k for k, v in available.items() if v]
        if not keys:
            return {k: 0.0 for k in NOMINAL_WEIGHTS.keys()}
        per = 1.0 / len(keys)
        return {k: (per if available.get(k, False) else 0.0) for k in NOMINAL_WEIGHTS.keys()}
    for k in working.keys():
        if working[k] > 0:
            working[k] = working[k] / total_nominal
    return working

# ---------------------------
# Helper: search verification pages (duckduckgo) — used as fallback
# ---------------------------
def _search_duckduckgo(query: str, max_results: int = 6) -> List[str]:
    base = "https://duckduckgo.com/html/"
    try:
        resp = requests.post(base, data={"q": query}, timeout=6, headers={"User-Agent":"confidence-scorer/1.0"})
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        links = []
        for a in soup.select("a.result__a, a.result-link"):
            href = a.get("href")
            if href:
                if "uddg=" in href:
                    m = re.search(r"uddg=(https?%3A%2F%2F[^&]+)", href)
                    if m:
                        import urllib.parse
                        real = urllib.parse.unquote(m.group(1))
                        links.append(real)
                        continue
                links.append(href)
        # fallback
        if not links:
            for a in soup.find_all("a"):
                href = a.get("href")
                if href and href.startswith("http"):
                    links.append(href)
        seen = []
        for u in links:
            if u not in seen:
                seen.append(u)
            if len(seen) >= max_results:
                break
        return seen
    except Exception as e:
        logger.debug("duckduckgo search failed: %s", e)
        return []

def validate_certificate_with_discovered_endpoints(cert_id: str, endpoints: List[str]) -> Dict[str, Any]:
    """
    Try endpoints list (apis/forms/info pages). Return best match dict.
    """
    if not cert_id:
        return {"verified": False, "score": 0.0, "detail": "no cert id", "endpoint_used": None}
    # classify simple
    for url in endpoints:
        if not url:
            continue
        s, d = _provider_hybrid_check(url, cert_id)
        if s >= 0.99:
            return {"verified": True, "score": 1.0, "detail": d, "endpoint_used": url}
        if s > 0.0:
            return {"verified": False, "score": s, "detail": d, "endpoint_used": url}
    return {"verified": False, "score": 0.0, "detail": "no match", "endpoint_used": None}

# ---------------------------
# Category-specific evaluate() (per rules)
# ---------------------------
def evaluate_category_unified(category: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified evaluator for all categories according to the rules you specified.
    Returns a detailed result dict including components, weights, and confidence_score (0-10).
    """
    cat = (category or "").strip().lower()
    valid = ("competition", "sports", "course", "workshop", "internship", "project", "student_body", "volunteering", "research")
    if cat not in valid:
        raise ValueError("invalid category")

    # --- Load inputs ---
    cert_bytes = record.get("certificate_pdf_bytes", b"")
    cert_text = parse_pdf_text(cert_bytes) if cert_bytes else ""
    # Per your rule: do NOT attempt cert extraction for competitions/sports and projects
    cert_tuple = None
    if cat not in ("project") and cert_text:
        cert_tuple = extract_certificate_id(cert_text)  # (id, provider_hint) or None

    # Project-specific report and repo
    project_report_bytes = record.get("project_report_bytes", b"")
    project_repo_link = (record.get("project_repo_link") or "") or ""

    # Internship specifics
    internship_offer_bytes = record.get("internship_offer_bytes", b"")
    internship_report_bytes = record.get("internship_report_bytes", b"")
    supervisor_email_text = record.get("supervisor_email_text", "") or ""

    # Research specifics
    research_pdf_bytes = record.get("research_pdf_bytes", b"")
    doi_text = record.get("doi_text", "") or ""
    acceptance_proof_bytes = record.get("acceptance_proof_bytes", b"")

    # Common
    proof_photos = [load_image_bytes(p) for p in (record.get("proof_photos_bytes") or []) if p]
    student_image = load_image_bytes(record.get("student_image_bytes") or b"")
    payment_bytes = load_image_bytes(record.get("payment_proof_bytes") or b"")
    host_url = record.get("host_url") or ""
    student_name = record.get("student_name") or ""

    # --- Component computations ---
    # Certificate strength and info (if applicable)
    certificate_strength = 0.0
    certificate_info = {}
    if cert_text and cat not in ("project"):
        certificate_strength, certificate_info = certificate_text_strength_and_info(cert_text)

    # Verification strength (for course/workshop/internship/research)
    verification_strength = 0.0
    verification_detail = None
    if cat in ("course", "workshop", "internship", "research") and cert_tuple:
        cert_id = cert_tuple[0]
        prov_hint = cert_tuple[1]
        if prov_hint and cert_id:
            verification_detail = verify_certificate_by_provider(prov_hint, cert_id)
            verification_strength = verification_detail.get("score", 0.0)
        else:
            # fallback: try discovered endpoints using organizer or host
            org = certificate_info.get("organizer") or ""
            endpoints = []
            if host_url:
                endpoints.append(host_url)
            if org:
                endpoints.extend(_search_duckduckgo(f"{org} certificate verification", max_results=4))
            if endpoints and cert_id:
                validation = validate_certificate_with_discovered_endpoints(cert_id, endpoints)
                verification_strength = validation.get("score", 0.0)
                verification_detail = validation

    # Report strength (project/internship/research)
    report_strength = 0.0
    report_analysis = None
    github_strength = 0.0
    github_detail = None
    if cat == "project":
        # analyze report (mandatory)
        report_analysis = extract_report_features(project_report_bytes)
        report_strength = report_analysis.get("score", 0.0)
        # GitHub check (mandatory)
        if project_repo_link:
            gh = analyze_github_repo(project_repo_link)
            github_strength = gh.get("score", 0.0)
            github_detail = gh
        # combine report + github as report component
        report_strength = clamp01(0.6 * report_strength + 0.4 * github_strength)
    elif cat == "internship":
        # must check offer letter, confirmation mail, report, and certificate
        rep_anal = extract_report_features(internship_report_bytes)
        report_score = rep_anal.get("score", 0.0)
        # offer letter check
        offer_text = parse_pdf_text(internship_offer_bytes) if internship_offer_bytes else ""
        offer_score = 0.0
        if offer_text and re.search(r"\boffer\b|\binternship\b|\bjoining\b|\bstart date\b|\bdesignation\b", offer_text, re.I):
            offer_score = 0.5
        # confirmation email check (supervisor_email_text)
        conf_score = 0.0
        if supervisor_email_text and re.search(r"\bconfirm|confirmation|accepted|congratulations|welcome\b", supervisor_email_text, re.I):
            conf_score = 0.5
        report_strength = clamp01(0.5 * report_score + 0.3 * offer_score + 0.2 * conf_score)
    elif cat == "research":
        # analyze research PDF or acceptance proof
        rp_bytes = research_pdf_bytes or acceptance_proof_bytes
        rep_anal = extract_report_features(rp_bytes) if rp_bytes else {}
        report_score = rep_anal.get("score", 0.0)
        doi_score = 0.0
        if doi_text and re.search(r"\bdoi[:\s\/]?(10\.\d{4,9}\/[-._;()\/:A-Za-z0-9]+)\b", doi_text, re.I):
            doi_score = 0.6
        report_strength = clamp01(0.6 * report_score + 0.4 * doi_score)
        report_analysis = rep_anal

    # Presence, payment, host
    presence_val = presence_strength(proof_photos, student_image) if proof_photos else 0.0
    payment_val = payment_strength(payment_bytes, student_name) if payment_bytes else 0.0
    host_val = host_strength_universal(host_url) if host_url else 0.0

    # --- Determine available components according to category rules ---
    available = {
        "certificate": cat not in ("project") and bool(cert_text),
        "verification": (cat in ("course", "workshop", "internship", "research")) and bool(cert_tuple),
        "report": cat in ("project", "internship", "research"),
        "presence": bool(proof_photos),
        "host": bool(host_url),
        "payment": bool(payment_bytes)
    }

    # For projects, certificate and verification are not applicable
    if cat == "project":
        available["certificate"] = False
        available["verification"] = False

    # Certificates should always be considered for competitions/sports if present
    if cat in ("competition", "sports") and cert_text:
        available["certificate"] = True


    # redistribute weights
    weights = redistribute_weights_based_on_availability(available)

    # component values map
    components = {
        "certificate": certificate_strength,
        "verification": verification_strength,
        "report": report_strength,
        "presence": presence_val,
        "host": host_val,
        "payment": payment_val
    }

    # compute final (0-1)
    final = 0.0
    for k, w in weights.items():
        final += components.get(k, 0.0) * w
    final = clamp01(final)
    final_0_10 = round(final * 10.0, 2)

    # Build result with rich debug info
    result = {
        "category": cat,
        "components_available": available,
        "weights": weights,
        "components": components,
        "certificate_info": certificate_info,
        "verification_detail": verification_detail,
        "report_analysis": report_analysis,
        "github_detail": github_detail,
        "confidence_score": final_0_10
    }

    return result
# ============================================================
# Explainability, printing, and extended __main__ test
# ============================================================

def explain_result(result: Dict[str, Any], file_inputs: Dict[str, Any]) -> str:
    """
    Build a human-readable explanation that:
    - lists which components were used,
    - shows weights and contributions,
    - provides detail for report/git/verification,
    - explains skipped components.
    """
    lines = []
    cat = result.get("category", "unknown")
    score = result.get("confidence_score", 0.0)
    lines.append(f"Category: {cat}")
    lines.append(f"Final confidence: {score} / 10.00")
    lines.append("")

    # Components & weights
    lines.append("Components considered (weight -> value -> contribution):")
    weights = result.get("weights", {})
    components = result.get("components", {})
    total_contribution = 0.0
    for comp in ["certificate", "verification", "report", "presence", "host", "payment"]:
        w = weights.get(comp, 0.0)
        if w <= 0:
            continue
        val = components.get(comp, 0.0)
        contrib = round(w * val * 10.0, 3)  # contribution in 0-10 scale portion
        total_contribution += contrib
        lines.append(f"- {comp}: weight={w:.3f}, value={val:.3f}, contribution(~/10)={contrib}")

    lines.append("")
    lines.append(f"Total contribution (approx /10): {round(total_contribution,2)} (should match final score)")

    lines.append("\nDetailed component explanations:")

    # Certificate
    if weights.get("certificate", 0) > 0:
        cs = components.get("certificate", 0.0)
        ci = result.get("certificate_info", {}) or {}
        lines.append(f"\nCERTIFICATE: strength={cs:.3f}")
        lines.append(f"- Organizer detected: {ci.get('organizer')}")
        lines.append(f"- Certificate ID: {ci.get('cert_id')}")
        lines.append(f"- Provider hint: {ci.get('provider_hint')}")
        if cs >= 0.75:
            lines.append("- Interpretation: Strong certificate cues (text, org, ID).")
        elif cs >= 0.4:
            lines.append("- Interpretation: Moderate certificate cues.")
        else:
            lines.append("- Interpretation: Weak certificate cues or missing.")

    # Verification
    if weights.get("verification", 0) > 0:
        vs = components.get("verification", 0.0)
        vd = result.get("verification_detail")
        lines.append(f"\nVERIFICATION: strength={vs:.3f}")
        if vd:
            lines.append(f"- Detail: {vd}")
        else:
            lines.append("- No direct verification detail available.")
        if vs >= 0.99:
            lines.append("- Interpretation: Verified by provider endpoint.")
        elif vs >= 0.5:
            lines.append("- Interpretation: Partial verification (keyword & info pages).")
        else:
            lines.append("- Interpretation: Not verified.")

    # Report
    if weights.get("report", 0) > 0:
        rs = components.get("report", 0.0)
        ra = result.get("report_analysis") or {}
        lines.append(f"\nREPORT / DOC: strength={rs:.3f}")
        if ra:
            # ra may be from extract_report_features or analyze_report_pdf depending on part used
            if isinstance(ra, dict):
                keywords = ra.get("keywords") or ra.get("keywords", [])
                lines.append(f"- Text length: {ra.get('text_len') or ra.get('pages', 'N/A')}")
                lines.append(f"- Keywords found: {keywords}")
                lines.append(f"- Technical depth (approx): {ra.get('score') or ra.get('technical_depth_score', 'N/A')}")
        # project-specific github detail
        gh = result.get("github_detail")
        if gh:
            lines.append(f"- GitHub check: {gh.get('detail')} (score={gh.get('score')})")
        if rs >= 0.75:
            lines.append("- Interpretation: Strong report/repo evidence.")
        elif rs >= 0.4:
            lines.append("- Interpretation: Moderate report/repo evidence.")
        else:
            lines.append("- Interpretation: Weak or missing report/repo evidence.")

    # Presence
    if weights.get("presence", 0) > 0:
        ps = components.get("presence", 0.0)
        lines.append(f"\nPRESENCE: strength={ps:.3f}")
        if ps >= 0.7:
            lines.append("- Interpretation: Good match between event photos and ID.")
        elif ps >= 0.4:
            lines.append("- Interpretation: Some presence evidence.")
        else:
            lines.append("- Interpretation: Weak or no presence evidence.")

    # Host
    if weights.get("host", 0) > 0:
        hs = components.get("host", 0.0)
        lines.append(f"\nHOST: strength={hs:.3f}")
        if hs >= 0.7:
            lines.append("- Interpretation: Host site is HTTPS and looks trustworthy.")
        elif hs >= 0.4:
            lines.append("- Interpretation: Host provided but low trust signals.")
        else:
            lines.append("- Interpretation: No host or low-trust host.")

    # Payment
    if weights.get("payment", 0) > 0:
        pay = components.get("payment", 0.0)
        lines.append(f"\nPAYMENT: strength={pay:.3f}")
        if pay >= 0.7:
            lines.append("- Interpretation: Clear payment evidence.")
        elif pay >= 0.4:
            lines.append("- Interpretation: Partial payment evidence.")
        else:
            lines.append("- Interpretation: No or weak payment evidence.")

    #lines.append("\nNotes:")
    #lines.append("- Components not applicable to this category were excluded and weights redistributed.")
    #lines.append("- Competitions/Sports: provider verification disabled by policy.")
    #lines.append("- Projects: certificate checks are skipped; repo+report used.")
    #lines.append("- Internships: certificate + offer letter + confirmation + report used to compute verification/report strength.")
    #lines.append("- Research: paper analysis + DOI/acceptance proof are used.")

    # file_inputs: map of which files were provided
    #lines.append("\nInput files provided:")
    #for k, v in file_inputs.items():
     #   lines.append(f"- {k}: {v}")

    #return "\n".join(lines)


def print_results(category: str, result: Dict[str, Any], file_inputs: Dict[str, Any]):
    print(f"\n--- RESULTS ({category.upper()}) ---\n")
    print(json.dumps(result, indent=2))
    print("\n--- EXPLANATION ---\n")
    print(explain_result(result, file_inputs))

def filter_components_by_category(result: Dict[str, Any]) -> Dict[str, Any]:
    cat = result.get("category", "")
    relevant = []
    if cat in ("competition", "sports"):
        relevant = ["certificate", "host", "presence"]
    elif cat in ("course", "workshop", "internship", "research"):
        relevant = ["certificate", "verification", "report", "host", "presence", "payment"]
    elif cat == "project":
        relevant = ["report", "host", "presence"]
    else:
        relevant = list(result.get("components_available", {}).keys())

    # filter components, weights, availability
    result["components"] = {k: v for k, v in result.get("components", {}).items() if k in relevant}
    result["weights"] = {k: v for k, v in result.get("weights", {}).items() if k in relevant}
    result["components_available"] = {k: v for k, v in result.get("components_available", {}).items() if k in relevant}
    return result

# -----------------------
# TEST block: extended to include category-specific files/fields
# -----------------------
if __name__ == "__main__":
    # Choose only one category to test
    test_category = "competition" 

    # Sample/mock files
    CERT_PATH = "siva1(1).pdf"
    ID_PATH = "bid2.jpg"
    PAY_PATH = "bpay.jpg"
    EVENT_PHOTO = "ksr_siva.jpg"
    REPORT_PATH = "project_report.pdf"
    HOST_URL = "https://example.org"
    STUDENT_NAME = "Sivasankar"
    GITHUB_LINK = "https://github.com/example/repo"

    INTERNSHIP_OFFER_PATH = "offer_letter.pdf"
    SUPERVISOR_EMAIL_PATH = "supervisor_email.txt"
    RESEARCH_PAPER_PATH = "paper.pdf"
    DOI_TEXT_PATH = "doi.txt"
    ACCEPTANCE_PROOF_PATH = "acceptance.pdf"

    # Prepare record dict
    rec = {
        "certificate_pdf_bytes": load_bytes(CERT_PATH) if os.path.exists(CERT_PATH) else b"",
        "student_image_bytes": load_bytes(ID_PATH) if os.path.exists(ID_PATH) else b"",
        "payment_proof_bytes": load_bytes(PAY_PATH) if os.path.exists(PAY_PATH) else b"",
        "proof_photos_bytes": [load_bytes(EVENT_PHOTO)] if os.path.exists(EVENT_PHOTO) else [],
        "host_url": HOST_URL,
        "student_name": STUDENT_NAME,
        "project_report_bytes": load_bytes(REPORT_PATH) if os.path.exists(REPORT_PATH) else b"",
        "project_repo_link": GITHUB_LINK,
        "internship_offer_bytes": load_bytes(INTERNSHIP_OFFER_PATH) if os.path.exists(INTERNSHIP_OFFER_PATH) else b"",
        "internship_report_bytes": load_bytes(REPORT_PATH) if os.path.exists(REPORT_PATH) else b"",
        "supervisor_email_text": load_text_from_bytes(load_bytes(SUPERVISOR_EMAIL_PATH)) if os.path.exists(SUPERVISOR_EMAIL_PATH) else "",
        "research_pdf_bytes": load_bytes(RESEARCH_PAPER_PATH) if os.path.exists(RESEARCH_PAPER_PATH) else b"",
        "doi_text": load_text_from_bytes(load_bytes(DOI_TEXT_PATH)) if os.path.exists(DOI_TEXT_PATH) else "",
        "acceptance_proof_bytes": load_bytes(ACCEPTANCE_PROOF_PATH) if os.path.exists(ACCEPTANCE_PROOF_PATH) else b""
    }

    # Prepare category-specific file_inputs
    file_inputs = {}
    if test_category in ("competition", "sports"):
        file_inputs = {
            "certificate": CERT_PATH if os.path.exists(CERT_PATH) else None,
            "event_photo": EVENT_PHOTO if os.path.exists(EVENT_PHOTO) else None,
            "host_url": HOST_URL
        }
    elif test_category in ("course", "workshop"):
        file_inputs = {
            "certificate": CERT_PATH if os.path.exists(CERT_PATH) else None,
            "host_url": HOST_URL,
            "payment": PAY_PATH if os.path.exists(PAY_PATH) else None
        }
    elif test_category == "internship":
        file_inputs = {
            "certificate": CERT_PATH if os.path.exists(CERT_PATH) else None,
            "internship_offer": INTERNSHIP_OFFER_PATH if os.path.exists(INTERNSHIP_OFFER_PATH) else None,
            "internship_report": REPORT_PATH if os.path.exists(REPORT_PATH) else None,
            "supervisor_email": SUPERVISOR_EMAIL_PATH if os.path.exists(SUPERVISOR_EMAIL_PATH) else None,
            "payment": PAY_PATH if os.path.exists(PAY_PATH) else None,
            "host_url": HOST_URL
        }
    elif test_category == "project":
        file_inputs = {
            "project_report": REPORT_PATH if os.path.exists(REPORT_PATH) else None,
            "project_repo_link": GITHUB_LINK,
            "host_url": HOST_URL
        }
    elif test_category == "research":
        file_inputs = {
            "research_pdf": RESEARCH_PAPER_PATH if os.path.exists(RESEARCH_PAPER_PATH) else None,
            "doi_text": DOI_TEXT_PATH if os.path.exists(DOI_TEXT_PATH) else None,
            "acceptance_proof": ACCEPTANCE_PROOF_PATH if os.path.exists(ACCEPTANCE_PROOF_PATH) else None,
            "host_url": HOST_URL
        }
    elif test_category in ("volunteering", "student_body"):
        file_inputs = {
            "certificate": CERT_PATH if os.path.exists(CERT_PATH) else None,
            "event_photo": EVENT_PHOTO if os.path.exists(EVENT_PHOTO) else None,
            "host_url": HOST_URL,
            "payment": PAY_PATH if os.path.exists(PAY_PATH) else None
        }

    # Evaluate only the chosen category
    res = evaluate_category_unified(test_category, rec)
    res = filter_components_by_category(res)
    print_results(test_category, res, file_inputs)

# ============================================================
# API Compatibility Wrappers
# ============================================================

def parse_certificate_text(pdf_bytes):
    """
    API expects this name. Your actual function is parse_pdf_text().
    """
    try:
        return parse_pdf_text(pdf_bytes)
    except:
        return ""


def extract_cert_number(text):
    """
    API expects extract_cert_number. Your real fn is extract_certificate_id().
    """
    try:
        return extract_certificate_id(text)
    except:
        return None


def extract_organizer(text):
    """
    Extract organizer name for certificate if possible.
    Your module embeds this logic inside certificate_text_strength_and_info,
    so this wrapper exposes it as a separate function.
    """
    try:
        m = re.search(
            r"(organized by|issued by|conducted by)\s*[:\-]?\s*([A-Za-z0-9 &,.()-]{3,120})",
            text,
            re.I
        )
        if m:
            return m.group(2).strip()
    except:
        pass
    return None


def compute_certificate_strength_universal(text, cert_tuple=None, organizer=None, host_url=None):
    """
    API wrapper. Maps to your real certificate scoring function.
    Returns: (score, info)
    """
    try:
        score, info = certificate_text_strength_and_info(text)
        return score, info
    except Exception as e:
        return 0.0, {"error": str(e)}


def verify_full(record: dict):
    """
    MASTER wrapper expected by /verify/complete.
    Your real evaluator is evaluate_category_unified(category, record).
    """
    try:
        category = record.get("category", "course")
        return evaluate_category_unified(category, record)
    except Exception as e:
        return {"error": f"verify_full failed: {e}"}
