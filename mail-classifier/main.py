# mail-classifier/main.py
import os
import re
import json
import time
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
import numpy as np
import requests
from datetime import datetime
import dateparser

# ML
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mail-classifier")

# ----- Configuration from env -----
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
MODEL_PATH = os.getenv("MODEL_PATH")  
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NOTIFY_THRESHOLD = float(os.getenv("NOTIFY_THRESHOLD", "0.80"))
FALLBACK_THRESHOLD = float(os.getenv("FALLBACK_THRESHOLD", "0.45"))  
DATA_DIR = os.getenv("DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----- FastAPI app -----
app = FastAPI(title="Mail Classifier")

# ----- Models / Classifier load -----
logger.info("Cargando modelo de embeddings...")
try:
    if MODEL_PATH:
        # if user mounted a local model path, SentenceTransformer will load by name or path
        embedder = SentenceTransformer(MODEL_PATH)
    else:
        embedder = SentenceTransformer(MODEL_NAME)
    logger.info("Embeddings cargados OK.")
except Exception as e:
    logger.exception("Error cargando embeddings: %s", e)
    raise

clf = None
clf_path = os.path.join(".", "clf.pkl")
if os.path.exists(clf_path):
    try:
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        logger.info("Clasificador cargado desde clf.pkl")
    except Exception as e:
        logger.exception("Error cargando clf.pkl: %s", e)
        clf = None
else:
    logger.info("No se encontró clf.pkl — se usará heurística/similitud.")

# Prepare small set of positive/negative example texts for similarity fallback
POSITIVE_EXAMPLES = [
    "Confirmación de vuelo PNR: ABC123. Fecha de salida 2025-12-01. Itinerario incluido.",
    "Factura adjunta por importe €123,45. Vencimiento: 2025-11-30. Empresa: ACME S.A.",
    "Pago recibido: extracto bancario, cargo en su cuenta. Aviso importante.",
    "Confirmación de reserva de hotel - check in 2025-12-05. Código de reserva: HTRT56."
]
NEGATIVE_EXAMPLES = [
    "Boletín semanal: nuestras ofertas del mes. Suscríbete para recibir descuentos.",
    "Promoción: 50% en productos. No requiere acción urgente.",
    "Newsletter de la comunidad con eventos y noticias generales."
]

logger.info("Computando embeddings de ejemplos...")
pos_embs = embedder.encode(POSITIVE_EXAMPLES, convert_to_numpy=True)
neg_embs = embedder.encode(NEGATIVE_EXAMPLES, convert_to_numpy=True)


# ----- Helpers -----
def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts/styles
    for s in soup(["script", "style"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    return text.strip()


def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # remove excessive whitespace
    return re.sub(r"\s+", " ", s).strip()


def extract_amounts(text: str):
    # basic regex to capture things like €123,45  / $123.45 / 123,45 €
    patterns = [
        r"€\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?",
        r"\$\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?",
        r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s?(?:EUR|€|EUR\.)"
    ]
    matches = []
    for p in patterns:
        matches += re.findall(p, text, flags=re.IGNORECASE)
    return matches


def extract_pnr(text: str):
    # PNR/reservation codes are often 6 alnum uppercase, tune if needed
    m = re.findall(r"\b[A-Z0-9]{6}\b", text)
    # filter common words that match pattern accidentally (e.g., 'PLEASE')
    res = [x for x in m if not re.match(r"^\d{6}$", x) or len(x) == 6]
    return res


def try_extract_dates(text: str):
    # use dateparser to look for probable dates; returns list of datetimes
    try:
        found = dateparser.search.search_dates(text, languages=['es', 'en'])
        if not found:
            return []
        return [d[1] for d in found]
    except Exception:
        return []


def rules_quick_check(subject: str, sender: str, body: str, has_attachment: bool):
    """
    Rules that immediately mark a message as important (fast wins).
    Returns (matched: bool, reason: str, category: str)
    """
    subj = subject.lower() if subject else ""
    sfrom = sender.lower() if sender else ""
    body_lower = body.lower() if body else ""

    # 1) PNR codes
    pnr_codes = extract_pnr(body)
    if pnr_codes:
        return (True, f"Código PNR detectado: {pnr_codes[0]}", "flight")

    # 2) Amounts
    amounts = extract_amounts(body)
    if amounts:
        return (True, f"Importe detectado: {amounts[0]}", "financial")

    # 3) Keywords in subject
    important_keywords = ["factura", "invoice", "pago", "payment", "confirmación", "confirmation", "reserva", "booking"]
    for kw in important_keywords:
        if kw in subj:
            return (True, f"Palabra clave en asunto: '{kw}'", "important_keyword")

    # 4) Dates in near future
    dates = try_extract_dates(body)
    if dates:
        now = datetime.now()
        for d in dates:
            if d > now:
                delta = (d - now).days
                if delta <= 30:
                    return (True, f"Fecha próxima detectada: {d.strftime('%Y-%m-%d')}", "upcoming_event")

    return (False, "", "")


# ----- Pydantic models -----
class EmailClassifyRequest(BaseModel):
    subject: str = ""
    from_: str = ""
    body: str = ""
    date: Optional[str] = None
    has_attachment: bool = False

    class Config:
        fields = {"from_": "from"}


class EmailClassifyResponse(BaseModel):
    important: bool
    reason: str
    confidence: float
    category: str = ""
    subject: str = ""
    from_: str = ""

    class Config:
        fields = {"from_": "from"}


# ----- Main endpoint -----
@app.post("/classify", response_model=EmailClassifyResponse)
async def classify_email(req: EmailClassifyRequest):
    """
    Classify an email as important or not.
    """
    try:
        # Extract text from HTML if needed
        body_text = html_to_text(req.body) if req.body else ""
        body_clean = clean_text(body_text)

        # 1) Quick rules check
        matched, reason, category = rules_quick_check(
            req.subject, req.from_, body_clean, req.has_attachment
        )
        if matched:
            return EmailClassifyResponse(
                important=True,
                reason=reason,
                confidence=0.95,
                category=category,
                subject=req.subject,
                from_=req.from_
            )

        # 2) If we have a trained classifier, use it
        if clf is not None:
            combined_text = f"{req.subject} {body_clean}"
            emb = embedder.encode([combined_text], convert_to_numpy=True)
            pred = clf.predict(emb)[0]
            proba = clf.predict_proba(emb)[0]
            confidence = float(max(proba))
            
            if pred == 1:
                return EmailClassifyResponse(
                    important=True,
                    reason="Clasificador ML: importante",
                    confidence=confidence,
                    category="ml_prediction",
                    subject=req.subject,
                    from_=req.from_
                )
            else:
                return EmailClassifyResponse(
                    important=False,
                    reason="Clasificador ML: no importante",
                    confidence=confidence,
                    category="ml_prediction",
                    subject=req.subject,
                    from_=req.from_
                )

        # 3) Fallback: similarity with examples
        combined_text = f"{req.subject} {body_clean}"
        emb = embedder.encode([combined_text], convert_to_numpy=True)
        
        pos_sim = cosine_similarity(emb, pos_embs).max()
        neg_sim = cosine_similarity(emb, neg_embs).max()
        
        if pos_sim > neg_sim and pos_sim > FALLBACK_THRESHOLD:
            return EmailClassifyResponse(
                important=True,
                reason=f"Similitud con ejemplos importantes ({pos_sim:.2f})",
                confidence=float(pos_sim),
                category="similarity",
                subject=req.subject,
                from_=req.from_
            )
        else:
            return EmailClassifyResponse(
                important=False,
                reason=f"Similitud baja o negativa ({neg_sim:.2f})",
                confidence=float(neg_sim),
                category="similarity",
                subject=req.subject,
                from_=req.from_
            )

    except Exception as e:
        logger.exception("Error clasificando email: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


# ----- Run server -----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
