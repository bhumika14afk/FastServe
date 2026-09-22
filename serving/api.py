"""
FastAPI gateway for FastServe.
Serves requests to whichever model backend is configured (baseline, quantized, distilled).
"""
from fastapi import FastAPI

app = FastAPI(title="FastServe API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/triage")
def triage_ticket(ticket_text: str):
    """
    Classify a support ticket, extract entities, and draft a first response.
    TODO: wire up model backend.
    """
    raise NotImplementedError("Model backend not yet connected.")
