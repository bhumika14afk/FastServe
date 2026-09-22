"""
FastAPI gateway for FastServe.

Serves ticket-triage requests using the baseline model.
Run locally with the tiny test model first (USE_LOCAL_TEST_MODEL=true)
to confirm wiring, then switch to the real baseline model on a GPU instance.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel

from serving.model_loader import load_baseline_model, generate_response

app = FastAPI(title="FastServe API")

# Toggle via environment variable — defaults to the tiny local test model
# so this runs fine on your laptop CPU without downloading an 8B model.
USE_LOCAL_TEST_MODEL = os.getenv("USE_LOCAL_TEST_MODEL", "true").lower() == "true"

# Loaded once at startup, not per-request
model, tokenizer = load_baseline_model(use_local_test_model=USE_LOCAL_TEST_MODEL)


class TicketRequest(BaseModel):
    ticket_text: str


class TicketResponse(BaseModel):
    raw_output: str


@app.get("/health")
def health_check():
    return {"status": "ok", "using_local_test_model": USE_LOCAL_TEST_MODEL}


@app.post("/triage", response_model=TicketResponse)
def triage_ticket(request: TicketRequest):
    """
    Classify a support ticket, extract entities, and draft a first response.

    This baseline version just prompts the model directly. Once wired up,
    this is what you'll benchmark against the quantized/distilled variants.
    """
    prompt = (
        "You are a customer support triage assistant. "
        "Classify the ticket, identify severity, and draft a first response.\n\n"
        f"Ticket: {request.ticket_text}\n\n"
        "Response:"
    )
    output = generate_response(model, tokenizer, prompt)
    return TicketResponse(raw_output=output)
