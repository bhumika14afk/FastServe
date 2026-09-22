"""
Model loading utilities for FastServe.

Supports swapping between a tiny local-testing model (CPU-friendly)
and the real baseline model (meant to run on a GPU instance).
Controlled via the MODEL_NAME environment variable or configs/model_config.yaml.
"""
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "model_config.yaml")

# Small model for local CPU testing/wiring checks — swap to the real
# baseline (e.g. meta-llama/Meta-Llama-3-8B) once you're on a GPU instance.
LOCAL_TEST_MODEL = "sshleifer/tiny-gpt2"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_baseline_model(use_local_test_model: bool = False):
    """
    Loads the baseline model and tokenizer.

    Args:
        use_local_test_model: if True, loads a tiny CPU-friendly model
            so you can verify the pipeline works before running the
            real model on GPU. Set to False for actual GPU inference.
    """
    if use_local_test_model:
        model_name = LOCAL_TEST_MODEL
    else:
        config = load_config()
        model_name = config["baseline"]["name"]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
