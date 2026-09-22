"""
Evaluation harness for FastServe model variants.
Measures task accuracy, latency, throughput, and cost per 1K requests.
"""


def evaluate_accuracy(model, eval_set):
    """Run the model over the labeled eval set and return accuracy/F1."""
    raise NotImplementedError


def evaluate_latency(model, num_requests: int = 100):
    """Measure p50/p95/p99 latency over N requests."""
    raise NotImplementedError
