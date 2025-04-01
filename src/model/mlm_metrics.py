import torch
import evaluate
from transformers import EvalPrediction
from math import exp

f1_metric = evaluate.load("f1")


def compute_metrics(p: EvalPrediction):
    logits, labels = p.predictions, p.label_ids

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)

    predictions = torch.argmax(probs, dim=-1)

    mask = labels != -100

    correct_predictions = (predictions[mask] == labels[mask]).sum().item()
    total_masked_tokens = mask.sum().item()
    masked_accuracy = (
        correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(torch.tensor(logits[mask]), torch.tensor(labels[mask]))
    perplexity = exp(loss.item())

    f1_score = f1_metric.compute(
        predictions=predictions[mask].tolist(),
        references=labels[mask].tolist(),
        average="macro",
    )

    return {
        "masked_accuracy": masked_accuracy,
        "perplexity": perplexity,
        "f1_score": f1_score["f1"],
    }
