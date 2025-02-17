import evaluate
import numpy as np


seqeval = evaluate.load("seqeval")


def compute_metrics(dataset_blueprint):
    def compute_internal(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [
                dataset_blueprint.id2label[p]
                for (p, l) in zip(prediction, label)
                if l != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [
                dataset_blueprint.id2label[l]
                for (p, l) in zip(prediction, label)
                if l != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_internal


def compute_metrics_baseline(dataset_blueprint):
    def compute_internal(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [
                dataset_blueprint.id2label[p.item()]
                for (p, l) in zip(prediction, label)
                if l != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [
                dataset_blueprint.id2label[l.item()]
                for (p, l) in zip(prediction, label)
                if l != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_internal
