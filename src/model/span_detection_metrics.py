import evaluate
import numpy as np
from sklearn.metrics import classification_report

seqeval = evaluate.load("seqeval")


def compute_metrics(dataset_blueprint):
    def compute_internal(p):
        """
        Evaluate both token-level and span-level metrics.
        
        Args:
            y_true: List of lists containing true binary labels (0/1)
            y_pred: List of lists containing predicted binary labels (0/1)
            dataset_blueprint: Dataset object containing label mappings
        
        Returns:
            dict: Dictionary containing both token and span level metrics
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [
                p
                for (p, l) in zip(prediction, label)
                if l != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [
                l
                for (p, l) in zip(prediction, label)
                if l != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]
        # Get token-level metrics
        token_metrics = evaluate_tokens(true_labels, true_predictions)
        
        # Convert to IO format for seqeval span-level evaluation
        y_true_io = convert_to_io(true_labels, dataset_blueprint.id2label)
        y_pred_io = convert_to_io(true_predictions, dataset_blueprint.id2label)
        
        # Calculate span-level metrics
        results = seqeval.compute(predictions=y_pred_io, references=y_true_io)
        return {
            # Token-level metrics
            'token_f1': token_metrics['token_f1'],
            'token_precision': token_metrics['token_precision'],
            'token_recall': token_metrics['token_recall'],
            'token_accuracy': token_metrics['token_accuracy'],
            # Span-level metrics
            'span_f1': results['overall_f1'],
            'span_precision': results['overall_precision'],
            'span_recall': results['overall_recall'],
            'span_accuracy': results['overall_accuracy'],
        }
    
    return compute_internal


def convert_to_io(labels, id2label):
    """
    Convert binary labels to IO format.
    1 -> I-MANIPULATION
    0 -> O

    Args:
        labels: List of lists containing binary labels (0/1)
        id2label: Dictionary mapping label IDs to label names

    Returns:
        List of lists containing IO
    """
    io_labels = []
    for sequence in labels:
        io_sequence = ['I-MANIPULATION' if label == 1 else 'O' for label in sequence]
        io_labels.append(io_sequence)
    return io_labels


def evaluate_tokens(y_true, y_pred):
    """
    Evaluate token-level metrics without considering span continuity.
    
    Args:
        y_true: List of lists containing true binary labels (0/1)
        y_pred: List of lists containing predicted binary labels (0/1)
    
    Returns:
        dict: Dictionary containing token-level F1, precision, and recall scores
    """
    # Flatten the nested lists
    y_true_flat = [label for seq in y_true for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]
    
    # Calculate token-level metrics
    report = classification_report(y_true_flat, y_pred_flat, 
                                labels=[0, 1],
                                target_names=['O', 'I-MANIPULATION'],
                                output_dict=True)
    
    return {
        'token_f1': report['I-MANIPULATION']['f1-score'],
        'token_precision': report['I-MANIPULATION']['precision'],
        'token_recall': report['I-MANIPULATION']['recall'],
        'token_accuracy': report['accuracy'],
    }
