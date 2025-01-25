
def calculate_metrics(predictions_b, ground_truth_b, predictions_c, ground_truth_c):
    """
    Calculate evaluation metrics for facial recognition on known (B) and unknown (C) datasets.

    Args:
        predictions_b (list or np.ndarray): Predicted labels for known individuals (B).
        ground_truth_b (list or np.ndarray): True labels for known individuals (B).
        predictions_c (list or np.ndarray): Predicted labels for unknown individuals (C).
        ground_truth_c (list or np.ndarray): True labels for unknown individuals (C).

    Returns:
        dict: Dictionary containing calculated metrics.
    """
    metrics = {}

    # Known individuals (B) metrics
    metrics["Precision_B"] = precision_score(ground_truth_b, predictions_b, average="weighted", zero_division=0)
    metrics["Recall_B"] = recall_score(ground_truth_b, predictions_b, average="weighted", zero_division=0)
    metrics["F1_Score_B"] = f1_score(ground_truth_b, predictions_b, average="weighted", zero_division=0)

    # Unknown individuals (C) metrics
    metrics["Precision_C"] = precision_score(ground_truth_c, predictions_c, average="binary", pos_label=9999, zero_division=0)
    metrics["Recall_C"] = recall_score(ground_truth_c, predictions_c, average="binary", pos_label=9999, zero_division=0)
    metrics["F1_Score_C"] = f1_score(ground_truth_c, predictions_c, average="binary", pos_label=9999, zero_division=0)

    # False Acceptance Rate (FAR) for C
    false_accepts = np.sum((np.array(predictions_c) != 9999) & (np.array(ground_truth_c) == 9999))
    total_unknown = np.sum(np.array(ground_truth_c) == 9999)
    metrics["FAR"] = false_accepts / total_unknown if total_unknown > 0 else 0

    # False Rejection Rate (FRR) for B
    false_rejects = np.sum((np.array(predictions_b) == 9999) & (np.array(ground_truth_b) != 9999))
    total_known = len(ground_truth_b)
    metrics["FRR"] = false_rejects / total_known if total_known > 0 else 0

    # Accuracy
    correct_b = np.sum(np.array(predictions_b) == np.array(ground_truth_b))
    correct_c = np.sum(np.array(predictions_c) == np.array(ground_truth_c))
    total_predictions = len(ground_truth_b) + len(ground_truth_c)
    metrics["Accuracy"] = (correct_b + correct_c) / total_predictions if total_predictions > 0 else 0

    # Recognition Rate
    metrics["Recognition_Rate"] = (correct_b / len(ground_truth_b)) * 100 if len(ground_truth_b) > 0 else 0

    # Average Error Rate
    metrics["Average_Error"] = (metrics["FAR"] + metrics["FRR"]) / 2

    # Fail Rate
    metrics["Fail_Rate"] = (1 - metrics["Accuracy"]) * 100

    return metrics

# Example usage:
# predictions_b = [1, 2, 1, 3]  # Example predicted labels for known individuals
# ground_truth_b = [1, 2, 1, 3]  # Example true labels for known individuals
# predictions_c = [9999, 1, 9999]  # Example predicted labels for unknown individuals
# ground_truth_c = [9999, 9999, 9999]  # Example true labels for unknown individuals

# metrics = calculate_metrics(predictions_b, ground_truth_b, predictions_c, ground_truth_c)
# print(metrics)