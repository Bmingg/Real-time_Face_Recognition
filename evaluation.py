def evaluate_model(true_positive_count, false_negative_count, total_count):
    """
    Evaluate model performance for facial recognition using precision, FAR, FRR, recognition rates, and other metrics.

    Args:
        true_count (int): Number of true positive matches (correct predictions).
        total_count (int): Total number of attempts or predictions.

    Returns:
        dict: Dictionary containing various metrics, including recognition rates.
    """
    # Compute precision
    recognition_rate = (true_positive_count / total_count) * 100 if total_count > 0 else 0
    # Compute False Accept Rate (FAR) and False Reject Rate (FRR)
    false_rejects = total_count - true_positive_count  # Assuming binary outcomes

    # FRR: Proportion of false rejects out of total attempts
    frr = false_rejects / total_count if total_count > 0 else 0

    # Average Error and Failure Rate
    fail_rate = 100 - recognition_rate if total_count > 0 else 100

    # Corrected print statement
    print(
        f'"Recognition Rate (%)": {recognition_rate}, '
        f'"FRR": {frr}, '
    )

    return {
        "True Positive Detection": true_positive_count,
        "Total Detection": total_count,
        "False Negative Detection": false_negative_count,
        "Recognition Rate (%)": recognition_rate,
        "FRR": frr
    }
