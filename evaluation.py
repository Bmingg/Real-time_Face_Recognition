def evaluate_model(true_count, total_count):
    """
    Evaluate model performance for facial recognition using precision, FAR, FRR, recognition rates, and other metrics.

    Args:
        true_count (int): Number of true positive matches (correct predictions).
        total_count (int): Total number of attempts or predictions.

    Returns:
        dict: Dictionary containing various metrics, including recognition rates.
    """
    # Compute precision
    precision = true_count / total_count if total_count > 0 else 0

    # Compute False Accept Rate (FAR) and False Reject Rate (FRR)
    false_accepts = total_count - true_count  # Number of false accepts
    false_rejects = total_count - true_count  # Assuming binary outcomes

    # FAR: Proportion of false accepts out of total attempts
    far = false_accepts / total_count if total_count > 0 else 0

    # FRR: Proportion of false rejects out of total attempts
    frr = false_rejects / total_count if total_count > 0 else 0

    # Average Error and Failure Rate
    avg_error = (far + frr) / 2
    fail_rate = (1 - precision) * 100 if total_count > 0 else 100

    # Recognition Rate (proportion of correct matches)
    recognition_rate = (true_count / total_count) * 100 if total_count > 0 else 0

    # Corrected print statement
    print(
        f'"Recognition Rate (%)": {recognition_rate}, '
        f'"Precision": {precision}, '
        f'"FAR": {far}, '
        f'"FRR": {frr}, '
        f'"Average Error": {avg_error}, '
        f'"Fail Rate (%)": {fail_rate}'
    )

    return {
        "True Positive Detection": true_count,
        "Total Detection": total_count,
        "Recognition Rate (%)": recognition_rate,
        "Precision": precision,
        "FAR": far,
        "FRR": frr,
        "Average Error": avg_error,
        "Fail Rate (%)": fail_rate,
    }
