import json

def calculate_metrics(total_tp, total_fp, total_fn, total_tn):
    """
    Calculate evaluation metrics based on aggregated values.
    """
    # Calculate Precision
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    # Calculate Recall
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    # Calculate F1 Score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate False Rejection Rate (FRR)
    frr = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    # Calculate False Acceptance Rate (FAR)
    far = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "FRR (False Rejection Rate)": frr,
        "FAR (False Acceptance Rate)": far
    }

def main(file_path):
    """
    Main function to process the execution_model_summary.txt and calculate metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                # Preprocess the line to replace single quotes with double quotes
                valid_json_line = line.strip().replace("'", "\"")
                # Parse the JSON string into a Python dictionary
                summary = json.loads(valid_json_line)
                # Aggregate values
                total_tp += summary.get('True Positive', 0)
                total_fp += summary.get('False Positive', 0)
                total_fn += summary.get('False Negative', 0)
                total_tn += summary.get('True Negative', 0)

    # Calculate metrics for the aggregated values
    metrics = calculate_metrics(total_tp, total_fp, total_fn, total_tn)

    # Print aggregated values and metrics
    print("Aggregated Values:")
    print(f"True Positive (TP): {total_tp}")
    print(f"False Positive (FP): {total_fp}")
    print(f"False Negative (FN): {total_fn}")
    print(f"True Negative (TN): {total_tn}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    # Replace with your file path
    file_path = "execution_model_summary.txt"
    main(file_path)
