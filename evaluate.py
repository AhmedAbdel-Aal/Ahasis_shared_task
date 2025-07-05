"""
AHASIS Shared Task 2025 - Sentiment Analysis Evaluation
This script provides functions to evaluate sentiment analysis models for hotel reviews
in Saudi and Darija dialects with multiple classification metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional

from utils import load_json, save_json


def compute_basic_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """
    Compute basic classification metrics for sentiment analysis.

    Args:
        y_true: List of ground truth sentiment labels (0.0: negative, 1.0: neutral, 2.0: positive)
        y_pred: List of predicted sentiment labels (0.0: negative, 1.0: neutral, 2.0: positive)

    Returns:
        Dictionary containing accuracy, macro and weighted precision, recall, and F1 scores
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    return metrics


def compute_class_metrics(
    y_true: List[float], y_pred: List[float]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics for each sentiment category.

    Args:
        y_true: List of ground truth sentiment labels
        y_pred: List of predicted sentiment labels

    Returns:
        Dictionary with per-class precision, recall, and F1 scores
    """
    # Get classification report as dictionary
    report = classification_report(y_true, y_pred, output_dict=True)
    print(report)

    # Extract metrics for each class (negative: 0.0, neutral: 1.0, positive: 2.0)
    class_metrics = {
        "negative": report.get("0.0", {}),
        "neutral": report.get("1.0", {}),
        "positive": report.get("2.0", {}),
    }

    return class_metrics


def compute_advanced_metrics(
    y_true: List[float], y_pred: List[float]
) -> Dict[str, float]:
    """
    Compute advanced evaluation metrics for sentiment analysis.

    Args:
        y_true: List of ground truth sentiment labels
        y_pred: List of predicted sentiment labels

    Returns:
        Dictionary with Cohen's Kappa and Matthews Correlation Coefficient
    """
    metrics = {
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
    }

    return metrics


def plot_confusion_matrix(
    y_true: List[float],
    y_pred: List[float],
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and optionally save confusion matrix.

    Args:
        y_true: List of ground truth sentiment labels
        y_pred: List of predicted sentiment labels
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the confusion matrix plot (if None, plot is displayed)
    """
    labels = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def evaluate_by_dialect(
    y_true: List[float], y_pred: List[float], dialects: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate performance separated by dialect.

    Args:
        y_true: List of ground truth sentiment labels
        y_pred: List of predicted sentiment labels
        dialects: List of dialect labels for each example ('Saudi' or 'Darija')

    Returns:
        Dictionary with metrics for each dialect
    """
    # Create dataframe for easier filtering
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "dialect": dialects})

    dialect_metrics = {}

    # Calculate metrics for each dialect
    for dialect in df["dialect"].unique():
        dialect_df = df[df["dialect"] == dialect]
        dialect_y_true = dialect_df["y_true"].tolist()
        dialect_y_pred = dialect_df["y_pred"].tolist()

        # Skip if there are no examples for this dialect
        if len(dialect_y_true) == 0:
            continue

        metrics = compute_basic_metrics(dialect_y_true, dialect_y_pred)
        metrics.update(compute_advanced_metrics(dialect_y_true, dialect_y_pred))
        dialect_metrics[dialect] = metrics

    return dialect_metrics


def evaluate_model(
    y_true: List[float],
    y_pred: List[float],
    dialects: Optional[List[str]] = None,
    save_cm: bool = False,
    cm_save_path: str = "confusion_matrix.png",
) -> Dict:
    """
    Complete model evaluation including all metrics and optionally dialect-specific analysis.

    Args:
        y_true: List of ground truth sentiment labels
        y_pred: List of predicted sentiment labels
        dialects: Optional list of dialect labels for each example
        save_cm: Whether to save the confusion matrix plot
        cm_save_path: Path to save the confusion matrix plot if save_cm is True

    Returns:
        Dictionary with all evaluation metrics
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError("Length of ground truth and predictions must match")

    if dialects is not None and len(dialects) != len(y_true):
        raise ValueError("Length of dialects must match ground truth and predictions")

    # Calculate all metrics
    evaluation_results = {
        "overall": compute_basic_metrics(y_true, y_pred),
        "per_class": compute_class_metrics(y_true, y_pred),
        # "advanced": compute_advanced_metrics(y_true, y_pred),
    }

    # Add dialect-specific evaluation if dialects are provided
    if dialects is not None:
        evaluation_results["by_dialect"] = evaluate_by_dialect(y_true, y_pred, dialects)

    # Generate confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, normalize=True, save_path=cm_save_path if save_cm else None
    )

    return evaluation_results


def print_evaluation_report(evaluation_results: Dict) -> None:
    """
    Print formatted evaluation report with all metrics.

    Args:
        evaluation_results: Dictionary with evaluation metrics from evaluate_model
    """
    print("=" * 50)
    print("SENTIMENT ANALYSIS EVALUATION REPORT")
    print("=" * 50)

    # Overall metrics
    print("\n--- OVERALL METRICS ---")
    overall = evaluation_results["overall"]
    print(f"Accuracy: {overall['accuracy']:.4f}")
    print(f"Macro F1: {overall['f1_macro']:.4f}")
    print(f"Weighted F1: {overall['f1_weighted']:.4f}")

    # Per-class metrics
    print("\n--- PER-CLASS METRICS ---")
    per_class = evaluation_results["per_class"]
    for cls, metrics in per_class.items():
        print(f"\n{cls.upper()}:")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  F1-score: {metrics.get('f1-score', 0):.4f}")
        print(f"  Support: {metrics.get('support', 0)}")

    # Advanced metrics
    # print("\n--- ADVANCED METRICS ---")
    # advanced = evaluation_results["advanced"]
    # print(f"Cohen's Kappa: {advanced['cohen_kappa']:.4f}")
    # print(f"Matthews Correlation Coefficient: {advanced['matthews_corrcoef']:.4f}")

    # Dialect-specific metrics if available
    if "by_dialect" in evaluation_results:
        print("\n--- DIALECT-SPECIFIC METRICS ---")
        by_dialect = evaluation_results["by_dialect"]
        for dialect, metrics in by_dialect.items():
            print(f"\n{dialect.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['f1_macro']:.4f}")
            print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")

    print("\n" + "=" * 50)


def load_predicted_results(output_dir: str):
    """
    Load ground truth from dev data and predictions from experiment output files.
    """
    try:
        # Load predictions from experiment output files
        ids = []
        y_pred = []
        dialects_pred = []

        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                file_id = filename.split(".")[0]
                result = load_json(os.path.join(output_dir, filename))

                ids.append(file_id)
                y_pred.append(result["structured_data"]["overall_sentiment"])
                dialects_pred.append(result["structured_data"]["dialect"])
        # transform to dataframe
        df_pred = pd.DataFrame(
            {"ID": ids, "Sentiment_pred": y_pred, "dialect_pred": dialects_pred}
        )

        return df_pred

    except Exception as e:
        print(f"Error loading files: {e}")
        return None


def align_evaluation_data(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> tuple:
    # Ensure ID columns are the same type (convert both to string)
    df_true["ID"] = df_true["ID"].astype(float).astype(str)
    df_pred["ID"] = df_pred["ID"].astype(float).astype(str)

    # Merge dataframes on ID (inner join to only keep IDs present in both)
    merged = pd.merge(
        df_true[["ID", "Sentiment", "Dialect"]],
        df_pred[["ID", "Sentiment_pred", "dialect_pred"]],
        on="ID",
        how="inner",
    )

    print(f"Number of samples in merged data: {len(merged)}")

    # Convert to lists in consistent order
    y_true = merged["Sentiment"].tolist()
    y_pred = merged["Sentiment_pred"].tolist()
    dialect_true = merged["Dialect"].tolist()
    dialect_pred = merged["dialect_pred"].tolist()
    ids = merged["ID"].tolist()

    return y_true, y_pred, dialect_true, dialect_pred, ids


def main():
    """
    Main function to run the evaluation script.
    """
    # Example usage
    experiment_id = 3
    experiment_dir = f"./outputs/experiment_{str(experiment_id)}/output_dev"  # Path to experiment directory
    dev_data_file = "data/dev_subset.csv"  # Path to dev data

    # Load results
    df_pred = load_predicted_results(experiment_dir)
    df_true = pd.read_csv(dev_data_file)

    # Align data
    y_true, y_pred, dialect_true, dialect_pred, ids = align_evaluation_data(
        df_true, df_pred
    )
    evaluation_results = evaluate_model(
        y_true,
        y_pred,
        dialects=dialect_true,
        save_cm=True,
        cm_save_path=f"./outputs/experiment_{str(experiment_id)}/confusion_matrix_experiment_{experiment_id}.png",
    )
    # Print evaluation report
    print_evaluation_report(evaluation_results)
    # Save evaluation results to JSON file
    output_dir = f"./outputs/experiment_{str(experiment_id)}/metrics.json"
    save_json(output_dir, evaluation_results)


if __name__ == "__main__":
    main()
