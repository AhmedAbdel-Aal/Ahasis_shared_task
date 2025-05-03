import os
import json
import pandas as pd
from typing import List, Dict


def collect_sentiments(experiment_dir: str) -> Dict[str, str]:
    """
    Collect overall_sentiment from all JSON files in experiment output directory.

    Args:
        experiment_dir: Path to experiment's output_test directory

    Returns:
        Dictionary mapping IDs to their overall_sentiment values
    """
    sentiments = {}
    for filename in os.listdir(experiment_dir):
        if filename.endswith(".json"):
            file_id = int(filename.split(".")[0])
            with open(os.path.join(experiment_dir, filename), "r") as f:
                data = json.load(f)
                sentiments[file_id] = data["structured_data"]["overall_sentiment"]
    return sentiments


def update_test_file(
    test_path: str, sentiments: Dict[str, str], output_path: str
) -> None:
    """
    Update test CSV file with sentiment predictions and save to output path.

    Args:
        test_path: Path to original test.csv
        sentiments: Dictionary mapping IDs to sentiment predictions
        output_path: Where to save the updated CSV
    """
    df = pd.read_csv(test_path)

    # Keep ID as float type (remove the astype(str) conversion)

    # Add sentiment predictions column
    df["Sentiment"] = df["ID"].map(sentiments)

    # Save updated file
    df.to_csv(output_path, index=False)


def main(experiment_num: int):
    """
    Main function to prepare submission file for a specific experiment.

    Args:
        experiment_num: Experiment number to process
    """
    # Define paths
    experiment_dir = f"outputs/experiment_{experiment_num}/output_test"
    test_path = "data/test.csv"
    output_path = f"outputs/experiment_{experiment_num}/predictions.csv"

    # Process files
    sentiments = collect_sentiments(experiment_dir)
    update_test_file(test_path, sentiments, output_path)

    print(f"Successfully updated test file with predictions at: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=int, required=True, help="Experiment number to process"
    )
    args = parser.parse_args()

    main(args.experiment)


# python prepare_submission.py --experiment 1
