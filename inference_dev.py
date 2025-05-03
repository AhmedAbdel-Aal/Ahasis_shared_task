"""
Process sentiment analysis dataframe and call LLM for each review.
This script loads a dataframe, processes each row one by one, calls an LLM,
and saves the results in individual files.
"""

import os
import pandas as pd
import time
from typing import Dict, List, Any, Optional
import argparse
from tqdm import tqdm
import logging
import json
import concurrent.futures
from functools import partial
import dotenv
import datetime


from xml_parser import extract_analysis_from_xml

from utils import load_dataframe, save_json, load_json, format_prompt
from llm import llm_call


dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sentiment_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def check_output_directory(output_dir: str) -> None:
    """
    Check if output directory exists, create if it doesn't.

    Args:
        output_dir: Path to the output directory
    """
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)


def get_completed_ids(output_dir: str) -> List[str]:
    """
    Get list of already processed IDs based on existing files.
    """
    completed_ids = []

    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            # Remove file extension to get the ID
            file_id = os.path.splitext(filename)[0]
            completed_ids.append(file_id)

    logger.info(f"Found {len(completed_ids)} already processed reviews")
    return completed_ids


def process_single_review(
    row: pd.Series,
    prompt_template: str,
    backend: str = "openai",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Process a single review by calling the LLM.

    Args:
        row: DataFrame row containing the review data
        prompt_template: Template for the prompt to send to the LLM

    Returns:
        Dictionary containing the LLM response and structured data
    """
    try:
        # Extract review text from the row
        review_text = row["Text"]

        # Format the prompt with the review text
        formatted_prompt = format_prompt(prompt_template, review=review_text)

        # Call the LLM and get response
        llm_response = llm_call(formatted_prompt, backend=backend, model=model)

        # Extract structured data from XML response
        structured_data = extract_analysis_from_xml(llm_response)

        # Add metadata to the result
        result = {
            "id": row["ID"],
            "original_text": review_text,
            "dialect": row["Dialect"],
            "true_sentiment": row["Sentiment"],
            "llm_response": llm_response,
            "structured_data": structured_data,
        }

        return result

    except Exception as e:
        logger.error(f"Error processing review ID {row['ID']}: {e}")
        return None


def save_result(result: Dict[str, Any], output_dir: str) -> None:
    """
    Save the result to a JSON file.

    Args:
        result: Dictionary containing the result data
        output_dir: Directory to save the file
    """
    file_id = str(result["id"])
    file_path = os.path.join(output_dir, f"{file_id}.json")
    save_json(file_path, result)


def process_dataframe(
    df: pd.DataFrame,
    prompt_template: str,
    output_dir: str,
    delay: float = 1.0,
    backend: str = "openai",
    model: str = "gpt-4o-mini",
) -> None:
    """
    Process the dataframe and call LLM for each review.

    Args:
        df: DataFrame containing the reviews
        prompt_template: Template for the prompt to send to the LLM
        output_dir: Directory to save the results
        batch_size: Number of reviews to process before saving
        delay: Delay in seconds between LLM calls to avoid rate limits
    """
    # Make sure output directory exists
    check_output_directory(output_dir)

    # Get list of already processed IDs
    completed_ids = get_completed_ids(output_dir)

    # Filter out already processed rows
    df_to_process = df[~df["ID"].astype(str).isin(completed_ids)]

    if len(df_to_process) == 0:
        logger.info("All reviews have already been processed")
        return

    logger.info(f"Processing {len(df_to_process)} out of {len(df)} reviews")

    # Process each row
    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        # Process the review
        result = process_single_review(row, prompt_template, backend, model)

        # Save the result
        if result:
            save_result(result, output_dir)

        # Delay to avoid hitting API rate limits
        # time.sleep(delay)


def process_dataframe_parallel(
    df: pd.DataFrame,
    prompt_template: str,
    output_dir: str,
    max_workers: int = 4,
    batch_size: int = 10,
    delay: float = 0.1,
    backend: str = "openai",
    model: str = "gpt-4o",
) -> None:
    """
    Process the dataframe in parallel by calling LLM for multiple reviews simultaneously.

    Args:
        df: DataFrame containing the reviews
        prompt_template: Template for the prompt to send to the LLM
        output_dir: Directory to save the results
        max_workers: Maximum number of parallel workers
        batch_size: Number of reviews to process before saving
        delay: Delay in seconds between batches to avoid rate limits
    """
    # Make sure output directory exists
    check_output_directory(output_dir)

    # Get list of already processed IDs
    completed_ids = get_completed_ids(output_dir)

    # Filter out already processed rows
    df_to_process = df[~df["ID"].astype(str).isin(completed_ids)]

    if len(df_to_process) == 0:
        logger.info("All reviews have already been processed")
        return

    logger.info(
        f"Processing {len(df_to_process)} out of {len(df)} reviews in parallel with {max_workers} workers"
    )

    # Create a partial function with fixed arguments
    process_func = partial(
        process_single_review,
        prompt_template=prompt_template,
        backend=backend,
        model=model,
    )

    # Process in batches to better control memory and rate limits
    batches = [
        df_to_process[i : i + batch_size]
        for i in range(0, len(df_to_process), batch_size)
    ]

    with tqdm(total=len(df_to_process)) as pbar:
        for batch in batches:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all tasks in the current batch
                futures = {
                    executor.submit(process_func, row): row["ID"]
                    for _, row in batch.iterrows()
                }

                # Process completed futures
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        save_result(result, output_dir)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(
                            f"Error processing review ID {futures[future]}: {e}"
                        )
                        pbar.update(1)

            # Delay between batches to avoid hitting rate limits
            time.sleep(delay)


def create_experiment_directory(experiment_num: int) -> str:
    """
    Create experiment directory structure and return path to metadata file.
    """
    experiment_dir = f"outputs/experiment_{experiment_num}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/output_dev", exist_ok=True)
    os.makedirs(f"{experiment_dir}/output_test", exist_ok=True)
    return experiment_dir


def update_metadata(metadata_path: str, run_data: dict) -> None:
    """
    Update experiment metadata file with new run information.
    """
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "experiment_number": run_data["experiment_number"],
            "created_at": run_data["start_time"],
            "prompt_file": run_data["prompt_file"],
            "backend": run_data["backend"],
            "model": run_data["model"],
            "directory_structure": {
                "output_dev": "output_dev",
                "output_test": "output_test",
            },
            "runs": [],
        }

    metadata["runs"].append(run_data)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    """
    Main function to run the script with experiment tracking.
    """
    # Experiment configuration
    experiment_num = 2  # Could be auto-incremented based on existing experiments
    prompt_template_file = "prompts/p1.py"
    backend = "openai"
    model = "gpt-4o"

    # Create experiment directory
    experiment_dir = create_experiment_directory(experiment_num)
    metadata_path = f"{experiment_dir}/metadata.json"

    metadata = {
        "experiment_number": experiment_num,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_file": prompt_template_file,
        "backend": backend,
        "model": model,
        "directory_structure": {
            "output_dev": "output_dev",
            "output_test": "output_test",
        },
        "runs": [],
    }
    # Prepare run data
    run_data = {
        "run_id": len(metadata["runs"]) + 1 if os.path.exists(metadata_path) else 1,
        "experiment_number": experiment_num,
        "prompt_file": prompt_template_file,
        "backend": backend,
        "model": model,
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "processing": "dev",
        "total_files": 0,
        "processed_files": 0,
        "changes": [],
    }

    # Load data and prompt
    input_file = "data/dev_subset.csv"
    output_dir = f"{experiment_dir}/output_dev"

    df = load_dataframe(input_file)  # [:5]
    run_data["total_files"] = len(df)

    with open(prompt_template_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Process data
    # process_dataframe( df, prompt_template, output_dir, delay=0.1, backend=backend, model=model)
    process_dataframe_parallel(
        df,
        prompt_template,
        output_dir,
        max_workers=8,
        batch_size=10,
        delay=0.1,
        backend=backend,
        model=model,
    )

    # Update metadata with completion info
    run_data.update(
        {
            "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed",
            "processed_files": len(df),
            "changes": ["Processed initial dataset"],
        }
    )
    update_metadata(metadata_path, run_data)

    logger.info(
        f"Experiment {experiment_num} completed. Results saved to {experiment_dir}"
    )


if __name__ == "__main__":
    main()
