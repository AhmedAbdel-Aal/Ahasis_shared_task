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
    process_func = partial(process_single_review, prompt_template=prompt_template)

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


def main_args():
    """
    Main function to run the script with command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process sentiment analysis dataframe with LLM"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input dataframe")
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--prompt", "-p", required=True, help="Path to prompt template file"
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.0,
        help="Delay between LLM calls in seconds",
    )

    args = parser.parse_args()

    # Load the dataframe
    df = load_dataframe(args.input)

    # Load the prompt template
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Process the dataframe
    process_dataframe(df, prompt_template, args.output, delay=args.delay)

    logger.info("Processing complete")


def main():
    """
    Main function to run the script.
    """
    # Example usage
    input_file = "data/dev_subset.csv"
    output_dir = "outputs/output_dev_1"
    prompt_template_file = "prompts/p1.py"
    backend = "openai"  # or "deepseek", "mistral", etc.
    model = "gpt-4o-mini"  # or any other model you want to use

    # Load the dataframe
    df = load_dataframe(input_file)

    # Load the prompt template
    with open(prompt_template_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Process the dataframe in parallel
    # process_dataframe_parallel(df, prompt_template, output_dir, max_workers=4, batch_size=10, delay=0.1)
    process_dataframe(
        df, prompt_template, output_dir, delay=0.1, backend=backend, model=model
    )

    logger.info("Processing complete")


if __name__ == "__main__":
    main()
