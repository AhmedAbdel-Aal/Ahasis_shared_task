import json
import pandas as pd


def load_json(file_path: str):
    """Load and return data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json_old(file_path: str, data: any):
    """Save data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_json(file_path: str, data: any):
    """Save data to a JSON file while preserving Arabic text formatting.

    Args:
        file_path: Path to the output JSON file
        data: Data to be saved (dict, list, etc.)
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_txt(file_path: str):
    """Load and return contents of a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def save_txt(file_path: str, data: str):
    """Save data to a text file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load dataframe from file based on file extension.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".tsv"):
        return pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with variables."""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required prompt variable: {e}")
