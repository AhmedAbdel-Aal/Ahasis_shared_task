import re
from typing import Dict, List, Any


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_analysis_from_xml(text: str) -> Dict[str, Any]:
    """
    Extracts structured analysis from XML response using regex pattern matching.

    Args:
        text: The XML response text

    Returns:
        Dictionary containing parsed analysis with structure:
        {
            'dialect': str,
            'aspects': List[Dict],
            'overall_sentiment': str,
            'overall_justification': str
        }
    """
    # Extract dialect
    dialect = extract_xml(text, "dialect")

    # Extract overall sentiment and justification
    overall_sentiment = extract_xml(text, "overall_sentiment")
    overall_justification = extract_xml(text, "overall_justification")

    # Extract all aspects
    aspects = []
    aspects_text = extract_xml(text, "aspects")

    if aspects_text:
        # Find all individual aspect blocks
        aspect_blocks = re.findall(r"<aspect>(.*?)</aspect>", aspects_text, re.DOTALL)

        for block in aspect_blocks:
            name = extract_xml(block, "name")
            sentiment = extract_xml(block, "sentiment")
            justification = extract_xml(block, "justification")

            if name and sentiment:  # Only add if we have required fields
                aspects.append(
                    {
                        "name": name,
                        "sentiment": sentiment,
                        "justification": justification,
                    }
                )

    return {
        "dialect": dialect,
        "aspects": aspects,
        "overall_sentiment": overall_sentiment,
        "overall_justification": overall_justification,
    }
