"""Common utility functions shared across the kommune_skrap package."""

import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PyPDF2 import PdfReader

_IGNORE_FILENAMES_CSV = Path(__file__).parent / "ignore_filenames.csv"


def load_ignore_filenames(csv_path: Path = _IGNORE_FILENAMES_CSV) -> list[str]:
    """Load the list of filenames to ignore from a CSV file.

    The CSV is expected to have a single column with one filename per row and
    no header.

    Args:
        csv_path: Path to the CSV file. Defaults to the bundled
            ``ignore_filenames.csv`` inside the package.

    Returns:
        A list of filename strings to ignore.
    """
    return pd.read_csv(csv_path, header=None, encoding="utf-8").iloc[:, 0].tolist()


def filter_ignored_filenames(
    df: pd.DataFrame,
    col: str = "Filnavn",
    ignore_filenames: list[str] | None = None,
) -> pd.DataFrame:
    """Remove rows where *col* contains any entry from *ignore_filenames*.

    The check is a case-sensitive substring match, so a single ignore entry
    can match multiple rows (e.g. ``"Møtedokumenter"`` removes every filename
    that includes that word).

    Args:
        df: DataFrame to filter.
        col: Column name to check against the ignore list.
        ignore_filenames: List of substrings to filter out. If *None*, the
            bundled ``ignore_filenames.csv`` is loaded automatically.

    Returns:
        Filtered DataFrame with matching rows removed.
    """
    if ignore_filenames is None:
        ignore_filenames = load_ignore_filenames()
    pattern = "|".join(ignore_filenames)
    mask = df[col].str.contains(pattern, na=False)
    return df[~mask].reset_index(drop=True)


def clean_text(text: str) -> str:
    """Remove everything but normal text, punctuation, and Nordic letters from a string."""
    # Remove common formatting operators
    text = re.sub(r"\n|\t|\r", " ", text)
    # Remove everything but normal text, punctuation, and Nordic letters
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:æøåÆØÅ]", "", text)
    # Remove excess spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_url(url: str | Path) -> str:
    """Extract and clean text from a pdf at the given URL or file path.

    Args:
        url: URL string or Path to the PDF file.

    Returns:
        Cleaned text extracted from the PDF, or empty string on failure.
    """
    try:
        response = requests.get(str(url))
        if response.status_code == 200:
            pdf_file = BytesIO(response.content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            print(f"Failed to retrieve text from {url}")
            return ""
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return ""
    return clean_text(text)


def identify_keyword(*, section_title: str, keywords: list[str]) -> str | None:
    """Identify which keyword is present in the section title.

    Args:
        section_title: The title of the section.
        keywords: List of keywords to search for.

    Returns:
        The first keyword found in the section title (case-insensitive), or None.
    """
    for keyword in keywords:
        if keyword in section_title.lower():
            return keyword
    return None


def download_pdf(*, url: str, download_dir: Path, new_filename: str) -> None:
    """Download a PDF file from the given URL to the specified directory.

    Args:
        url: URL to the PDF file.
        download_dir: Directory to save the PDF file.
        new_filename: Filename to use (without the .pdf extension).
    """
    response = requests.get(url)
    if response.status_code == 200:
        filepath = download_dir / (new_filename + ".pdf")
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {download_dir}/{new_filename}.pdf")
    else:
        print(f"Failed to download: {new_filename}.pdf")
