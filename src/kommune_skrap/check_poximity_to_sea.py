"""Go through pdf files and assess whether each case is close to the sea or not."""

import webbrowser
from pathlib import Path

import pandas as pd

from kommune_skrap.utils import extract_text_from_url

SEA_PROXIMITY_KEYWORDS = [
    "100 m",
    "100-m",
    "100m",
    "hundre meter",
    "hundre-meter",
    "hundremeter",
    "17.2",
    "17-2",
    "1-8",
    "sjøbod",
    "brygge",
    "vassdrag",
    "50 m",
    "50-m",
    "50m",
    "femti meter",
    "femti-meter",
    "femtimeter",
    "§ 11",
    "§11",
]


def assess_sea_proximity(labels_file, near_sea_file):
    """Check if the cases are close to the sea."""
    files_df = pd.read_csv(labels_file, parse_dates=["Dato"])
    if near_sea_file.exists():
        near_sea_df = pd.read_csv(near_sea_file)
    else:
        near_sea_df = pd.DataFrame(columns=["filename", "near_sea"])
    near_sea_list = []
    files_df = files_df.sort_values(by="Dato", ascending=False)
    # Loop through the files
    for _, row in files_df.iterrows():
        filename = Path(row["Filnavn"])
        if row["Filnavn"] in near_sea_df["filename"].values:
            continue
        url = row["URL"]
        # Extract text from the PDF
        text = extract_text_from_url(url)
        # Replace different types of dashes with a standard dash for easier matching
        text = text.replace(" – ", " - ").lower()
        # Check for key words
        if row["Avgjørelse"] == "utsatt":
            decision = "utsatt"
        elif any(keyword in text for keyword in SEA_PROXIMITY_KEYWORDS):
            # Open pdf file for user to see
            webbrowser.open(url)
            label = input(
                "Ikke sjøen (n) / lagre (l) / nær vassdrag (v) / nær sjøen (): "
            ).lower()
            if label == "n":
                decision = "nei"
            elif label == "l":
                break
            elif label == "v":
                decision = "vassdrag"
            else:
                decision = "ja"
        else:
            decision = "nei"  # row["near_sea"]
        near_sea_list.append((filename, decision))
    # Save the results
    new_df = pd.DataFrame(near_sea_list, columns=["filename", "near_sea"])
    near_sea_df = pd.concat([near_sea_df, new_df], ignore_index=True)
    near_sea_df.to_csv(near_sea_file, index=False)


if __name__ == "__main__":
    data_folder = Path("D:/kommune-skrap/data/kristiansand")
    labels_file = data_folder / "labels_2025.csv"
    near_sea_file = data_folder / "near_sea_2025.csv"
    assess_sea_proximity(labels_file, near_sea_file)
