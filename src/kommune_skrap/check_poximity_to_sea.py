"""Go through pdf files and assess whether each case is close to the sea or not."""

import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

from kommune_skrap.classify_files import extract_text_from_pdf


def assess_sea_proximity(labels_file, near_sea_file):
    """Check if the cases are close to the sea."""
    files_df = pd.read_csv(labels_file)
    near_sea_df = pd.read_csv(near_sea_file)
    near_sea_list = []
    dates = [
        datetime.strptime(Path(pdf_path).parts[-2], "%d.%m-%Y").date()
        for pdf_path in files_df["filename"]
    ]
    files_df["date"] = dates
    files_df = files_df.sort_values(by="date", ascending=False)
    # Loop through the files
    for i, row in files_df.iterrows():
        pdf_path = Path(row["filename"])
        if row["filename"] in near_sea_df["filename"].values:
            continue
        if pdf_path.exists():
            # Extract text from the PDF
            text = extract_text_from_pdf(pdf_path)
            # Check for key words
            if (
                "100 m" in text
                or "100-m" in text
                or "100m" in text
                or "hundre meter" in text
                or "hundre-meter" in text
                or "17.2" in text
                or "17-2" in text
                or "1-8" in text
            ):
                # Open pdf file for user to see
                webbrowser.open(pdf_path.absolute().as_uri())
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
            near_sea_list.append((pdf_path, decision))
        else:
            print(f"Warning: {pdf_path} not found.")
    # Save the results
    new_df = pd.DataFrame(near_sea_list, columns=["filename", "near_sea"])
    near_sea_df = pd.concat([near_sea_df, new_df], ignore_index=True)
    near_sea_df.to_csv(near_sea_file, index=False)


if __name__ == "__main__":
    data_folder = Path("./data")
    prediction_file = data_folder / "predictions.csv"
    near_sea_file = data_folder / "training_data" / "near_sea.csv"
    assess_sea_proximity(prediction_file, near_sea_file)
