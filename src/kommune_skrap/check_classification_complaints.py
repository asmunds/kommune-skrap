"""Check prediction in data_folder/predictions.csv by opening files one by one
and confirming or correcting the predicted label. Add the correct label to the
training set and retrain the model.
"""

import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd


def check_classifications(labels_file):
    """Check classifications in labels_file."""
    training_df = pd.read_csv(labels_file)
    dates = [
        datetime.strptime(Path(pdf_path).parts[-2], "%d.%m-%Y").date()
        for pdf_path in training_df["filename"]
    ]
    training_df["date"] = dates
    training_df = training_df.sort_values(by="date", ascending=False)
    for i, row in training_df.iterrows():
        pdf_path = Path(row["filename"])
        if pdf_path.exists():
            if "klage" in pdf_path.stem.lower():
                # Open pdf file for user to see
                webbrowser.open(pdf_path.absolute().as_uri())
                label = input(
                    "Innvilget (g) / utsatt (u) / lagre (l) / avslått (): "
                ).lower()
                if label == "g":
                    training_df.loc[i, "label"] = "innvilget"
                elif label == "u":
                    training_df.loc[i, "label"] = "utsatt"
                elif label == "l":
                    break
                else:
                    training_df.loc[i, "label"] = "avslått"
        else:
            print(f"Warning: {pdf_path} not found.")
    training_df.to_csv(labels_file, index=False)


if __name__ == "__main__":
    data_folder = Path("./data")
    labels_file = data_folder / "training_data/labels.csv"
    check_classifications(labels_file)
