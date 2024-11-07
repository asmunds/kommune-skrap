"""
Make a training data set from PDF files and labels, and train a model to classify the text in the PDF files.

Suggestion from Chat GPT, modified to fit my purposes.
"""

import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd


def create_training_set(data_folder: Path, labels_file: Path):
    """Create a training set from PDF files and labels."""
    training_df = pd.read_csv(labels_file, header=0, parse_dates=["date"])
    training_data = [row.tolist() for _, row in training_df.iterrows()]
    label = ""
    for subfolder in data_folder.iterdir():
        if label == "l":
            break
        if subfolder.is_dir():
            print("\n", subfolder.name)
            date = datetime.strptime(subfolder.name, "%d.%m-%Y")
            # if date < datetime.now() - pd.DateOffset(weeks=1):
            #     continue
            for pdf_path in subfolder.iterdir():
                if (
                    pdf_path.exists()
                    and pdf_path.suffix == ".pdf"
                    and pdf_path not in training_df["filename"]
                ):
                    # Open pdf file for user to see
                    webbrowser.open(pdf_path.absolute().as_uri())

                    label = input(
                        "Innvilget (g) / utsatt (u) / lagre (l) / avslått (): "
                    ).lower()
                    if label == "g":
                        decision = "innvilget"
                    elif label == "u":
                        decision = "utsatt"
                    elif label == "l":
                        break
                    else:
                        decision = "avslått"
                    training_data.append((pdf_path, decision, date))
    training_df = pd.DataFrame(training_data, columns=["filename", "label", "date"])
    training_df.to_csv(labels_file, index=False)


if __name__ == "__main__":
    data_folder = Path("./data/training_data")
    labels_file = data_folder / "labels.csv"
    create_training_set(data_folder, labels_file)
