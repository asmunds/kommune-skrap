"""Check predictions in data_folder/predictions.csv by opening files one by one
and confirming or correcting the predicted label. If the file is associated with
another file, meaning they are concerning the same dispensation case, consider
together. In the end, add the correct label to the training set.
"""

import re
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd


def check_associated_files(labels_file: Path, prediction_file: Path):
    """Check classifications in predictions file."""
    labels_df = pd.read_csv(labels_file)
    alread_labeled_files = [Path(path).stem for path in labels_df["filename"]]
    prediction_df = pd.read_csv(prediction_file)
    new_labels_data = []
    dates = [
        datetime.strptime(Path(pdf_path).parts[-2], "%d.%m-%Y").date()
        for pdf_path in prediction_df["filename"]
    ]
    prediction_df["date"] = dates
    prediction_df = prediction_df.sort_values(by="date", ascending=False)
    done_files = []
    for i, row in prediction_df.iterrows():
        pdf_path = Path(row["filename"])
        # Check that we have not done this file already
        if str(pdf_path) in done_files:
            continue
        # Check that file exists
        if not pdf_path.exists():
            print(f"Warning: {pdf_path} not found.")
            continue
        # Find associated files
        associated_files = find_associated_files(
            filename=pdf_path.stem, date=row["date"], prediction_df=prediction_df
        )
        # If there are associated files, ask user which to consider as the final decision
        if len(associated_files) > 1:
            print("----------------------")
            for nr, file in enumerate(associated_files):
                print(f"{nr}) {file}")
            # filenr = input("Bruk fil nr? 0 () / 1 / 2 / osv...").lower()
            # if filenr == "":
            filenr = 0
            # else:
            #     filenr = int(filenr)
        elif len(associated_files) == 0:
            associated_files = [pdf_path]
            filenr = 0
        else:
            filenr = 0
        usef = Path(associated_files[filenr])
        # Check if file is already in labeled data
        if usef.stem in alread_labeled_files:
            decision = labels_df.iloc[alread_labeled_files.index(usef.stem)]["label"]
            # print(usef.stem, ": ", decision.upper())
        # If not, ask user to label it
        else:
            # Open pdf file for user to see
            webbrowser.open(usef.absolute().as_uri())
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
        # Update the labeled data
        for file in associated_files:
            _decision = decision if file == usef else "utsatt"
            new_labels_data.append((file, _decision, Path(file).parts[-2]))
            done_files.append(file)
    # Save the updated labeled data
    new_labels_df = pd.DataFrame(new_labels_data, columns=["filename", "label", "date"])
    new_labels_df.to_csv(Path("./data/training_data/labels_new.csv"), index=False)


def find_associated_files(
    filename: str, date: datetime, prediction_df: pd.DataFrame
) -> list:
    """Find associated files, given by sharing a numeric code in the filename."""
    associated_files = []
    # Get numeric code from filename by looking for numbers connected with at least one _
    numeric_code = re.findall(r"\d+(?:_\d+)+", filename)
    if numeric_code:
        for code in numeric_code:
            for i, row in prediction_df.iterrows():
                if code in re.findall(r"\d+(?:_\d+)+", row["filename"]):
                    if row["filename"] not in associated_files:
                        # Make sure the file is not too far away in time
                        file_date = datetime.strptime(
                            Path(row["filename"]).parts[-2], "%d.%m-%Y"
                        ).date()
                        if abs((file_date - date).days) <= 700:
                            associated_files.append(row["filename"])
    elif " - " in filename:
        address = filename.split(" - ")[0]
        for i, row in prediction_df.iterrows():
            if address in row["filename"]:
                associated_files.append(row["filename"])
    elif ", " in filename:
        address = filename.split(", ")[0]
        for i, row in prediction_df.iterrows():
            if address in row["filename"]:
                associated_files.append(row["filename"])
    elif "Dispensasjon til å spille musikk til kl. 02.00" in filename:
        return []
    elif "Helgøya, behandling av klage på avslag" in filename:
        return []
    else:
        print("\nWarning: No numeric code found in\n", filename)
        pass
    return associated_files


if __name__ == "__main__":
    data_folder = Path("./data")
    labels_file = data_folder / "training_data/labels.csv"
    prediction_file = data_folder / "predictions.csv"
    check_associated_files(labels_file, prediction_file)
