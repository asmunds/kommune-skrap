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

from kommune_skrap.classify_files import extract_text_from_pdf


def check_associated_files(
    labels_file: Path, prediction_file: Path, reassess: bool = False
):
    """Check classifications in predictions file, taking into account cases that are associated."""
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
        # If we are not reassessing, skip files that are already labeled
        if not reassess and pdf_path.stem in alread_labeled_files:
            continue
        # Check that file exists
        if not pdf_path.exists():
            print(f"Warning: {pdf_path} not found.")
            continue
        # Find associated files
        associated_files = find_associated_files(
            filename=pdf_path.stem, date=row["date"], prediction_df=prediction_df
        )
        associated_files = remove_labeled_files(
            associated_files=associated_files,
            alread_labeled_files=alread_labeled_files,
            done_files=done_files,
        )
        unique_files = get_unique_files(associated_files=associated_files)
        # If there are associated files, ask user which to consider as the final decision
        if len(unique_files) > 1:
            print("----------------------")
            for nr, file in enumerate(unique_files):
                print(f"{nr}) {file}")
            # filenr = input("Bruk fil nr? 0 () / 1 / 2 / osv...").lower()
            # if filenr == "":
            filenr = 0
            # else:
            #     filenr = int(filenr)
        elif len(unique_files) == 0:
            unique_files = [pdf_path]
            filenr = 0
        else:
            filenr = 0
        usef = Path(unique_files[filenr])
        # Check if file is already in labeled data
        if usef.stem in alread_labeled_files and len(unique_files) == 1:
            decision = labels_df.iloc[alread_labeled_files.index(usef.stem)]["label"]
            # print(usef.stem, ": ", decision.upper())
        # If not, ask user to label it
        else:
            # Open pdf files for user to see
            for file in unique_files[::-1]:
                webbrowser.open(Path(file).absolute().as_uri())
            label = input(
                "Innvilget (g) / utsatt (u) / lagre (l) / avslått (): "
            ).lower()
            if label == "l":
                break
            else:
                decision = decision_from_label(label)
                if len(decision) > 1 and len(decision) != len(associated_files):
                    label = input("Feil antall bokstaver - prøv igjen!: ").lower()
                    decision = decision_from_label(label)
        # Update the labeled data
        i = 0
        for file in associated_files:
            if file not in unique_files:
                _decision = "utsatt"
            else:
                if isinstance(decision, list):
                    _decision = decision[i]
                    i += 1
                else:
                    _decision = decision if file == usef else "utsatt"
            new_labels_data.append((file, _decision, Path(file).parts[-2]))
            done_files.append(file)
    # Save the updated labeled data
    new_labels_df = pd.DataFrame(new_labels_data, columns=["filename", "label", "date"])
    if reassess:
        new_file = Path("./data/training_data/labels_new.csv")
        if new_file.exists():
            sure = input("File already exists. Overwrite? (y/n): ").lower()
            if sure == "n":
                new_file_name = input("New file name (without path and suffix): ")
                new_file = Path(f"./data/training_data/{new_file_name}.csv")
        new_labels_df.to_csv(new_file, index=False)
    else:
        labels_df = pd.concat([labels_df, new_labels_df], ignore_index=True)
        labels_df.to_csv(labels_file, index=False)


def decision_from_label(label: str) -> str | list[str]:
    """Return decision from label."""
    if label == "g":
        return "innvilget"
    elif label == "u":
        return "utsatt"
    elif len(label) > 1:
        return [decision_from_label(l) for l in label]
    else:
        return "avslått"


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
        print("\nWarning: No recognizable pattern found in\n", filename)
        pass
    return associated_files


def get_unique_files(associated_files: list) -> list:
    """Find identical files, given by having identical content."""
    hash_list = []
    unique_files = []
    for file in associated_files:
        # Get tect for file
        text = extract_text_from_pdf(Path(file))
        # Get hash for text
        text_hash = hash(text)
        if text_hash not in hash_list:
            hash_list.append(text_hash)
            unique_files.append(file)
    return unique_files


def remove_labeled_files(
    associated_files: list, alread_labeled_files: list, done_files: list
) -> list:
    """Remove files that are already labeled."""
    return [
        file
        for file in associated_files
        if Path(file).stem not in alread_labeled_files and file not in done_files
    ]


if __name__ == "__main__":
    data_folder = Path("./data")
    labels_file = data_folder / "training_data/labels.csv"
    prediction_file = data_folder / "predictions.csv"
    check_associated_files(labels_file, prediction_file)
