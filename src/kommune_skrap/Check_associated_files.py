"""Check predictions in data_folder/predictions.csv by opening files one by one
and confirming or correcting the predicted label. If the file is associated with
another file, meaning they are concerning the same dispensation case, consider
together. In the end, add the correct label to the training set.
"""

import re
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

from kommune_skrap.utils import extract_text_from_url, filter_ignored_filenames


def check_associated_files(
    labels_file: Path, all_files_with_links: Path, reassess: bool = False
):
    """Check classifications in predictions file, taking into account cases that are associated."""
    if labels_file.exists():
        labels_df = pd.read_csv(labels_file)
    else:
        labels_df = pd.DataFrame(columns=["Filnavn", "Avgjørelse", "Dato"])
    already_labeled_files = labels_df["Filnavn"].tolist()
    all_files = pd.read_csv(all_files_with_links)
    # Remove files whose name contains any entry from the ignore list
    all_files = filter_ignored_filenames(all_files)

    n_total = len(all_files)
    n_already_labeled = len(already_labeled_files)

    new_labels_data = []
    dates = [datetime.strptime(date, "%d.%m-%Y").date() for date in all_files["Dato"]]
    all_files["Dato"] = dates
    all_files = all_files.sort_values(by="Dato", ascending=False, ignore_index=True)
    done_files = []
    start_time = time.time()
    print("Starting iteration")
    for _, row in all_files.iterrows():
        filename = row["Filnavn"]
        print_progress(done_files, n_already_labeled, n_total, start_time)
        # Check that we have not done this file already
        if filename in done_files:
            continue
        # If we are not reassessing, skip files that are already labeled
        if not reassess and filename in already_labeled_files:
            continue
        # Find associated files
        associated_files = find_associated_files(
            filename=filename, date=row["Dato"], all_files=all_files
        )
        unique_files = get_unique_files(associated_files=associated_files)
        # If there are associated files, ask user which to consider as the final decision
        if len(unique_files) > 1:
            print("\n----------------------")
            for nr, row in enumerate(unique_files["Filnavn"]):
                print(f"{nr}) {row}")
            # filenr = input("Bruk fil nr? 0 () / 1 / 2 / osv...").lower()
            # if filenr == "":
            filenr = 0
            # else:
            #     filenr = int(filenr)
        elif len(unique_files) == 0:
            unique_files = associated_files = pd.DataFrame([row])
            filenr = 0
        else:
            filenr = 0
        usef = unique_files.iloc[filenr]
        # Check if file is already in labeled data
        if usef["Filnavn"] in already_labeled_files and len(unique_files) == 1:
            decision = labels_df.iloc[already_labeled_files.index(usef["Filnavn"])][
                "Avgjørelse"
            ]
            # print(usef.stem, ": ", decision.upper())
        # If not, ask user to label it
        else:
            # Open pdf files for user to see
            for _, row in unique_files[::-1].iterrows():
                webbrowser.open(row["URL"])
            label = input(
                "\nInnvilget (g) / utsatt (u) / lagre (l) / avslått (): ",
            ).lower()
            if label == "l":
                break
            else:
                decision = decision_from_label(label)
                if isinstance(decision, list) and len(decision) != len(unique_files):
                    label = input("Feil antall bokstaver - prøv igjen!: ").lower()
                    decision = decision_from_label(label)
        # Update the labeled data
        i = 0
        for _, row in associated_files.iterrows():
            if row["Filnavn"] not in unique_files["Filnavn"].values:
                _decision = "utsatt"
            else:
                if isinstance(decision, list):
                    _decision = decision[i]
                    i += 1
                else:
                    _decision = decision if all(row == usef) else "utsatt"
            new_labels_data.append((row["Filnavn"], _decision, row["Dato"]))
            done_files.append(row["Filnavn"])
    # Save the updated labeled data
    new_labels_df = pd.DataFrame(
        new_labels_data, columns=["Filnavn", "Avgjørelse", "Dato"]
    )
    if reassess:
        new_file = data_folder / "labels_new.csv"
        if new_file.exists():
            sure = input("File already exists. Overwrite? (y/n): ").lower()
            if sure == "n":
                new_file_name = input("New file name (without path and suffix): ")
                new_file = data_folder / f"{new_file_name}.csv"
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
    filename: str, date: datetime, all_files: pd.DataFrame
) -> pd.DataFrame:
    """Find associated files, given by sharing a numeric code in the filename."""
    associated_files = []
    # Get numeric code from filename by looking for numbers connected with at least one _
    numeric_code = re.findall(r"\d+(?:_\d+)+", filename)
    if numeric_code:
        for code in numeric_code:
            for _, row in all_files.iterrows():
                if code in re.findall(r"\d+(?:_\d+)+", row["Filnavn"]):
                    if row["Filnavn"] not in [a["Filnavn"] for a in associated_files]:
                        # Make sure the file is not too far away in time
                        file_date = row["Dato"]
                        if abs((file_date - date).days) <= 700:
                            associated_files.append(row)
    elif " - " in filename:
        if (
            "Klage" not in filename.split(" - ")[0]
            and "dispensasjon" not in filename.split(" - ")[0].lower()
        ):
            address = filename.split(" - ")[0]
        elif "dispensasjon" in filename.split(" - ")[0].lower():
            address = " - ".join(filename.split(" - ")[1:])
        for _, row in all_files.iterrows():
            if address in row["Filnavn"]:
                associated_files.append(row)
    elif ", " in filename and "Klage" not in filename.split(", ")[0]:
        address = filename.split(", ")[0]
        for _, row in all_files.iterrows():
            if address in row["Filnavn"]:
                associated_files.append(row)
    elif "Dispensasjon til å spille musikk til kl. 02.00" in filename:
        return pd.DataFrame()
    elif "Helgøya, behandling av klage på avslag" in filename:
        return pd.DataFrame()
    else:
        print("\nWarning: No recognizable pattern found in\n", filename)
        pass
    return pd.DataFrame(associated_files)


def print_progress(done_files, n_already_labeled, n_total, start_time):
    """Print progress of labeling."""
    n_done = len(done_files)
    progress = (n_done + n_already_labeled) / n_total * 100
    if n_done > 0:
        eta = (
            (time.time() - start_time)
            / n_done
            * (n_total - n_already_labeled - n_done)
            / 60
        )
    else:
        eta = pd.NA
    print(f"{progress:.2f} %,  ETA: {eta:.1f} min", end="\r")


def get_unique_files(associated_files: pd.DataFrame) -> pd.DataFrame:
    """Find identical files, given by having identical content."""
    hash_list = []
    unique_files = []
    for _, row in associated_files.iterrows():
        url = row["URL"]
        # Get text for file
        text = extract_text_from_url(url)
        # Get hash for text
        text_hash = hash(text)
        if text_hash not in hash_list:
            hash_list.append(text_hash)
            unique_files.append(row)
    return pd.DataFrame(unique_files)


def remove_labeled_files(
    associated_files: list, already_labeled_files: list, done_files: list
) -> list:
    """Remove files that are already labeled."""
    return [
        file
        for file in associated_files
        if Path(file).stem not in already_labeled_files and file not in done_files
    ]


if __name__ == "__main__":
    data_folder = Path("D:/kommune-skrap/data/kristiansand")
    labels_file = data_folder / "labels_2025.csv"
    file_links = data_folder / "file_links_2025.csv"
    check_associated_files(labels_file, file_links)
    # TODO: Deretter: Sjekk om de er i nærheten av sjøen
