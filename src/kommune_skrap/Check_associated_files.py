"""Check predictions in data_folder/predictions.csv by opening files one by one
and confirming or correcting the predicted label. If the file is associated with
another file, meaning they are concerning the same dispensation case, consider
together. In the end, add the correct label to the training set.
"""

import time
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

from kommune_skrap.utils import extract_text_from_url, filter_ignored_filenames, find_associated_files


def check_associated_files(
    labels_file: Path, all_files_with_links: Path, reassess: bool = False
):
    """Check classifications in predictions file, taking into account cases that are associated."""
    if labels_file.exists():
        labels_df = pd.read_csv(labels_file)
    else:
        labels_df = pd.DataFrame(
            columns=["Filnavn", "Avgjørelse", "Dato", "URL", "Hash"]
        )
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
        if not associated_files.empty:
            associated_files_hash = hash(tuple(associated_files["Filnavn"]))
        # Get unique files among the associated files by looking at the content
        unique_files = get_unique_files(associated_files=associated_files)
        # If there are no associated files, we just need to label the file in consideration
        if len(unique_files) == 0:
            unique_files = associated_files = pd.DataFrame([row])
        # List files in consideration
        print("\n----------------------")
        for nr, row in enumerate(unique_files["Filnavn"]):
            print(f"{nr}) {row}")
        # Use the first file as the main file to label (the others will be labeled "utsatt" if not explicitly labeled otherwise)
        usef = unique_files.iloc[0]
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
            new_labels_data.append(
                (
                    row["Filnavn"],
                    _decision,
                    row["Dato"],
                    row["URL"],
                    associated_files_hash,
                )
            )
            done_files.append(row["Filnavn"])
    # Save the updated labeled data
    new_labels_df = pd.DataFrame(
        new_labels_data, columns=["Filnavn", "Avgjørelse", "Dato", "URL", "Hash"]
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

    def single_decision(single_label: str) -> str:
        if single_label == "g":
            return "innvilget"
        elif single_label == "u":
            return "utsatt"
        else:
            return "avslått"

    if len(label) > 1:
        return [single_decision(char) for char in label]
    return single_decision(label)


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
