"""Simply add files to predictions.csv file without predicting their labels."""

from pathlib import Path

import pandas as pd


def add_files_to_predictions_file(data_folder: Path, old_predictions: pd.DataFrame):
    """Predict the label of all PDF files in the data folder."""
    predictions = []
    for subitem in data_folder.iterdir():
        if subitem.is_dir() and subitem.name != "training_data":
            old_predictions = add_files_to_predictions_file(subitem, old_predictions)
        elif subitem.is_file():
            if subitem.exists() and subitem.suffix == ".pdf":
                if str(subitem) in old_predictions["filename"].values:
                    continue
                else:
                    label = "?"
                predictions.append((subitem, label))
    new_predictions = pd.DataFrame(predictions, columns=["filename", "predicted_label"])
    predictions_df = pd.concat([old_predictions, new_predictions], ignore_index=True)
    return predictions_df


if __name__ == "__main__":
    root_folder = Path(r"D:/kommune-skrap/data")
    data_folder = root_folder / "kristiansand"
    old_predictions = pd.read_csv(root_folder / "kristiansand_files.csv")
    predictions_df = add_files_to_predictions_file(
        data_folder=data_folder,
        old_predictions=old_predictions,
    )
    predictions_df.to_csv(root_folder / "new_kristiansand_files.csv", index=False)
