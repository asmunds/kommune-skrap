"""Check prediction in data_folder/predictions.csv by opening files one by one
and confirming or correcting the predicted label. Add the correct label to the
training set and retrain the model.
"""

import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd


def assess_predictions(labels_file, prediction_file):
    """Check prediction in data_folder/predictions.csv."""
    predictions_df = pd.read_csv(prediction_file)
    training_df = pd.read_csv(labels_file)
    list_correct = []
    training_data = []
    for _, row in predictions_df.iterrows():
        if row["filename"] in training_df["filename"].values:
            continue
        pdf_path = Path(row["filename"])
        date = datetime.strptime(pdf_path.parts[1], "%d.%m-%Y").date()
        if pdf_path.exists():
            # Open pdf file for user to see
            webbrowser.open(pdf_path.absolute().as_uri())
            print(f"Predicted label: {row['predicted_label']}")
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
            if decision == row["predicted_label"]:
                list_correct.append(True)
            else:
                list_correct.append(False)
            training_data.append((pdf_path, decision, date))
        else:
            print(f"Warning: {pdf_path} not found.")
    new_training_df = pd.DataFrame(training_data, columns=["filename", "label", "date"])
    training_df = pd.concat([training_df, new_training_df], ignore_index=True)
    training_df.to_csv(labels_file, index=False)

    print("Accuracy: ", sum(list_correct) / len(list_correct))


if __name__ == "__main__":
    data_folder = Path("./data")
    labels_file = data_folder / "training_data/labels.csv"
    prediction_file = data_folder / "predictions.csv"
    assess_predictions(labels_file, prediction_file)
