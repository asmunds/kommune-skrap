"""Train model and classify text in PDF files."""

from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from kommune_skrap.translator import load_translation_model, translate_text


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(pred: EvalPrediction) -> dict:
    """Computes and returns various evaluation metrics for the given predictions."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def extract_text_from_pdf(
    pdf_path: Path, translator_model, translator_tokenizer
) -> str:
    """Extract text from a PDF file and translate."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        break  # Decision is usually on the first page
    doc.close()
    translated_text = translate_text(text, translator_model, translator_tokenizer)
    return translated_text


def load_training_set(labels_file: Path) -> list:
    """Create a training set from PDF files and labels."""
    df = pd.read_csv(labels_file)
    training_data = []
    translator_model, translator_tokenizer = load_translation_model()
    for index, row in df.iterrows():
        pdf_path = Path(row["filename"])
        if pdf_path.exists():
            text = extract_text_from_pdf(
                pdf_path, translator_model, translator_tokenizer
            )
            training_data.append((text, row["label"]))
        else:
            print(f"Warning: {pdf_path} not found.")

    return training_data


def get_tokenizer(training_data, tokenizer: DistilBertTokenizer):
    """Get a tokenizer and encodings for the training data."""
    texts, labels = zip(*training_data)

    # Add presence keywords to tokenizer
    new_tokens = [
        "godkjent",
        "innvilget",
        "avslått",
        "godkjenning",
        "avslag",
        "utsatt",
        "søknaden innvilges",
        "søknaden gokjennes",
        "søknaden avslås",
        "søknaden utsettes",
    ]
    translator_model, translator_tokenizer = load_translation_model()
    for token in new_tokens:
        translated_token = translate_text(token, translator_model, translator_tokenizer)
        tokenizer.add_tokens([translated_token])

    # Tokenisering
    encodings = tokenizer(list(texts), truncation=True, padding=True)

    return tokenizer, encodings


def get_train_val_datasets(encodings, training_data):
    """Get training and validation datasets from encodings and labels."""
    texts, labels = zip(*training_data)
    labels = [1 if label == "innvilget" else 0 for label in labels]

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        encodings["input_ids"], labels, test_size=0.2
    )

    train_encodings = {"input_ids": train_texts}
    val_encodings = {"input_ids": val_texts}

    # Convert to correct dataset type
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    return train_dataset, val_dataset


def train_model(training_data, tokenizer: DistilBertTokenizer):
    """Train a model using the training data."""

    tokenizer, encodings = get_tokenizer(training_data=training_data)
    train_dataset, val_dataset = get_train_val_datasets(
        encodings=encodings, training_data=training_data
    )

    # Train the model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )

    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation Precision: {eval_results['eval_precision']:.4f}")
    print(f"Validation Recall: {eval_results['eval_recall']:.4f}")
    print(f"Validation F1 Score: {eval_results['eval_f1']:.4f}")

    return model


def load_model_and_tokenizer(
    model_dir: Path,
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizer]:
    """Load a trained model and tokenizer from a directory."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def evaluate_loaded_model(model, tokenizer, eval_dataset) -> None:
    """Evaluate the loaded model using the evaluation dataset."""
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation Precision: {eval_results['eval_precision']:.4f}")
    print(f"Validation Recall: {eval_results['eval_recall']:.4f}")
    print(f"Validation F1 Score: {eval_results['eval_f1']:.4f}")


def predict_label(model, tokenizer, pdf_path):
    """Predict the label of a PDF file using the trained model."""
    text = extract_text_from_pdf(pdf_path)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).item()
    return "innvilget" if predictions == 1 else "avslått"


def predict_all_files(model, tokenizer, data_folder: Path):
    """Predict the label of all PDF files in the data folder."""
    predictions = []
    for subfolder in data_folder.iterdir():
        if subfolder.is_dir() and subfolder.name != "training_data":
            print("\n", subfolder.name)
            for pdf_path in subfolder.iterdir():
                if pdf_path.exists() and pdf_path.suffix == ".pdf":
                    label = predict_label(model, tokenizer, pdf_path)
                    predictions.append((pdf_path, label))
    predictions_df = pd.DataFrame(predictions, columns=["filename", "predicted_label"])
    predictions_df.to_csv(data_folder / "predictions.csv", index=False)


if __name__ == "__main__":
    data_folder = Path("./data")
    labels_file = data_folder / "training_data/labels.csv"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    training_data = load_training_set(labels_file)
    model = train_model(training_data, tokenizer=tokenizer)
    predict_all_files(model, tokenizer, data_folder)

    # # Load model and evaluate
    # tokenizer, encodings = get_tokenizer(
    #     training_data=training_data, tokenizer=tokenizer
    # )
    # train_dataset, val_dataset = get_train_val_datasets(
    #     encodings=encodings, training_data=training_data
    # )
    # model, tokenizer = load_model_and_tokenizer(model_dir="./results/checkpoint-18")
    # evaluate_loaded_model(model, tokenizer, eval_dataset=val_dataset)
