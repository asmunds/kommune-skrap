"""Train model and classify text in PDF files."""

from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)


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


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_training_set(labels_file: Path) -> list:
    """Create a training set from PDF files and labels."""
    df = pd.read_csv(labels_file)
    training_data = []

    for index, row in df.iterrows():
        pdf_path = Path(row["filename"])
        if pdf_path.exists():
            text = extract_text_from_pdf(pdf_path)
            training_data.append((text, row["label"]))
        else:
            print(f"Warning: {pdf_path} not found.")

    return training_data


def train_model(training_data, tokenizer: DistilBertTokenizer):
    """Train a model using the training data."""

    texts, labels = zip(*training_data)
    labels = [1 if label == "innvilget" else 0 for label in labels]

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
    tokenizer.add_tokens(new_tokens)

    # Tokenisering
    encodings = tokenizer(list(texts), truncation=True, padding=True)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        encodings["input_ids"], labels, test_size=0.2
    )

    train_encodings = {"input_ids": train_texts}
    val_encodings = {"input_ids": val_texts}

    # Convert to correct dataset type
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

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
    )

    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Validation Loss(?): {eval_results['eval_loss']:.4f}")

    return model


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
    training_set = load_training_set(labels_file)
    model = train_model(training_set, tokenizer=tokenizer)
    predict_all_files(model, tokenizer, data_folder)
