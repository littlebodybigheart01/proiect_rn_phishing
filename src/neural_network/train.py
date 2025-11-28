import pandas as pd
import tensorflow as tf
import os
import shutil
from transformers import DistilBertTokenizer
from src.neural_network.model import build_distilbert_model

# --- CONFIGURĂRI ---
BATCH_SIZE = 16  # Mai mic, ca să nu umple memoria RAM
EPOCHS = 3  # BERT învață extrem de repede. 3 epoci sunt suficiente.
MAX_LENGTH = 128  # Lungimea maximă a secvenței

# Căi fișiere
TRAIN_PATH = "data/train/train_ai_generated.csv"
VAL_PATH = "data/validation/validation_ai_generated.csv"
SAVE_DIRECTORY = "models/phishing_distilbert"


def load_data():
    """Încarcă datele din CSV-uri."""
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError("Nu găsesc datele de antrenament. Rulează preprocesarea!")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # Ne asigurăm că sunt string-uri
    return train_df, val_df


def convert_to_tf_dataset(texts, labels, tokenizer):
    """
    Convertește textul brut într-un format optimizat pentru TensorFlow (tf.data.Dataset).
    """
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )

    # Creăm un dataset TensorFlow din input-uri (ids, masks) și etichete
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels.tolist()
    ))

    # Amestecăm și grupăm în batch-uri
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE)
    return dataset


def main():
    print("==========================================")
    print("   ANTRENARE MODEL PHISHING (DISTILBERT)  ")
    print("==========================================")

    # 1. Încărcare date
    try:
        train_df, val_df = load_data()
        print(f"[INFO] Date încărcate. Train: {len(train_df)} | Val: {len(val_df)}")
    except Exception as e:
        print(e)
        return

    # 2. Tokenizare
    print("[INFO] Descărcare Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    print("[INFO] Procesare dataset (Tokenizare)...")
    train_dataset = convert_to_tf_dataset(train_df['processed_text'].astype(str), train_df['label'], tokenizer)
    val_dataset = convert_to_tf_dataset(val_df['processed_text'].astype(str), val_df['label'], tokenizer)

    # 3. Construire Model
    print("[INFO] Construire model DistilBERT...")
    model = build_distilbert_model()

    # 4. Antrenare
    print(f"[INFO] Start antrenare pentru {EPOCHS} epoci...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )

    # 5. Salvare
    print(f"[INFO] Salvare model în '{SAVE_DIRECTORY}'...")

    # Creăm folderul dacă nu există
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)

    model.save_pretrained(SAVE_DIRECTORY)
    tokenizer.save_pretrained(SAVE_DIRECTORY)

    print("[SUCCES] Modelul este salvat și gata de utilizare!")


if __name__ == "__main__":
    main()