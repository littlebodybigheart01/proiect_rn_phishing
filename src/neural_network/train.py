import sys
import os

# Adăugăm directorul curent (rădăcina proiectului) în sys.path
sys.path.append(os.getcwd())

import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer
from src.neural_network.model import build_distilbert_model

# --- CONFIGURĂRI ---
BATCH_SIZE = 16
EPOCHS = 3
MAX_LENGTH = 128
MODEL_NAME = 'distilbert-base-multilingual-cased'

# --- CĂI FIȘIERE (Așa cum ai cerut) ---
TRAIN_PATH = "data/train/train_ai_generated.csv"
VAL_PATH = "data/validation/validation_ai_generated.csv"
SAVE_DIRECTORY = "models/phishing_distilbert_multilingual"

def load_data():
    """Încarcă datele din CSV-uri."""
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Nu găsesc fișierul: {TRAIN_PATH}. Verifică numele fișierului!")

    print(f"[DATA] Citire CSV din {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # Convertim coloana de text la string (pentru siguranță, în caz că sunt numere interpretate greșit)
    train_df['processed_text'] = train_df['processed_text'].astype(str)
    val_df['processed_text'] = val_df['processed_text'].astype(str)

    return train_df, val_df

def convert_to_tf_dataset(texts, labels, tokenizer):
    """
    Convertește textul în tensori și creează un tf.data.Dataset optimizat.
    """
    print("[DATA] Tokenizare...")
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )

    # Creăm dataset-ul
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels.tolist()
    ))

    # Optimizări de performanță
    dataset = dataset.shuffle(len(texts)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def main():
    print("==========================================")
    print("       ANTRENARE MODEL PHISHING           ")
    print("==========================================")

    # 1. Verificare GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[SYSTEM] Antrenare pe GPU: {gpus}")
    else:
        print("[SYSTEM] Antrenare pe CPU (va dura mai mult).")

    # 2. Încărcare date
    try:
        train_df, val_df = load_data()
        print(f"[INFO] Date încărcate. Train: {len(train_df)} | Val: {len(val_df)}")
    except Exception as e:
        print(f"[EROARE] {e}")
        return

    # 3. Tokenizare
    print(f"[INIT] Descărcare Tokenizer: {MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = convert_to_tf_dataset(train_df['processed_text'], train_df['label'], tokenizer)
    val_dataset = convert_to_tf_dataset(val_df['processed_text'], val_df['label'], tokenizer)

    # 4. Construire Model
    model = build_distilbert_model()
    model.summary()

    # 5. Antrenare
    print(f"\n[START] Începe antrenarea pentru {EPOCHS} epoci...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )

    # 6. Salvare
    print(f"\n[INFO] Salvare model în '{SAVE_DIRECTORY}'...")
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)

    model.save_pretrained(SAVE_DIRECTORY)
    tokenizer.save_pretrained(SAVE_DIRECTORY)

    print("[SUCCES] Modelul a fost salvat! Poți trece la etapa de testare/evaluare.")

if __name__ == "__main__":
    main()