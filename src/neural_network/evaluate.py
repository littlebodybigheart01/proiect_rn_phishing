import os
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

sys.path.append(os.getcwd())

# --- CONFIGURARE ---
MODEL_PATH = "models/phishing_distilbert_multilingual"
TEST_DATA_PATH = "data/test/test_ai_generated.csv"
MAX_LENGTH = 128
BATCH_SIZE = 16


def evaluate_model():
    # 1. Încărcare Model și Tokenizer salvate anterior
    print(f"[INFO] Încărcare model din '{MODEL_PATH}'...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"[EROARE] Nu pot încărca modelul. Ai rulat train.py? Eroare: {e}")
        return

    # 2. Încărcare Date de Test
    print(f"[INFO] Încărcare date de test din '{TEST_DATA_PATH}'...")
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Asigurăm formatul string
    texts = df_test['processed_text'].astype(str).tolist()
    labels = df_test['label'].tolist()

    # 3. Tokenizare
    print("[INFO] Tokenizare date de test...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )

    # Creăm dataset TF (pentru viteză pe GPU)
    tf_dataset = tf.data.Dataset.from_tensor_slices(dict(encodings)).batch(BATCH_SIZE)

    # 4. Predicție
    print("[INFO] Se realizează predicțiile...")
    predictions = model.predict(tf_dataset)
    logits = predictions.logits

    # Transformăm logits în probabilități (Sigmoid)
    probs = tf.nn.sigmoid(logits).numpy().flatten()

    # Transformăm în 0 sau 1 (Pragul de 0.5)
    preds = (probs >= 0.5).astype(int)

    # 5. Raportare
    print("\n" + "=" * 50)
    print("REZULTATE EVALUARE (Pe date noi)")
    print("=" * 50)

    print(classification_report(labels, preds, target_names=["Legitim (0)", "Phishing (1)"]))

    # 6. Matricea de Confuzie (Grafic)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Legitim", "Phishing"],
                yticklabels=["Legitim", "Phishing"])
    plt.xlabel('Predicția Modelului')
    plt.ylabel('Realitatea (Eticheta)')
    plt.title('Matricea de Confuzie - DistilBERT')

    # Salvăm graficul
    os.makedirs("docs/datasets/plots", exist_ok=True)
    plt.savefig("docs/datasets/plots/confusion_matrix.png")
    print(f"[INFO] Matricea de confuzie salvată în 'docs/datasets/plots/confusion_matrix.png'")
    plt.show()  # (Opțional, dacă ai interfață grafică)


if __name__ == "__main__":
    evaluate_model()