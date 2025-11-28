import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification


def build_distilbert_model(learning_rate=5e-5):
    """
    Descarcă și configurează DistilBERT pentru clasificare binară.
    """
    # 1. Încărcăm modelul pre-antrenat
    # num_labels=1 -> Vrem o singură valoare la ieșire (scorul de phishing)
    model = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=1
    )

    # 2. Optimizator
    # Pentru Transformers, se folosește un Learning Rate foarte mic (5e-5)
    # pentru a nu strica ce a învățat modelul deja ("Fine-Tuning").
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 3. Funcția de pierdere (Loss)
    # from_logits=True este CRUCIAL aici.
    # Modelele HuggingFace nu au stratul final de activare (Sigmoid), scot valori brute.
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 4. Compilare
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model