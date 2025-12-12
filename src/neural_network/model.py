import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification


def build_distilbert_model(learning_rate=2e-5):
    """
    Descarcă și configurează DistilBERT Multilingual pentru clasificare binară.
    """
    model_name = 'distilbert-base-multilingual-cased'

    print(f"[MODEL] Se descarcă modelul pre-antrenat (format .h5): {model_name} ...")

    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        use_safetensors=False
    )

    # Optimizator
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss function
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Compilare
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model