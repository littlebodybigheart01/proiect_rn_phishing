# src/data_acquisition/generate_ai_data_massive.py
import os
import time
import json
import random
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

# --- CONFIGURARE ---
API_KEY = "API-ul tau"
TARGET_COUNT = 10000  # Câte vrei? (ex: 10.000)
BATCH_SIZE = 25  # Cerem 25 o dată (maxim safe pentru a nu tăia JSON-ul)
OUTPUT_FILE = os.path.join("data", "raw", "phishing_ai_massive.csv")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Teme pentru a evita repetiția
TOPICS = [
    "banking security alert", "package delivery failed", "netflix subscription expired",
    "crypto wallet blocked", "HR salary update", "lottery winner",
    "CEO urgent wire transfer", "university password reset", "dating app match",
    "cloud storage full", "unpaid invoice", "friend stranded abroad"
]


def get_diverse_prompt(batch_size, topic):
    return f"""
    Sarcina: Generează un dataset JSON pentru detectarea phishing-ului.
    Subiect principal pentru acest batch: {topic}.

    Generează exact {batch_size} mesaje, respectând regulile:
    1. Distribuție: 50% Phishing, 50% Legitim (pe același subiect).
    2. Format Phishing: Creează urgență, link-uri mascate, greșeli intenționate.
    3. Format Legitim: Ton calm, informativ, fără cereri de acțiune rapidă suspectă.
    4. Tipuri: Amestecă Email și SMS.

    Returnează DOAR un JSON array valid de forma:
    [
      {{"text": "Conținut mesaj...", "label": 1, "type": "sms", "topic": "{topic}"}},
      {{"text": "Conținut mesaj...", "label": 0, "type": "email", "topic": "{topic}"}}
    ]
    """


def save_batch_to_csv(data_batch, filepath):
    """Salvează datele incremental (append mode)"""
    df = pd.DataFrame(data_batch)
    # Dacă fișierul nu există, scriem header-ul. Dacă există, doar datele.
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', header=header, index=False)


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Verificăm dacă există deja date generate ca să continuăm
    current_count = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            current_count = len(pd.read_csv(OUTPUT_FILE))
            print(f"[INFO] Fișier existent găsit. Continuăm de la {current_count} rânduri.")
        except:
            print("[WARN] Fișierul existent pare corupt sau gol. Începem de la 0.")

    pbar = tqdm(total=TARGET_COUNT, initial=current_count)

    while current_count < TARGET_COUNT:
        # Alegem un subiect random pentru diversitate
        topic = random.choice(TOPICS)

        try:
            # Apel AI
            response = model.generate_content(get_diverse_prompt(BATCH_SIZE, topic))

            # Curățare text (uneori AI-ul mai pune text extra)
            txt = response.text.strip()
            if txt.startswith("```json"):
                txt = txt[7:-3]
            elif txt.startswith("```"):
                txt = txt[3:-3]

            batch_data = json.loads(txt)

            # Validare simplă (să fim siguri că avem câmpurile necesare)
            valid_data = [d for d in batch_data if 'text' in d and 'label' in d]

            if valid_data:
                save_batch_to_csv(valid_data, OUTPUT_FILE)
                count = len(valid_data)
                current_count += count
                pbar.update(count)

            # Pauză de siguranță (4 secunde e foarte safe pentru Free Tier)
            time.sleep(4)

        except Exception as e:
            # Dacă apare o eroare (ex: server overload), așteptăm puțin mai mult și continuăm
            print(f" [Eroare pasageră: {e}]. Aștept 10 secunde...")
            time.sleep(10)

    pbar.close()
    print(f"\n[GATA] Ai generat {current_count} exemple în {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
