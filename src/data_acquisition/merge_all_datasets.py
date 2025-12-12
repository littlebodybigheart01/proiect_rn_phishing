import pandas as pd
import os
import numpy as np

# --- CONFIGURARE CĂI ---
# Asigură-te că numele fișierelor de aici corespund exact cu ce ai în folderul data/raw/
FILE_EMAIL = "data/raw/emailreal.csv"
FILE_SMS = "data/raw/smsreal.csv"
FILE_AI_RO = "data/raw/phishing_ai_ro_only.csv"

OUTPUT_FILE = "data/raw/final_multilingual_dataset.csv"


def clean_text(text):
    """
    Funcție optimizată pentru curățarea textului.
    Elimină caracterele care strică structura CSV (Newlines, Tabs).
    """
    if not isinstance(text, str):
        return ""
    # Înlocuim Enter și Tab cu spațiu simplu
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Eliminăm spațiile multiple (ex: "  " -> " ")
    return " ".join(text.split())


def process_email_dataset():
    """Procesează Phishing_Email.csv"""
    if not os.path.exists(FILE_EMAIL):
        print(f"[SKIP] Nu am găsit fișierul: {FILE_EMAIL}")
        return pd.DataFrame()

    print(f"[1/3] Procesare Email-uri ({FILE_EMAIL})...")
    try:
        # Citim doar coloanele relevante pentru a economisi memorie
        # 'Email Text' și 'Email Type' sunt numele standard din acest dataset
        df = pd.read_csv(FILE_EMAIL, usecols=['Email Text', 'Email Type'])

        # Redenumire standard
        df.rename(columns={'Email Text': 'text', 'Email Type': 'label_raw'}, inplace=True)

        # Mapare Etichete: Safe Email -> 0, Phishing Email -> 1
        # Folosim .strip() pentru a elimina spațiile ascunse din etichete
        df['label'] = df['label_raw'].astype(str).str.strip().map({
            'Safe Email': 0,
            'Phishing Email': 1
        })

        # Adăugăm metadate
        df['type'] = 'email'
        df['source'] = 'real_enron'
        df['lang'] = 'en'

        # Eliminăm rândurile care nu au putut fi etichetate (erori in CSV original)
        df.dropna(subset=['label'], inplace=True)

        return df[['text', 'label', 'type', 'source', 'lang']]

    except Exception as e:
        print(f"[EROARE] La procesarea Email: {e}")
        return pd.DataFrame()


def process_sms_dataset():
    """Procesează spam.csv (SMS Collection)"""
    if not os.path.exists(FILE_SMS):
        print(f"[SKIP] Nu am găsit fișierul: {FILE_SMS}")
        return pd.DataFrame()

    print(f"[2/3] Procesare SMS-uri ({FILE_SMS})...")
    try:
        # Acest fișier are encoding 'latin-1' adesea
        # Coloanele sunt v1 (label) și v2 (text)
        df = pd.read_csv(FILE_SMS, encoding='latin-1', usecols=['v1', 'v2'])

        # Redenumire
        df.rename(columns={'v2': 'text', 'v1': 'label_raw'}, inplace=True)

        # Mapare: ham -> 0, spam -> 1
        df['label'] = df['label_raw'].map({'ham': 0, 'spam': 1})

        df['type'] = 'sms'
        df['source'] = 'real_sms_collection'
        df['lang'] = 'en'

        return df[['text', 'label', 'type', 'source', 'lang']]

    except Exception as e:
        print(f"[EROARE] La procesarea SMS: {e}")
        return pd.DataFrame()


def process_ai_ro_dataset():
    """Procesează datele generate de AI (Română)"""
    if not os.path.exists(FILE_AI_RO):
        print(f"[SKIP] Nu am găsit datele AI ({FILE_AI_RO})...")
        return pd.DataFrame()

    print(f"[3/3] Procesare Date AI Română ({FILE_AI_RO})...")
    try:
        df = pd.read_csv(FILE_AI_RO)

        # Asigurăm compatibilitatea coloanelor
        if 'lang' not in df.columns: df['lang'] = 'ro'
        if 'source' not in df.columns: df['source'] = 'ai_generated'

        # AI-ul generează deja 0/1, deci nu trebuie mapare, doar verificare
        # Dacă coloanele au alte nume în fișierul generat, le poți redenumi aici

        return df[['text', 'label', 'type', 'source', 'lang']]

    except Exception as e:
        print(f"[EROARE] La procesarea AI RO: {e}")
        return pd.DataFrame()


def main():
    print("==========================================")
    print("   UNIFICARE DATASET-URI PENTRU PHISHING  ")
    print("==========================================")

    # 1. Încărcare
    df_email = process_email_dataset()
    df_sms = process_sms_dataset()
    df_ai = process_ai_ro_dataset()

    # 2. Combinare
    # Folosind o listă și pd.concat este metoda cea mai eficientă de memorie
    datasets = [df_email, df_sms, df_ai]

    # Filtrare dataset-uri goale
    valid_datasets = [d for d in datasets if not d.empty]

    if not valid_datasets:
        print("[EROARE CRITICĂ] Niciun set de date nu a putut fi încărcat!")
        return

    df_final = pd.concat(valid_datasets, ignore_index=True)
    print(f"\n[INFO] Total rânduri brute: {len(df_final)}")

    # 3. Curățare Avansată
    print("[INFO] Se curăță textul și se elimină duplicatele...")

    # Eliminare rânduri fără text sau label
    df_final.dropna(subset=['text', 'label'], inplace=True)

    # Curățare text (eliminare newlines pentru a nu strica CSV-ul final)
    df_final['text'] = df_final['text'].astype(str).apply(clean_text)

    # Eliminare duplicate (foarte important când combini mai multe surse)
    initial_len = len(df_final)
    df_final.drop_duplicates(subset=['text'], inplace=True)
    print(f"   -> Duplicate eliminate: {initial_len - len(df_final)}")

    # Eliminare texte prea scurte (ex: "ok", "hi") care nu sunt relevante
    df_final = df_final[df_final['text'].str.len() > 5]

    # Convertire label la int (pentru siguranță)
    df_final['label'] = df_final['label'].astype(int)

    # 4. Amestecare (Shuffle)
    # Este vital să amestecăm datele ca să nu avem toate cele în română la final
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Salvare
    # quoting=1 pune ghilimele în jurul textului, protejând virgulele din interiorul mesajelor
    df_final.to_csv(OUTPUT_FILE, index=False, quoting=1)

    print("\n" + "=" * 40)
    print("   STATISTICI FINALE")
    print("=" * 40)
    print(f"Fișier salvat: {OUTPUT_FILE}")
    print(f"Total Mesaje:  {len(df_final)}")
    print("-" * 20)
    print("Distribuție Clase:")
    print(df_final['label'].value_counts().to_string())
    print("-" * 20)
    print("Distribuție Limbi:")
    print(df_final['lang'].value_counts().to_string())
    print("=" * 40)


if __name__ == "__main__":
    main()