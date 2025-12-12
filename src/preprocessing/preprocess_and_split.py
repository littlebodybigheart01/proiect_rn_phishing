import os
import yaml
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Căi implicite
CONFIG_PATH = os.path.join("config", "preprocessing_config.yaml")


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def clean_text(text, config):
    """Aplică regulile de curățare din config."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    if config['preprocessing']['lowercase']:
        text = text.lower()

    # 2. Strip HTML (simplificat)
    if config['preprocessing']['strip_html']:
        text = re.sub(r'<[^>]+>', '', text)

    # 3. Normalize Whitespace
    if config['preprocessing']['normalize_whitespace']:
        text = " ".join(text.split())

    # 4. Remove URLs (dacă e setat pe True - de obicei la phishing vrem să le lăsăm sau să le mascăm)
    if config.get('preprocessing', {}).get('remove_urls', False):
        text = re.sub(r'http\S+', '', text)

    return text


def main():
    print("[INFO] Încărcare configurare...")
    cfg = load_config(CONFIG_PATH)

    # Căi fișiere
    input_file = cfg['dataset']['input_path']

    # Verificăm dacă s-au generat datele
    if not os.path.exists(input_file):
        print(f"[EROARE] Nu găsesc fișierul de intrare: {input_file}")
        print("Așteaptă să termine scriptul de generare AI!")
        return

    print(f"[INFO] Citire date din {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   -> {len(df)} rânduri inițiale.")

    # 1. Eliminare duplicate
    if cfg['preprocessing']['remove_duplicates']:
        initial_len = len(df)
        df.drop_duplicates(subset=['text'], inplace=True)
        print(f"[INFO] Eliminate {initial_len - len(df)} duplicate.")

    # 2. Eliminare text gol
    if cfg['missing_values']['drop_empty_text']:
        df.dropna(subset=['text'], inplace=True)
        df = df[df['text'].str.strip() != ""]

    # 3. Curățare Text
    print("[INFO] Aplicare curățare text...")
    df['processed_text'] = df['text'].apply(lambda x: clean_text(x, cfg))

    # 4. Împărțire (Train / Val / Test)
    # Folosim 'processed_text' ca input și 'label' ca output
    X = df['processed_text']
    y = df['label']

    # Prima împărțire: Train vs (Test + Val)
    train_size = cfg['split']['train_size']
    # Restul (ex: 20%) se duce în temp
    test_val_size = 1.0 - train_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y,
        train_size=train_size,
        stratify=y if cfg['split']['stratify'] else None,
        random_state=cfg['split']['random_seed']
    )

    # A doua împărțire: Val vs Test (împărțim temp în două jumătăți egale)
    # De obicei val_size și test_size sunt egale în config (ex: 0.1 și 0.1)
    # Deci împărțim temp la jumătate (0.5)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp if cfg['split']['stratify'] else None,
        random_state=cfg['split']['random_seed']
    )

    # Salvare
    print("[INFO] Salvare seturi de date...")

    # Asigurare foldere
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/validation", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    # Salvăm întregul CSV preprocesat (opțional)
    df.to_csv(f"data/processed/processed_{cfg['dataset']['subset']}.csv", index=False)

    # Salvăm seturile finale
    X_train.to_csv(f"data/train/train_{cfg['dataset']['subset']}.csv", index=False)
    X_val.to_csv(f"data/validation/validation_{cfg['dataset']['subset']}.csv", index=False)
    X_test.to_csv(f"data/test/test_{cfg['dataset']['subset']}.csv", index=False)

    print(f"[SUCCES] Gata!")
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


if __name__ == "__main__":
    main()