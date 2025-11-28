# README - Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale

**Instituție:** POLITEHNICA București - FIIR

**Student:** Chelu Fabian-Catalin

**Data:** 28.11.2025

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale" (Detectarea Phishing-ului în mesaje text). Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN (bazat pe arhitectura DistilBERT), respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

## 1\. Structura Repository-ului Github (versiunea Etapei 3)

proiect-rn-phishing/  
├── README.md  
├── docs/  
│ └── datasets/ # rapoarte EDA  
├── data/  
│ ├── raw/ # phishing_ai_generated.csv (date brute AI)  
│ ├── processed/ # processed_ai_generated.csv (date curățate)  
│ ├── train/ # train_ai_generated.csv (80%)  
│ ├── validation/ # validation_ai_generated.csv (10%)  
│ └── test/ # test_ai_generated.csv (10%)  
├── src/  
│ ├── preprocessing/ # preprocess_and_split.py  
│ ├── data_acquisition/ # generate_ai_data.py  
│ └── neural_network/ # model.py, train.py (pregătite pentru etapa următoare)  
├── config/ # preprocessing_config.yaml  
└── requirements.txt # tensorflow, transformers, pandas, scikit-learn  

## 2\. Descrierea Setului de Date

### 2.1 Sursa datelor

- **Origine:** Date sintetice generate folosind Large Language Models (Google Gemini 2.5 Flash).
- **Modul de achiziție:** ☐ Senzori reali / ☐ Simulare / ☐ Fișier extern / ☑ Generare programatică (Script Python via API).
- **Perioada / condițiile colectării:** Noiembrie 2025, utilizând prompt-uri diversificate pe topicuri de securitate (bancar, curierat, servicii streaming, HR, crypto).

### 2.2 Caracteristicile dataset-ului

- **Număr total de observații:** ~5,000 (extensibil prin scriptul de generare).
- **Număr de caracteristici (features):** 3 relevante (text, label, type).
- **Tipuri de date:** ☐ Numerice / ☑ Categoriale (Text) / ☐ Temporale / ☐ Imagini
- **Format fișiere:** ☑ CSV / ☐ TXT / ☐ JSON / ☐ PNG / ☐ Altele: \[...\]

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
| --- | --- | --- | --- | --- |
| text | string | -   | Conținutul mesajului (SMS/Email) | Lungime variabilă |
| --- | --- | --- | --- | --- |
| label | categorial | -   | Clasificare (Ținta) | {0: Legitim, 1: Phishing} |
| --- | --- | --- | --- | --- |
| type | categorial | -   | Canalul de comunicare | {sms, email} |
| --- | --- | --- | --- | --- |

**Fișier recomandat:** data/README.md

## 3\. Analiza Exploratorie a Datelor (EDA) - Sintetic

### 3.1 Statistici descriptive aplicate

- **Lungime text:** Analiza numărului de caractere și cuvinte per mesaj (diferențe SMS vs Email).
- **Distribuția claselor:** Verificarea echilibrului 50% Phishing / 50% Legitim impus la generare.
- **Vocabular:** Analiza frecvenței cuvintelor cheie (ex: "urgent", "click", "password").

### 3.2 Analiza calității datelor

- **Detectarea valorilor lipsă:** 0% (Scriptul de generare filtrează automat erorile și rândurile goale).
- **Detectarea valorilor inconsistente:** Eliminarea mesajelor care nu respectă formatul JSON.
- **Identificarea duplicatelor:** Verificare strictă pe coloana text pentru a evita memorarea de către model.

### 3.3 Probleme identificate

- \[x\] **Repetitivitate:** AI-ul tinde să repete anumite șabloane ("Dear customer"). _Soluție:_ Rotirea topicurilor în prompt (Banking, Netflix, Crypto, HR).
- \[x\] **Lungime variabilă:** Diferență mare între lungimea SMS-urilor (<160 caractere) și Email-uri. _Soluție:_ Padding/Truncation la nivelul Tokenizer-ului.

## 4\. Preprocesarea Datelor

### 4.1 Curățarea datelor

- **Eliminare duplicate:** Pe baza conținutului textului.
- **Tratarea valorilor lipsă:** Eliminarea automată a rândurilor goale.
- **Normalizare:**
  - Lowercase (litere mici).
  - Strip HTML (eliminare tag-uri &lt;br&gt;, &lt;div&gt; din email-uri).
  - Normalize Whitespace (eliminare spații duble și tab-uri).

### 4.2 Transformarea caracteristicilor

- **Tokenizare:** Utilizarea DistilBertTokenizer (WordPiece) specific modelului pre-antrenat.
- **Encoding:** Conversia textului în input_ids și attention_mask.
- **Truncation/Padding:** Uniformizarea secvențelor la max_length=128.

### 4.3 Structurarea seturilor de date

**Împărțire realizată:**

- 80% - train
- 10% - validation
- 10% - test

**Principii respectate:**

- **Stratificare:** Menținerea proporției claselor (Phishing/Legitim) în toate subseturile.
- **Random Seed:** 42 (pentru reproductibilitate).
- **Separare:** Datele de test nu sunt văzute niciodată de model la antrenare.

### 4.4 Salvarea rezultatelor preprocesării

- Date preprocesate în data/processed/processed_ai_generated.csv.
- Seturi train/val/test în foldere dedicate (data/train, data/validation, data/test).
- Parametrii de preprocesare în config/preprocessing_config.yaml.

## 5\. Fișiere Create în Această Etapă

- data/raw/phishing_ai_generated.csv - date brute.
- data/processed/processed_ai_generated.csv - date curățate.
- data/train/train_ai_generated.csv - set antrenare.
- data/validation/validation_ai_generated.csv - set validare.
- data/test/test_ai_generated.csv - set testare.
- src/preprocessing/preprocess_and_split.py - codul de preprocesare.

## 6\. Stare Etapă (de completat de student)

- \[x\] Structură repository configurată
- \[x\] Dataset analizat (EDA realizată)
- \[x\] Date preprocesate
- \[x\] Seturi train/val/test generate
- \[x\] Documentație actualizată în README + data/README.md