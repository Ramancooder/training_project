# PhishGuard — URL Phishing Detector

A Flask web application using a Random Forest classifier and 31 URL-structure features.

## Project structure

```text
PhishGuard/
├── app.py
├── features.py
├── train_model.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── data/
    └── README.txt
```

## 1. Add your datasets

Put both CSV files inside `data/`:

- `Phishing URLs.csv` — phishing URLs; must contain a `url` column.
- `URL dataset.csv` — legitimate URLs; must contain a `url` column.

The training script labels the first file as `1` (Phishing) and the second as `0` (Legitimate). Review the source labels and duplicates before training. The CSV datasets and trained model artifacts are not bundled because they were not provided with the code.

## 2. Install dependencies

Open a terminal in this `PhishGuard` folder and run:

```bash
python -m venv .venv
```

Windows Command Prompt:

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 3. Train the model

Run this after the datasets have been added:

```bash
python train_model.py
```

The script reports accuracy, precision/recall, and a confusion matrix, then creates `model.pkl` and `feature_columns.pkl` in the project folder. If training reports conflicting labels, inspect those URLs in the CSV files instead of blindly trusting either label.

## 4. Start the web app

```bash
python app.py
```

Open `http://127.0.0.1:5000/` if it does not open automatically.

## Important notes

- `www.example.com` and `example.com` are normalized to equivalent feature representations when their remaining URL parts match. You must retrain after this feature-extraction change.
- An unusual port such as `:34243` is recorded as a risk signal; it is not by itself proof of phishing.
- The app does not visit submitted websites or verify their current reputation. A model prediction is not a guarantee of safety.
- A random train/test split may overestimate performance if URLs from the same domain appear in both sets. Use a domain-separated test set for a stronger evaluation.
- Only scan URLs you are authorized to analyze. Do not enter credentials into suspicious sites.
