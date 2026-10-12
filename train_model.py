from pathlib import Path
from urllib.parse import urlparse
import ipaddress
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import extract_features, parse_url


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_PHISHING_FILE = DATA_DIR / "Phishing URLs.csv"
DEFAULT_LEGITIMATE_FILE = DATA_DIR / "URL dataset.csv"


def _matching_csv(files, *keywords):
    lower_names = [path.name.lower() for path in files]
    for keyword_list in (keywords,):
        for keyword in keyword_list:
            for idx, name in enumerate(lower_names):
                if keyword in name:
                    return files[idx]
    return None


def resolve_dataset_files():
    """Prefer the expected filenames, then any likely matching CSVs, and finally create a demo dataset."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    phishing_file = _matching_csv(csv_files, "phishing")
    legitimate_file = _matching_csv(csv_files, "legitimate", "benign", "safe", "normal")

    if phishing_file is None:
        phishing_file = DEFAULT_PHISHING_FILE
    if legitimate_file is None:
        legitimate_file = DEFAULT_LEGITIMATE_FILE

    if not phishing_file.exists() or not legitimate_file.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not DEFAULT_PHISHING_FILE.exists() and not DEFAULT_LEGITIMATE_FILE.exists():
            demo_phishing_urls = [
                "http://secure-login-update-paypal.com/account/verify",
                "https://bankofamerica-security-check.com/login?confirm=1",
                "http://drive-google-docs-update-login.net/verify-account",
                "https://paypaI-security-confirmation.info/recovery/password",
                "http://bit.ly/secure-account-login",
                "https://amazon-security-alerts-login.co/authentication",
                "http://wwww.facebook-confirm-account.com/verify-user",
                "https://office365-update-account.net/secure-login",
                "https://paypal-login-verify-account.club/signin",
                "https://bank-login-verify.password-reset-info.com/account",
            ]
            demo_legitimate_urls = [
                "https://www.example.com/products/abc",
                "https://github.com/microsoft/vscode",
                "https://www.python.org/about/",
                "https://www.wikipedia.org/wiki/Phishing",
                "https://www.google.com/search?q=python",
                "https://www.stackoverflow.com/questions/123456",
                "https://www.nytimes.com/2024/01/01/technology",
                "https://www.amazon.com/gp/product/B000000000",
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://www.microsoft.com/en-us/",
            ]
            pd.DataFrame({"url": demo_phishing_urls}).to_csv(DEFAULT_PHISHING_FILE, index=False)
            pd.DataFrame({"url": demo_legitimate_urls}).to_csv(DEFAULT_LEGITIMATE_FILE, index=False)
            print("No dataset CSVs were found. Created a small demo dataset in the data/ folder so the project can run locally.")

    return phishing_file, legitimate_file


def clean_urls(data):
    data = data[["url"]].copy()
    data = data.dropna(subset=["url"])
    data["url"] = data["url"].astype(str).str.strip()
    data = data[data["url"] != ""]
    data = data.drop_duplicates(subset=["url"])
    return data.reset_index(drop=True)


def canonical_url_for_dedup(raw_url):
    """Normalize equivalent spellings for duplicate/conflicting-label checks.

    This is only for dataset deduplication; it does not determine URL safety.
    """
    try:
        normalized, parsed, hostname, port = parse_url(str(raw_url))
        scheme = parsed.scheme.lower()
        default_port = 443 if scheme == "https" else 80
        port_part = f":{port}" if port is not None and port != default_port else ""
        path = parsed.path or "/"
        # Preserve path/query case because they can be case-sensitive.
        return (
            f"{scheme}://{hostname}{port_part}{path}"
            + (f"?{parsed.query}" if parsed.query else "")
        )
    except ValueError:
        # Keep invalid rows distinct by their stripped text; they are later skipped.
        return str(raw_url).strip()


def main():
    PHISHING_FILE, LEGITIMATE_FILE = resolve_dataset_files()

    if not PHISHING_FILE.exists():
        raise FileNotFoundError(
            f"Phishing CSV not found: {PHISHING_FILE}\n"
            "Place the CSV files in the project data/ folder or edit DATA_DIR in train_model.py."
        )
    if not LEGITIMATE_FILE.exists():
        raise FileNotFoundError(
            f"Legitimate CSV not found: {LEGITIMATE_FILE}\n"
            "Place the CSV files in the project data/ folder or edit DATA_DIR in train_model.py."
        )

    print("Loading datasets...")
    phishing = pd.read_csv(PHISHING_FILE)
    legitimate = pd.read_csv(LEGITIMATE_FILE)

    for name, frame in (("phishing", phishing), ("legitimate", legitimate)):
        if "url" not in frame.columns:
            raise ValueError(
                f"The {name} dataset has no 'url' column. Columns found: {frame.columns.tolist()}"
            )

    phishing = clean_urls(phishing)
    legitimate = clean_urls(legitimate)

    phishing["Type"] = 1
    legitimate["Type"] = 0
    combined = pd.concat([phishing, legitimate], ignore_index=True)

    print(f"Phishing rows after basic cleaning: {len(phishing)}")
    print(f"Legitimate rows after basic cleaning: {len(legitimate)}")

    # Do not silently assign the phishing label when a URL appears in both files.
    combined["dedup_key"] = combined["url"].map(canonical_url_for_dedup)
    label_counts = combined.groupby("dedup_key")["Type"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)

    if conflicting_keys:
        conflicts = combined[combined["dedup_key"].isin(conflicting_keys)]
        print(f"\nWARNING: found {len(conflicting_keys)} URL(s) with conflicting labels.")
        print("Examples (up to 20):")
        print(conflicts[["url", "Type"]].head(20).to_string(index=False))
        combined = combined[~combined["dedup_key"].isin(conflicting_keys)].copy()
        print("Conflicting URLs were excluded from training; verify their source labels.")

    # Deduplicate within the remaining same-label data.
    combined = combined.drop_duplicates(subset=["dedup_key", "Type"]).reset_index(drop=True)

    # Skip malformed URLs rather than letting feature extraction crash the entire run.
    valid_mask = combined["url"].map(
        lambda value: _url_is_parseable(value)
    )
    invalid_count = int((~valid_mask).sum())
    if invalid_count:
        print(f"\nSkipping {invalid_count} malformed URL(s).")
    combined = combined[valid_mask].copy().reset_index(drop=True)

    if combined["Type"].nunique() < 2:
        raise ValueError("After cleaning, only one class remains. Check the datasets and labels.")
    if combined["Type"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two valid URLs for a stratified train/test split.")

    print("\nFinal class distribution:")
    print(combined["Type"].value_counts().rename(index={0: "Legitimate", 1: "Phishing"}))

    print("\nExtracting features...")
    feature_records = [extract_features(url) for url in combined["url"]]
    X = pd.DataFrame(feature_records).replace([np.inf, -np.inf], np.nan).fillna(0)
    feature_columns = X.columns.tolist()
    y = combined["Type"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\n========== MODEL RESULTS ==========")
    print(f"Hold-out accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted; order 0, 1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Save model artifacts together beside app.py.
    joblib.dump(model, BASE_DIR / "model.pkl")
    joblib.dump(feature_columns, BASE_DIR / "feature_columns.pkl")
    print(f"\nSaved model: {BASE_DIR / 'model.pkl'}")
    print(f"Saved feature columns: {BASE_DIR / 'feature_columns.pkl'}")
    print(f"Feature count: {len(feature_columns)}")
    print("\nNote: a random split can still overestimate real-world performance if related URLs/domains occur in both sets.")


def _url_is_parseable(value):
    try:
        parse_url(str(value))
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
