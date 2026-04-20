import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent.parent / "data"
PARAMS_FILE = DATA_DIR / "best_params.json"

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.1, 0.3],
}


def load_and_train() -> tuple[XGBClassifier, list[str], LabelEncoder]:
    train = pd.read_csv(DATA_DIR / "Training.csv")
    train.columns = [c.replace("_", " ") for c in train.columns]

    X = train.drop("prognosis", axis=1)
    y = train["prognosis"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    symptom_columns = list(X.columns)

    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            best_params = json.load(f)
        print(f"Loaded saved params: {best_params}")
        model = XGBClassifier(**best_params, eval_metric="mlogloss", random_state=42, n_jobs=-1)
        model.fit(X, y_enc)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.33, random_state=42)

        base = XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(base, PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best params: {best_params}")

        with open(PARAMS_FILE, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved best params to {PARAMS_FILE}")

        y_pred = model.predict(X_test)
        print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        model.fit(X, y_enc)

    return model, symptom_columns, le


def predict(model: XGBClassifier, symptom_columns: list[str], spoken_text: str, le: LabelEncoder = None) -> str:
    spoken = spoken_text.lower()
    row = np.zeros(len(symptom_columns))
    for i, symptom in enumerate(symptom_columns):
        if symptom in spoken:
            row[i] = 1

    df = pd.DataFrame([row], columns=symptom_columns)
    pred = model.predict(df)[0]
    return le.inverse_transform([pred])[0] if le is not None else str(pred)
