from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model():
    data = pd.DataFrame({
        "overtime": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                     0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "years_at_company": [5, 1, 8, 2, 1, 7, 6, 2, 9, 1,
                             4, 2, 8, 1, 5, 3, 7, 1, 6, 2],
        "attrition": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                      0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    X = data[["overtime", "years_at_company"]]
    y = data["attrition"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )
    model.fit(X_train, y_train)

    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    joblib.dump(model, output / "attrition_model.joblib")

    return model


if __name__ == "__main__":
    train_model()
    print("Model training completed.")
