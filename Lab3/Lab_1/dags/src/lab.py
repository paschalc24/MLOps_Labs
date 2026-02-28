import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import pickle
import os
import base64

def load_data():
    """
    Loads the Iris dataset from scikit-learn.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("Loading dataset")
    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target
    print(f"Loaded {len(df)} rows with columns:  {list(df.columns)}")
    serialized_data = pickle.dumps(df)                        # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    X = df[[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"]
    ]
    y = df["target"]

    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    # bytes -> base64 string for XCom
    serialized_data = pickle.dumps(data)
    return base64.b64encode(serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a Random Forest classifier model on the preprocessed data and saves it.
    Returns test data as base64 for model evaluation.
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)

    X_train = data["X_train"]
    y_train = data["y_train"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"Training accuracy: {train_acc}")

    # NOTE: This saves the last-fitted model (k=49), matching your original intent.
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    test_data = {
        "X_test": data["X_test"],
        "y_test": data["y_test"]
    }
    serialized_data = pickle.dumps(test_data)
    return base64.b64encode(serialized_data).decode("ascii")


def load_model(filename: str, data_b64: str):
    """
    Loads the saved model and prints test accuracy and
    classification report.
    """
    # load the saved (last-fitted) model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))


    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)

    X_test = data["X_test"]
    y_test = data["y_test"]

    predictions = loaded_model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {acc}")

    target_names = ["setosa", "versicolor", "virginica"]
    report = classification_report(y_test, predictions, target_names=target_names)
    print(f"Classification Report:\n{report}")

    return float(acc)
