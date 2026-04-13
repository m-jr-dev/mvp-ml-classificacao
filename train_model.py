
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "wine_classifier.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

FEATURE_COLUMNS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]
TARGET_COLUMN = "target"


def load_dataset() -> pd.DataFrame:
    columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    try:
        dataframe = pd.read_csv(DATA_URL, header=None, names=columns)
        return dataframe
    except Exception:
        wine = load_wine(as_frame=True)
        dataframe = wine.frame.copy()
        dataframe.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        dataframe = dataframe[[TARGET_COLUMN, *FEATURE_COLUMNS]]
        dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN] + 1
        return dataframe


def build_search_configs() -> dict[str, tuple[Pipeline, dict]]:
    numeric_features = FEATURE_COLUMNS

    base_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", "passthrough"),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", base_preprocessor),
            ("classifier", KNeighborsClassifier()),
        ]
    )
    tree_pipeline = Pipeline(
        steps=[
            ("preprocessor", base_preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )
    nb_pipeline = Pipeline(
        steps=[
            ("preprocessor", base_preprocessor),
            ("classifier", GaussianNB()),
        ]
    )
    svm_pipeline = Pipeline(
        steps=[
            ("preprocessor", base_preprocessor),
            ("classifier", SVC(probability=False, random_state=42)),
        ]
    )

    configs = {
        "KNN": (
            knn_pipeline,
            {
                "preprocessor__numeric__scaler": [StandardScaler(), MinMaxScaler()],
                "classifier__n_neighbors": [3, 5, 7, 9],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
        ),
        "Árvore de Decisão": (
            tree_pipeline,
            {
                "preprocessor__numeric__scaler": ["passthrough", StandardScaler(), MinMaxScaler()],
                "classifier__max_depth": [None, 3, 5, 8],
                "classifier__min_samples_split": [2, 4, 6],
                "classifier__criterion": ["gini", "entropy"],
            },
        ),
        "Naive Bayes": (
            nb_pipeline,
            {
                "preprocessor__numeric__scaler": [StandardScaler(), MinMaxScaler()],
                "classifier__var_smoothing": [1e-09, 1e-08, 1e-07],
            },
        ),
        "SVM": (
            svm_pipeline,
            {
                "preprocessor__numeric__scaler": [StandardScaler(), MinMaxScaler()],
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto"],
            },
        ),
    }

    return configs


def train_and_export() -> dict:
    dataframe = load_dataset()
    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configs = build_search_configs()

    ranking: list[dict] = []
    best_name = None
    best_search = None
    best_f1 = -1.0

    for model_name, (pipeline, params) in configs.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_macro": round(float(precision_score(y_test, predictions, average="macro")), 4),
            "recall_macro": round(float(recall_score(y_test, predictions, average="macro")), 4),
            "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
        }

        serialized_params = {
            key: str(value) for key, value in search.best_params_.items()
        }

        ranking.append(
            {
                "model": model_name,
                "best_params": serialized_params,
                "cv_best_score_f1_macro": round(float(search.best_score_), 4),
                **metrics,
            }
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_name = model_name
            best_search = search

    ranking = sorted(ranking, key=lambda item: item["f1_macro"], reverse=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_search.best_estimator_, MODEL_PATH)

    payload = {
        "dataset_url": DATA_URL,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "holdout_test_size": 0.20,
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "scoring": "f1_macro",
        },
        "ranking": ranking,
        "selected_model": best_name,
        "selected_model_metrics": ranking[0],
        "thresholds": {
            "minimum_accuracy": 0.90,
            "minimum_f1_macro": 0.90,
        },
    }

    METADATA_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


if __name__ == "__main__":
    metadata = train_and_export()
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
