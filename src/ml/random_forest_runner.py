from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def build_random_forest_pipeline(
    X,
    class_weight=None,
    n_estimators=200,
    random_state=42
):
    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def evaluate_binary_model(
    model,
    X_test,
    y_test,
    positive_label="bad",
    negative_label="good"
):
    predictions = model.predict(X_test)

    results = {
        "accuracy": round(
            float(
                accuracy_score(
                    y_test,
                    predictions
                )
            ),
            4
        ),

        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    pos_label=positive_label
                )
            ),
            4
        ),

        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    pos_label=positive_label
                )
            ),
            4
        ),

        "f1_score": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    pos_label=positive_label
                )
            ),
            4
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[
                positive_label,
                negative_label
            ]
        ).tolist()
    }

    return results