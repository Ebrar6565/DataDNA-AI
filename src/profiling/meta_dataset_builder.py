import pandas as pd


def build_meta_dataset(dataset_records):
    rows = []

    for dataset_name, record in dataset_records.items():
        fingerprint = record["fingerprint"]
        model_results = record["model_results"]

        baseline = model_results[
            "baseline_random_forest"
        ]

        balanced = model_results[
            "balanced_random_forest"
        ]

        recall_gain = (
            balanced["recall"]
            - baseline["recall"]
        )

        accuracy_change = (
            balanced["accuracy"]
            - baseline["accuracy"]
        )

        f1_change = (
            balanced["f1_score"]
            - baseline["f1_score"]
        )

        row = {
            "dataset_name": dataset_name,

            **fingerprint,

            "baseline_accuracy": baseline["accuracy"],
            "balanced_accuracy": balanced["accuracy"],

            "baseline_recall": baseline["recall"],
            "balanced_recall": balanced["recall"],

            "baseline_f1": baseline["f1_score"],
            "balanced_f1": balanced["f1_score"],

            "recall_gain": round(
                recall_gain,
                4
            ),

            "accuracy_change": round(
                accuracy_change,
                4
            ),

            "f1_change": round(
                f1_change,
                4
            )
        }

        rows.append(row)

    meta_dataset = pd.DataFrame(
        rows
    )

    return meta_dataset