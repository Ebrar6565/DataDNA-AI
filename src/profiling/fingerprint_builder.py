def create_dataset_fingerprint(profile):
    fingerprint = {
        "row_count": profile.get(
            "row_count",
            0
        ),

        "feature_count": profile.get(
            "feature_count",
            0
        ),

        "numeric_ratio": profile.get(
            "numeric_ratio",
            0
        ),

        "categorical_ratio": profile.get(
            "categorical_ratio",
            0
        ),

        "missing_ratio": profile.get(
            "missing_ratio",
            0
        ),

        "duplicate_ratio": profile.get(
            "duplicate_ratio",
            0
        ),

        "outlier_ratio": profile.get(
            "outlier_ratio",
            0
        ),

        "class_balance_ratio": profile.get(
            "class_balance_ratio",
            None
        )
    }

    return fingerprint