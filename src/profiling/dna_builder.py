import pandas as pd

def create_datadna_profile(
    df,
    target_column=None,
    class_names=None
):
    row_count = df.shape[0]
    column_count = df.shape[1]

    # Target sütunu feature analizinden ayrılır
    if (
        target_column is not None
        and target_column in df.columns
    ):
        feature_df = df.drop(
            columns=[target_column]
        )
    else:
        feature_df = df.copy()

    feature_count = feature_df.shape[1]

    numeric_columns = feature_df.select_dtypes(
        include="number"
    ).columns

    categorical_columns = feature_df.select_dtypes(
        exclude="number"
    ).columns

    numeric_count = len(numeric_columns)
    categorical_count = len(categorical_columns)

    numeric_ratio = (
        numeric_count / feature_count
        if feature_count
        else 0
    )

    categorical_ratio = (
        categorical_count / feature_count
        if feature_count
        else 0
    )

    # Eksik değer ve duplicate analizi tüm veri seti üzerinde
    missing_count = int(
        df.isna().sum().sum()
    )

    total_cells = (
        row_count * column_count
    )

    missing_ratio = (
        missing_count / total_cells
        if total_cells
        else 0
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    duplicate_ratio = (
        duplicate_count / row_count
        if row_count
        else 0
    )

    # Outlier analizi sadece sayısal feature'lara uygulanır
    outlier_counts = {}

    for column in numeric_columns:
        series = feature_df[column].dropna()

        if series.empty:
            outlier_counts[column] = 0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outlier_count = (
            (
                (series < lower_bound)
                |
                (series > upper_bound)
            )
            .sum()
        )

        outlier_counts[column] = int(
            outlier_count
        )

    total_outlier_cells = sum(
        outlier_counts.values()
    )

    numeric_cell_count = (
        feature_df[numeric_columns]
        .notna()
        .sum()
        .sum()
    )

    outlier_ratio = (
        total_outlier_cells
        / numeric_cell_count
        if numeric_cell_count
        else 0
    )

    profile = {
        "row_count": row_count,
        "column_count": column_count,
        "feature_count": feature_count,

        "numeric_count": numeric_count,
        "categorical_count": categorical_count,

        "numeric_ratio": round(
            float(numeric_ratio),
            4
        ),

        "categorical_ratio": round(
            float(categorical_ratio),
            4
        ),

        "missing_count": missing_count,

        "missing_ratio": round(
            float(missing_ratio),
            4
        ),

        "duplicate_count": duplicate_count,

        "duplicate_ratio": round(
            float(duplicate_ratio),
            4
        ),

        "outlier_count": int(
            total_outlier_cells
        ),

        "outlier_ratio": round(
            float(outlier_ratio),
            4
        ),

        "outlier_by_column": outlier_counts
    }

    if (
        target_column is not None
        and target_column in df.columns
    ):
        class_counts = (
            df[target_column]
            .value_counts()
        )

        profile["target_column"] = (
            target_column
        )

        profile["class_count"] = len(
            class_counts
        )

        profile["class_distribution"] = {
    (
        str(class_names.get(key, key))
        if class_names is not None
        else str(key)
    ): int(value)
    for key, value in class_counts.items()
}

        if len(class_counts) > 1:
            profile[
                "class_balance_ratio"
            ] = round(
                float(
                    class_counts.min()
                    / class_counts.max()
                ),
                4
            )

    return profile