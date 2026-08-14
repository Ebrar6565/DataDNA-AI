import pandas as pd


def create_datadna_profile(df, target_column=None):
    row_count = df.shape[0]
    column_count = df.shape[1]

    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns

    numeric_count = len(numeric_columns)
    categorical_count = len(categorical_columns)

    numeric_ratio = numeric_count / column_count if column_count else 0
    categorical_ratio = categorical_count / column_count if column_count else 0

    missing_count = int(df.isna().sum().sum())
    total_cells = row_count * column_count

    missing_ratio = (
        missing_count / total_cells
        if total_cells
        else 0
    )

    duplicate_count = int(df.duplicated().sum())

    duplicate_ratio = (
        duplicate_count / row_count
        if row_count
        else 0
    )

    outlier_counts = {}

    for column in numeric_columns:
        series = df[column].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = (
            (series < lower_bound) |
            (series > upper_bound)
        ).sum()

        outlier_counts[column] = int(outlier_count)

    total_outlier_cells = sum(outlier_counts.values())

    numeric_cell_count = (
        df[numeric_columns]
        .notna()
        .sum()
        .sum()
    )

    outlier_ratio = (
        total_outlier_cells / numeric_cell_count
        if numeric_cell_count
        else 0
    )

    profile = {
        "row_count": row_count,
        "column_count": column_count,
        "numeric_count": numeric_count,
        "categorical_count": categorical_count,
        "numeric_ratio": round(float(numeric_ratio), 4),
        "categorical_ratio": round(float(categorical_ratio), 4),
        "missing_count": missing_count,
        "missing_ratio": round(float(missing_ratio), 4),
        "duplicate_count": duplicate_count,
        "duplicate_ratio": round(float(duplicate_ratio), 4),
        "outlier_count": int(total_outlier_cells),
        "outlier_ratio": round(float(outlier_ratio), 4),
        "outlier_by_column": outlier_counts
    }

    if target_column is not None:
        class_counts = df[target_column].value_counts()

        profile["target_column"] = target_column
        profile["class_count"] = len(class_counts)

        profile["class_distribution"] = {
            str(key): int(value)
            for key, value in class_counts.items()
        }

        if len(class_counts) > 1:
            profile["class_balance_ratio"] = round(
                float(
                    class_counts.min()
                    / class_counts.max()
                ),
                4
            )

    return profile

    