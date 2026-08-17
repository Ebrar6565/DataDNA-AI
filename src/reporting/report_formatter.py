def format_datadna_report(report):
    profile = report["dataset_profile"]
    recommendations = report["data_recommendations"]
    model_results = report["model_results"]
    interpretations = report["model_interpretations"]

    baseline = model_results["baseline_random_forest"]
    balanced = model_results["balanced_random_forest"]

    class_distribution = profile.get(
        "class_distribution",
        {}
    )

    lines = []

    lines.append("=" * 55)
    lines.append("DATADNA AI - ANALYSIS REPORT")
    lines.append("=" * 55)

    lines.append("")
    lines.append("1. DATASET STRUCTURE")
    lines.append("-" * 55)

    lines.append(
        f"Row Count: {profile['row_count']}"
    )

    lines.append(
        f"Column Count: {profile['column_count']}"
    )

    lines.append(
        f"Numeric Columns: {profile['numeric_count']}"
    )

    lines.append(
        f"Categorical Columns: {profile['categorical_count']}"
    )

    lines.append(
        f"Numeric Ratio: {profile['numeric_ratio'] * 100:.2f}%"
    )

    lines.append(
        f"Categorical Ratio: {profile['categorical_ratio'] * 100:.2f}%"
    )

    lines.append("")
    lines.append("2. DATA QUALITY")
    lines.append("-" * 55)

    lines.append(
        f"Missing Value Ratio: {profile['missing_ratio'] * 100:.2f}%"
    )

    lines.append(
        f"Duplicate Row Ratio: {profile['duplicate_ratio'] * 100:.2f}%"
    )

    lines.append(
        f"IQR Outlier Candidate Ratio: {profile['outlier_ratio'] * 100:.2f}%"
    )

    lines.append("")
    lines.append("3. TARGET ANALYSIS")
    lines.append("-" * 55)

    if "target_column" in profile:
        lines.append(
            f"Target Column: {profile['target_column']}"
        )

        lines.append(
            f"Number of Classes: {profile['class_count']}"
        )

        for class_name, count in class_distribution.items():
            lines.append(
                f"{class_name}: {count}"
            )

        if "class_balance_ratio" in profile:
            lines.append(
                "Class Balance Ratio: "
                f"{profile['class_balance_ratio']:.4f}"
            )

    lines.append("")
    lines.append("4. DATA RECOMMENDATIONS")
    lines.append("-" * 55)

    for recommendation in recommendations:
        lines.append(
            f"- {recommendation}"
        )

    lines.append("")
    lines.append("5. RANDOM FOREST RESULTS")
    lines.append("-" * 55)

    lines.append("Baseline Random Forest")
    lines.append(
        f"  Accuracy: {baseline['accuracy']:.4f}"
    )
    lines.append(
        f"  Precision: {baseline['precision']:.4f}"
    )
    lines.append(
        f"  Recall: {baseline['recall']:.4f}"
    )
    lines.append(
        f"  F1-score: {baseline['f1_score']:.4f}"
    )

    lines.append("")

    lines.append("Balanced Random Forest")
    lines.append(
        f"  Accuracy: {balanced['accuracy']:.4f}"
    )
    lines.append(
        f"  Precision: {balanced['precision']:.4f}"
    )
    lines.append(
        f"  Recall: {balanced['recall']:.4f}"
    )
    lines.append(
        f"  F1-score: {balanced['f1_score']:.4f}"
    )

    lines.append("")
    lines.append("6. MODEL INTERPRETATION")
    lines.append("-" * 55)

    for interpretation in interpretations:
        lines.append(
            f"- {interpretation}"
        )

    lines.append("")
    lines.append("=" * 55)

    return "\n".join(lines)