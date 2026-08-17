def build_datadna_report(
    profile,
    recommendations,
    baseline_results,
    balanced_results,
    model_interpretations
):
    report = {
        "dataset_profile": profile,

        "data_recommendations": recommendations,

        "model_results": {
            "baseline_random_forest": baseline_results,
            "balanced_random_forest": balanced_results
        },

        "model_interpretations": model_interpretations
    }

    return report