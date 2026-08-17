import json

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.profiling.dna_builder import (
    create_datadna_profile
)

from src.profiling.recommendation_engine import (
    generate_recommendations
)

from src.ml.random_forest_runner import (
    build_random_forest_pipeline,
    evaluate_binary_model
)

from src.ml.model_interpreter import (
    compare_model_results
)

from src.reporting.report_builder import (
    build_datadna_report
)

from src.reporting.report_formatter import (
    format_datadna_report
)


def run_datadna_analysis(
    df,
    target_column,
    positive_label="bad",
    negative_label="good",
    class_names=None,
    test_size=0.20,
    random_state=42,
    save_reports=True
):
    # 1. DataDNA profilini oluştur
    profile = create_datadna_profile(
        df,
        target_column=target_column,
        class_names=class_names
    )

    # 2. Veri yapısına göre öneriler üret
    recommendations = generate_recommendations(
        profile
    )

    # 3. Feature ve target ayrımı
    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # 4. Train-test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 5. Baseline Random Forest
    baseline_model = build_random_forest_pipeline(
        X_train,
        class_weight=None,
        random_state=random_state
    )

    baseline_model.fit(
        X_train,
        y_train
    )

    baseline_results = evaluate_binary_model(
        baseline_model,
        X_test,
        y_test,
        positive_label=positive_label,
        negative_label=negative_label
    )

    # 6. Balanced Random Forest
    balanced_model = build_random_forest_pipeline(
        X_train,
        class_weight="balanced",
        random_state=random_state
    )

    balanced_model.fit(
        X_train,
        y_train
    )

    balanced_results = evaluate_binary_model(
        balanced_model,
        X_test,
        y_test,
        positive_label=positive_label,
        negative_label=negative_label
    )

    # 7. Raporlarda gösterilecek sınıf adını belirle
    if class_names is not None:
        interpretation_label = class_names.get(
            positive_label,
            positive_label
        )
    else:
        interpretation_label = positive_label

    # 8. Model sonuçlarını karşılaştır
    model_interpretations = compare_model_results(
        baseline_results,
        balanced_results,
        positive_label=interpretation_label
    )

    # 9. Yapılandırılmış DataDNA raporunu oluştur
    report = build_datadna_report(
        profile,
        recommendations,
        baseline_results,
        balanced_results,
        model_interpretations
    )

    # 10. İnsan tarafından okunabilir rapora dönüştür
    formatted_report = format_datadna_report(
        report
    )

    # 11. İstenirse raporları dosyaya kaydet
    if save_reports:
        project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        reports_directory = (
            project_root
            / "reports"
        )

        reports_directory.mkdir(
            exist_ok=True
        )

        txt_path = (
            reports_directory
            / "datadna_analysis_report.txt"
        )

        json_path = (
            reports_directory
            / "datadna_analysis_report.json"
        )

        with open(
            txt_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(
                formatted_report
            )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=4
            )

    # 12. Bütün önemli sonuçları geri döndür
    result = {
        "report": report,
        "formatted_report": formatted_report,
        "baseline_model": baseline_model,
        "balanced_model": balanced_model
    }

    return result