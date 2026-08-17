def compare_model_results(
    baseline_results,
    balanced_results,
    positive_label="bad"
):
    interpretations = []

    accuracy_change = (
        balanced_results["accuracy"]
        - baseline_results["accuracy"]
    )

    precision_change = (
        balanced_results["precision"]
        - baseline_results["precision"]
    )

    recall_change = (
        balanced_results["recall"]
        - baseline_results["recall"]
    )

    f1_change = (
        balanced_results["f1_score"]
        - baseline_results["f1_score"]
    )

    if recall_change > 0:
        interpretations.append(
            f"Balanced model, {positive_label} sınıfının recall "
            f"değerini {baseline_results['recall']:.4f} seviyesinden "
            f"{balanced_results['recall']:.4f} seviyesine yükseltti."
        )

    elif recall_change < 0:
        interpretations.append(
            f"Balanced model, {positive_label} sınıfının recall "
            "değerinde düşüşe neden oldu."
        )

    if accuracy_change < 0:
        interpretations.append(
            f"Balanced model kullanıldığında accuracy "
            f"{baseline_results['accuracy']:.4f} seviyesinden "
            f"{balanced_results['accuracy']:.4f} seviyesine düştü."
        )

    elif accuracy_change > 0:
        interpretations.append(
            "Balanced model genel accuracy değerini artırdı."
        )

    if precision_change < 0:
        interpretations.append(
            f"{positive_label} sınıfının precision değeri düşerken, "
            "model bu sınıfı yakalama konusunda daha hassas hale geldi."
        )

    elif precision_change > 0:
        interpretations.append(
            f"{positive_label} sınıfının precision değeri yükseldi."
        )

    if f1_change > 0:
        interpretations.append(
            f"{positive_label} sınıfının F1-score değeri "
            f"{baseline_results['f1_score']:.4f} seviyesinden "
            f"{balanced_results['f1_score']:.4f} seviyesine yükseldi."
        )

    elif f1_change < 0:
        interpretations.append(
            f"{positive_label} sınıfının F1-score değeri "
            f"{baseline_results['f1_score']:.4f} seviyesinden "
            f"{balanced_results['f1_score']:.4f} seviyesine düştü."
        )

    if recall_change > 0 and accuracy_change < 0:
        interpretations.append(
            "Sonuçlar bir performans dengesi (trade-off) olduğunu "
            "göstermektedir: azınlık sınıfını yakalama başarısı artarken "
            "genel doğruluk azalmıştır."
        )

    if not interpretations:
        interpretations.append(
            "Baseline ve Balanced Random Forest modelleri aynı performansı "
            "göstermiştir. Sınıflar dengeli olduğundan class weighting "
            "bu veri setinde model sonuçlarında değişiklik oluşturmamıştır."
        )

    return interpretations