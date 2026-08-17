def analyze_meta_dataset(meta_dataset):
    insights = []

    dataset_count = len(meta_dataset)

    insights.append(
        f"Meta analizde toplam {dataset_count} veri seti incelenmiştir."
    )

    if dataset_count < 5:
        insights.append(
            "Mevcut veri seti sayısı genellenebilir istatistiksel "
            "sonuçlar çıkarmak için sınırlıdır. Bulgular deneysel "
            "gözlemler olarak değerlendirilmelidir."
        )

    balanced_datasets = meta_dataset[
        meta_dataset["class_balance_ratio"] >= 0.90
    ]

    imbalanced_datasets = meta_dataset[
        meta_dataset["class_balance_ratio"] < 0.60
    ]

    if not balanced_datasets.empty:
        balanced_recall_gain = (
            balanced_datasets["recall_gain"].mean()
        )

        insights.append(
            "Tama yakın dengeli veri setlerinde ortalama recall "
            f"kazancı {balanced_recall_gain:.4f} olarak gözlenmiştir."
        )

    if not imbalanced_datasets.empty:
        imbalanced_recall_gain = (
            imbalanced_datasets["recall_gain"].mean()
        )

        insights.append(
            "Sınıf dengesi daha düşük veri setlerinde ortalama recall "
            f"kazancı {imbalanced_recall_gain:.4f} olarak gözlenmiştir."
        )

    correlation = meta_dataset[
        "class_balance_ratio"
    ].corr(
        meta_dataset["recall_gain"]
    )

    if correlation == correlation:
        insights.append(
            "Class balance ratio ile recall gain arasındaki Pearson "
            f"korelasyonu {correlation:.4f} olarak hesaplanmıştır."
        )

        if correlation < -0.50:
            insights.append(
                "Mevcut deneylerde sınıf dengesi arttıkça balanced "
                "Random Forest kaynaklı recall kazancının azalma "
                "eğiliminde olduğu gözlenmiştir."
            )

        elif correlation > 0.50:
            insights.append(
                "Mevcut deneylerde sınıf dengesi arttıkça recall "
                "kazancının da artma eğiliminde olduğu gözlenmiştir."
            )

        else:
            insights.append(
                "Mevcut deneylerde class balance ile recall gain "
                "arasında belirgin bir doğrusal eğilim gözlenmemiştir."
            )

    return insights