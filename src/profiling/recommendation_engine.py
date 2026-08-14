def generate_recommendations(profile):
    recommendations = []

    if profile["missing_ratio"] > 0:
        recommendations.append(
            "Eksik değerler tespit edildi. "
            "Modelleme öncesinde eksik değer işleme yöntemleri değerlendirilmelidir."
        )
    else:
        recommendations.append(
            "Eksik değer problemi tespit edilmedi."
        )

    if profile["duplicate_ratio"] > 0:
        recommendations.append(
            "Tekrarlı kayıtlar tespit edildi. "
            "Modelleme öncesinde bu kayıtların incelenmesi önerilir."
        )
    else:
        recommendations.append(
            "Tekrarlı kayıt problemi tespit edilmedi."
        )

    if profile["outlier_ratio"] > 0:
        recommendations.append(
            "Sayısal değişkenlerde IQR yöntemine göre "
            "aykırı değer adayları bulunmaktadır. "
            "Bu değerler silinmeden önce değişkenlerin anlamı incelenmelidir."
        )

    class_balance_ratio = profile.get(
        "class_balance_ratio"
    )

    if (
        class_balance_ratio is not None
        and class_balance_ratio < 0.60
    ):
        recommendations.append(
            "Hedef değişkende sınıf dengesizliği tespit edildi. "
            "Model değerlendirilirken yalnızca accuracy kullanılmamalı; "
            "precision, recall ve F1-score değerleri de incelenmelidir. "
            "Class weighting gibi yöntemler denenebilir."
        )

    return recommendations