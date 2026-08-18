# DataDNA AI

DataDNA AI, tablosal veri setlerinin yapısal ve veri kalitesi
özelliklerini otomatik olarak analiz eden, makine öğrenmesi
deneyleri gerçekleştiren ve farklı veri setlerinden elde edilen
sonuçları karşılaştırabilen modüler bir veri bilimi projesidir.

## Projenin Amacı

Projenin temel amacı, bir veri setinin yalnızca satır ve sütun
sayısını göstermek yerine veri setine ait karakteristik özellikleri
bir "DataDNA" profili altında toplamaktır.

Sistem;

- veri setinin yapısal özelliklerini inceler,
- veri kalitesi problemlerini tespit eder,
- hedef değişkenin sınıf dağılımını analiz eder,
- veri yapısına göre öneriler üretir,
- Random Forest modellerini eğitir,
- model sonuçlarını karşılaştırır,
- sonuçları otomatik olarak yorumlar,
- veri setini dataset fingerprint ile temsil eder,
- farklı veri setlerini meta-dataset içerisinde karşılaştırır.

## Proje Mimarisi

DataDNA AI aşağıdaki temel bileşenlerden oluşmaktadır:

### Profiling

`src/profiling/dna_builder.py`

Veri setinin temel DataDNA profilini oluşturur.

Analiz edilen özellikler:

- gözlem sayısı
- özellik sayısı
- sayısal ve kategorik değişken oranları
- eksik değer oranı
- tekrarlı kayıt oranı
- IQR tabanlı aykırı değer adayları
- hedef değişken ve sınıf dağılımı
- sınıf denge oranı

### Recommendation Engine

`src/profiling/recommendation_engine.py`

DataDNA profilini kullanarak veri setine yönelik otomatik
öneriler üretir.

### Machine Learning

`src/ml/random_forest_runner.py`

Kategorik değişkenleri One-Hot Encoding yöntemiyle dönüştürür
ve Random Forest tabanlı sınıflandırma modellerini çalıştırır.

İki model karşılaştırılır:

- Baseline Random Forest
- Balanced Random Forest

Değerlendirme metrikleri:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Model Interpretation

`src/ml/model_interpreter.py`

Baseline ve Balanced Random Forest sonuçlarını karşılaştırarak
model davranışındaki değişimleri otomatik olarak yorumlar.

### Reporting

`src/reporting/`

DataDNA analiz sonuçlarını yapılandırılmış ve okunabilir raporlara
dönüştürür.

Üretilen rapor formatları:

- TXT
- JSON

### DataDNA Pipeline

`src/datadna_pipeline.py`

Profil oluşturma, öneri üretme, model eğitimi, değerlendirme,
yorumlama ve raporlama işlemlerini tek bir analiz akışı altında
birleştirir.

### Dataset Fingerprint

`src/profiling/fingerprint_builder.py`

Bir veri setinin tamamını temel meta-özelliklerden oluşan kompakt
bir fingerprint ile temsil eder.

### Meta-Dataset

`src/profiling/meta_dataset_builder.py`

Farklı veri setlerinden elde edilen fingerprint değerleri ile
makine öğrenmesi sonuçlarını aynı tablo içerisinde birleştirir.

Meta-dataset içerisinde bir satır tek bir veri kaydını değil,
bütün bir veri setini temsil eder.

### DataDNA Intelligence

`src/intelligence/meta_analyzer.py`

Birden fazla veri setinden elde edilen sonuçları birlikte analiz
ederek deneysel veri seti seviyesinde içgörüler üretir.

## Analiz Edilen Veri Setleri

Projede farklı yapısal özelliklere sahip dört veri seti kullanılmıştır:

| Veri Seti | Gözlem | Özellik | Sınıf Dengesi |
|---|---:|---:|---:|
| German Credit | 1000 | 20 | 0.4286 |
| Breast Cancer | 569 | 30 | 0.5938 |
| Iris Binary | 100 | 4 | 1.0000 |
| Digits Binary | 360 | 64 | 0.9780 |

## Meta-Analiz

Dataset fingerprint ve Random Forest sonuçları kullanılarak
veri seti seviyesinde bir meta-dataset oluşturulmuştur.

Mevcut dört deneyde sınıf denge oranı ile Balanced Random Forest
recall kazancı arasında yaklaşık -0.8713 Pearson korelasyonu
gözlenmiştir.

Bu sonuç yalnızca dört veri setine dayandığından genel bir
istatistiksel sonuç olarak değil, deneysel bir gözlem olarak
değerlendirilmektedir.

## Proje Yapısı

```text
DataDNA-AI/
│
├── notebooks/
│   └── 01_datadna_profiler.ipynb
│
├── src/
│   ├── profiling/
│   │   ├── dna_builder.py
│   │   ├── recommendation_engine.py
│   │   ├── fingerprint_builder.py
│   │   └── meta_dataset_builder.py
│   │
│   ├── ml/
│   │   ├── random_forest_runner.py
│   │   └── model_interpreter.py
│   │
│   ├── reporting/
│   │   ├── report_builder.py
│   │   └── report_formatter.py
│   │
│   ├── intelligence/
│   │   └── meta_analyzer.py
│   │
│   └── datadna_pipeline.py
│
├── reports/
├── data/
├── requirements.txt
└── README.md