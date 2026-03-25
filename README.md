# YZM304 Derin Ogrenme Proje Modulu I

## Introduction
Bu repo, `YZM304_Proje_Odevi1_2526.pdf` icin tekrar uretilebilir bir ikili siniflandirma calismasi sunar. Veri kaynagi olarak `sklearn.datasets.load_breast_cancer` kullanilir. Calisma, laboratuvarda kurulan iki katmanli temel modelden baslayip veri on isleme, mimari iyilestirme, veri miktari etkisi, reglarizasyon ve farkli kutuphane tekrarlarini tek bir komutla ureten kanonik bir akis kurar.

## Methods
### Ortam
- Python: `3.12`
- Sanal ortam: `py -3.12 -m venv .venv`
- Kurulum: `.\bootstrap.ps1`
- Calistirma: `.\run_project.ps1` veya `.\.venv\Scripts\python.exe -m src.run_all`
- Test: `.\.venv\Scripts\python.exe -m pytest`

### Veri ve Bolme Stratejisi
- Veri seti: `sklearn.datasets.load_breast_cancer`
- Ornek sayisi: `569`
- Ozellik sayisi: `30`
- Siniflar: `malignant (0)`, `benign (1)`
- Sinif dagilimi: `212 malignant (%37.3)`, `357 benign (%62.7)`
- Train / validation / test bolmesi: `%60 / %20 / %20`
- Gercek bolme sayilari: `341 train`, `114 validation`, `114 test`
- Tum ayirmalar `seed=42` ile stratified olarak uretilir.

### Modeller ve Hiperparametreler
- Kayip fonksiyonu: binary cross entropy
- Optimizasyon: full-batch SGD
- Karar esigi: `0.5`
- Global tekrar uretilebilirlik tohumu: `42`
- Agirlik baslatma semasi:
  - Sigmoid gizli katmanlar ve cikis katmani icin `N(0, sqrt(2 / (fan_in + fan_out)))`
  - ReLU gizli katmanlar icin `N(0, sqrt(2 / fan_in))`
  - Tum bias degerleri `0`
- Full-batch'in sayisal karsiligi:
  - Tam train bolmesi icin `batch_size=341`
  - `%75` train varyanti icin `batch_size=255`
  - `%50` train varyanti icin `batch_size=170`
  - Bu nedenle `1 epoch = 1 optimizer update` ve `n_steps = epochs`
- Dengesiz sinif dagilimi nedeniyle model secimi: validation balanced accuracy azalan, `n_steps` artan, parametre sayisi artan, validation ROC-AUC azalan sira
- NumPy deneyleri:
  - `baseline_raw`: `30-8-1`, sigmoid, ham veri, `lr=0.08`, `epochs=1200`
  - `baseline_scaled`: `30-8-1`, sigmoid, standardizasyon, `lr=0.08`, `epochs=1200`
  - `wide_scaled`: `30-16-1`, sigmoid, standardizasyon, `lr=0.05`, `epochs=1400`
  - `deep_scaled_no_l2`: `30-32-16-1`, relu, standardizasyon, `lr=0.01`, `epochs=1400`
  - `deep_scaled_l2`: `30-32-16-1`, relu, standardizasyon, `lr=0.01`, `epochs=1400`, `L2=1e-3`
  - `deep_scaled_l2_data50`: ayni derin model, train verisinin `%50`'si
  - `deep_scaled_l2_data75`: ayni derin model, train verisinin `%75`'i
  - `deep_scaled_l2_data100`: ayni derin model, train verisinin `%100`'u
- Backend tekrarlar:
  - NumPy referansi
  - `scikit-learn MLPClassifier`
  - PyTorch `nn.Module`
- Backend bazli egitim ayrintilari:
  - NumPy: deterministik full-batch SGD, `shuffle=False`, `momentum=0`, L2 ceza terimi loss icine eklenir
  - sklearn: `solver='sgd'`, `batch_size=train_size`, `shuffle=False`, `momentum=0.0`, `nesterovs_momentum=False`, `max_iter=1`, `warm_start=True`, epoch dongusu `partial_fit`
  - PyTorch: `torch.optim.SGD(lr=...)`, `momentum=0`, `weight_decay=L2`, `torch.manual_seed(42)`, `torch.use_deterministic_algorithms(True)`
- Ortak sabitler:
  - baslangic agirliklari `data/weights/*.npz`
  - optimizer ailesi `SGD`
  - split manifesti `data/splits/split_manifest.json`

### Repo Yapisi
- `src/`: veri hazirlama, NumPy MLP, sklearn adapter, PyTorch model ve raporlama kodlari
- `data/`: ham veri disa aktarimi, split manifestleri ve baslangic agirliklari
- `outputs/`: tum tablolar, confusion matrix'ler, egitim egrileri ve ozet raporlar
- `tests/`: tekrar uretilebilirlik ve egitim dogrulama testleri

## Results
Tum sonuc tablolari `outputs/tables/`, gorseller `outputs/figures/`, ozet raporlar `outputs/reports/` altinda uretilir. Bu calistirmada model secim tablosu `outputs/tables/model_selection.csv`, backend karsilastirmasi `outputs/tables/backend_comparison_metrics.csv`, secilen model raporu ise `outputs/reports/selected_model_report.md` olarak uretildi.

### NumPy Deney Ozetleri
| Deney | Val balanced acc | Val accuracy | Test accuracy | n_steps | Yorum |
| --- | --- | --- | --- | --- | --- |
| `baseline_raw` | `0.5000` | `0.6228` | `0.6316` | `1200` | Ham ozelliklerle belirgin underfitting goruldu. |
| `baseline_scaled` | `0.9581` | `0.9649` | `0.9649` | `1200` | Standardizasyon tek basina ciddi iyilestirme sagladi. |
| `wide_scaled` | `0.9651` | `0.9737` | `0.9737` | `1400` | En iyi balanced accuracy grubunda ve ayni adim sayisinda en dusuk parametreli model. |
| `deep_scaled_no_l2` | `0.9581` | `0.9649` | `0.9561` | `1400` | Derin mimari testte genis modele gore geride kaldi. |
| `deep_scaled_l2` | `0.9581` | `0.9649` | `0.9561` | `1400` | L2, accuracy'i degistirmedi fakat agirlik normunu hafif dusurdu. |
| `deep_scaled_l2_data50` | `0.9651` | `0.9737` | `0.9649` | `1400` | Balanced accuracy'de secilen modelle esit, fakat daha fazla parametre kullaniyor. |
| `deep_scaled_l2_data75` | `0.9651` | `0.9737` | `0.9474` | `1400` | Balanced accuracy'de esit, testte daha zayif genelledi. |
| `deep_scaled_l2_data100` | `0.9581` | `0.9649` | `0.9561` | `1400` | Derin modelin tam veri surumu. |

### Secilen Model
- Secilen model: `wide_scaled`
- Mimari: `30-16-1`
- Hidden activation: `sigmoid`
- Preprocessing: `StandardScaler`
- Validation balanced accuracy: `0.9651`
- Validation accuracy: `0.9737`
- Test accuracy: `0.9737`
- Test F1: `0.9790`
- Test ROC-AUC: `0.9954`

Sinif dagilimi dengeli olmadigi icin secim kurali validation balanced accuracy > dusuk `n_steps` > dusuk parametre sayisi > validation ROC-AUC olarak uygulandi. `wide_scaled`, `deep_scaled_l2_data50` ve `deep_scaled_l2_data75` validation balanced accuracy tarafinda esitlenmesine ragmen ayni adim sayisinda daha az parametre kullandigi icin secildi.

### Backend Karsilastirmasi
| Mimari | NumPy test acc | sklearn test acc | PyTorch test acc | NumPy test balanced acc |
| --- | --- | --- | --- | --- |
| `30-8-1` | `0.9649` | `0.9649` | `0.9649` | `0.9673` |
| `30-16-1` | `0.9737` | `0.9737` | `0.9737` | `0.9742` |
| `30-32-16-1` | `0.9561` | `0.9561` | `0.9561` | `0.9554` |

Bu tablo, ayni split, ayni baslangic agirliklari ve ayni SGD ayarlari ile backend tekrarlarinin esdeger sonuc verdigini gosterir. Confusion matrix gorselleri `outputs/figures/confusion_matrix_*` dosyalari olarak kaydedildi.

## Discussion
Calisma iki temel sonucu ortaya koydu. Birincisi, laboratuvar modeli icin en kritik iyilestirme veri on islemedir; ham veriyle kurulan `baseline_raw` modeli `0.62` validation accuracy etrafinda kalirken standardizasyon sonrasi ayni mimari `0.96+` duzeyine cikti. Ikincisi, daha derin mimari her zaman daha iyi model vermedi; bu veri setinde genis ama tek gizli katmanli `30-16-1` modeli, daha derin `30-32-16-1` mimarisinden daha dengeli genelleme sagladi.

L2 reglarizasyon bu kurulumda accuracy tarafinda belirgin bir artis getirmedi, ancak agirlik normunu `10.1610` seviyesinden `10.1606` seviyesine indirerek parametre buyuklugunu kontrol etti. Veri miktari deneyleri, `%50` ve `%75` train alti-kumelerinin validation tarafinda rekabetci kalabildigini; fakat test genellemesinde tam veriyle secilen genis modelin daha istikrarli oldugunu gosterdi.

Repo tek kanonik akisi korur; eski durumlarla uyumluluk katmani eklemez. Temiz checkout sonrasinda `.venv` kurulumu, testlerin calistirilmasi ve `python -m src.run_all` komutuyla tum ciktilar yeniden uretilebilir.

# Alperen Aydın 22290435
