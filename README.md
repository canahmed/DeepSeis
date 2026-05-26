# DeepSeis

Makine ogrenmesi tabanli sismik anomali tespiti ve olay cevresi davranis analizi projesi.

Bu calisma Recep Tayyip Erdogan Universitesi Bilgisayar Muhendisligi Bitirme Tezi kapsaminda gelistirilmistir.

**Ogrenci:** Can Ahmedi Yasar PARLAK

**Danisman:** Dr. Ogr. Uyesi Tugcem PARTAL

## Proje Ozeti

DeepSeis, surekli sismik kayitlari kisa zaman pencerelerine ayirarak her pencere icin zaman-frekans temsili ureten, bu temsillerden ozet STFT oznitelikleri cikaran ve normal/anomali ayrimini makine ogrenmesi modelleriyle degerlendiren bir arastirma prototipidir.

Calisma deterministik deprem tahmini iddiasi tasimaz. Nihai deney, Mw >= 4.0 deprem olaylarinin +-60 dakika cevresindeki pencereleri "anomali", ayni buyuklukteki olaylardan en az 48 saat uzak pencereleri ise "temiz normal" olarak tanimlar. Bu nedenle proje, depremi kesin olarak onceden tahmin eden bir sistem degil; olay cevresi sismik pencereler ile olaylardan uzak normal pencereler arasindaki ayirt edilebilir zaman-frekans farklarini inceleyen bir prototiptir.

## Guncel Sonuc

Nihai deney protokolu:

- Kanal: `HHZ`
- Pozitif sinif: Mw >= 4.0 olaylarinin +-60 dakika cevresi
- Negatif sinif: Mw >= 4.0 olaylardan en az 48 saat uzak pencereler
- Oznitelik seti: 51 boyutlu STFT ozet oznitelikleri
- En iyi model: `HistGradientBoostingClassifier`
- Model dosyasi: `artifacts/models/mw40_pm60m_hhz_clean_gap48h_hist_gradient.joblib`

Test metrikleri:

| Metrik | Deger |
|---|---:|
| Accuracy | 0.8124 |
| Precision | 0.8066 |
| Recall | 0.8219 |
| F1-Score | 0.8142 |
| ROC-AUC | 0.8979 |
| PR-AUC | 0.9090 |

Karar esigi dogrulama kumesinde F1 skorunu en iyi yapan deger olarak secilmistir: `0.4433`.

## Proje Yapisi

```text
DeepSeis/
|-- app.py                         # Flask tabanli arayuz ve model servisi
|-- configs/                       # Istasyon ve deney ayarlari
|-- data/
|   `-- raw/catalog/               # Deprem katalog dosyasi
|-- scripts/                       # Deney, analiz ve model egitim scriptleri
|-- src/
|   |-- data_ingestion/            # Katalog ve MiniSEED veri indirme araclari
|   |-- preprocessing/             # Okuma, filtreleme, pencereleme, STFT hazirligi
|   |-- features/                  # Sinyal ve spektral oznitelik fonksiyonlari
|   |-- models/                    # Derin ogrenme ve feature tabanli model kodlari
|   |-- utils/                     # Etiketleme ve bolme yardimcilari
|   `-- visualization/             # Dalga formu ve spektrogram gorsellestirme
|-- artifacts/
|   |-- figures/                   # Deney gorselleri
|   |-- reports/                   # Metrik ve analiz raporlari
|   `-- models/                    # Nihai kucuk model dosyasi
|-- templates/                     # Web arayuzu HTML sablonu
|-- tez/                           # LaTeX tez raporu ve tez gorselleri
|-- requirements.txt               # Python bagimliliklari
`-- README.md
```

Buyuk ham veri, islenmis `.npy` dosyalari, ara oznitelikler, checkpoint dosyalari ve buyuk model dosyalari GitHub'a yuklenmez. Bu dosyalar `.gitignore` ile disarida tutulur.

## Kurulum

Windows PowerShell icin:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Arayuzu Calistirma

Nihai feature tabanli modeli Flask arayuzunde calistirmak icin:

```powershell
$env:DEEPSEIS_BACKEND="feature"
$env:DEEPSEIS_MODEL="feature_hist_gradient"
$env:DEEPSEIS_FEATURE_MODEL="artifacts/models/mw40_pm60m_hhz_clean_gap48h_hist_gradient.joblib"
$env:DEEPSEIS_SPLITS_DIR="data/processed/experiments/mw40_pm60m_hhz"
$env:PORT="5000"
python app.py
```

Arayuz:

```text
http://127.0.0.1:5000
```

Durum kontrolu:

```text
http://127.0.0.1:5000/status
```

Not: Arayuzun test pencerelerini okuyabilmesi icin `data/processed/experiments/mw40_pm60m_hhz` altindaki islenmis deney dosyalarinin yerel makinede bulunmasi gerekir. Bu buyuk dosyalar GitHub'a dahil edilmemistir.

## Temel Is Akisi

1. Sismik MiniSEED kayitlari ve deprem kataloglari hazirlanir.
2. Sinyaller kisa zaman pencerelerine ayrilir.
3. Her pencere STFT ile zaman-frekans uzayina tasinir.
4. STFT temsillerinden 51 boyutlu ozet oznitelik vektoru uretilir.
5. Farkli etiketleme protokolleri altinda modeller egitilir ve karsilastirilir.
6. Nihai HistGradientBoosting modeli `joblib` formatinda kaydedilir.
7. Model Flask arayuzune baglanarak pencere bazli normal/anomali skoru uretir.

## Kullanilan Bilesenler

| Kategori | Teknoloji |
|---|---|
| Dil | Python |
| Sismik veri | ObsPy |
| Sinyal isleme | NumPy, SciPy |
| Veri analizi | Pandas |
| Modelleme | Scikit-learn, PyTorch |
| Gorsellestirme | Matplotlib |
| Web arayuzu | Flask |
| Tez raporu | LaTeX |

## Onemli Scriptler

| Dosya | Amac |
|---|---|
| `scripts/train_clean_feature_model.py` | Temiz normal/anomali protokolu icin feature tabanli modelleri egitir |
| `scripts/train_feature_model.py` | Kronolojik ve alternatif feature deneylerini calistirir |
| `scripts/evaluate_checkpoint.py` | Kaydedilmis derin ogrenme checkpoint'lerini degerlendirir |
| `scripts/evaluate_baselines.py` | Baseline metrikleri uretir |
| `scripts/audit_project.py` | Proje raporlarini ve deney durumunu ozetler |
| `scripts/audit_channels.py` | Kanal/veri uygunlugunu inceler |
| `scripts/analyze_labeling_matrix.py` | Etiketleme protokollerini karsilastirir |

## Tez Raporu

LaTeX tez dosyalari `tez/` klasorundedir. Rapor, proje ile uyumlu olacak sekilde guncellenmistir:

- Literatur ve yontem bolumleri STFT + makine ogrenmesi yaklasimina gore duzenlendi.
- Bulgular bolumunde kronolojik deneylerin sinirlari ve nihai temiz protokol sonuclari ayrildi.
- Nihai metrikler, ROC/PR egri gorselleri, karisiklik matrisi ve model karsilastirma grafigi eklendi.
- Sonuc bolumu deterministik deprem tahmini iddiasi tasimayacak bicimde yazildi.

## Akademik Yorum

DeepSeis'in en guclu ve savunulabilir sonucu, olay cevresi anomalileri ile olaylardan uzak temiz normal pencereler arasinda ayirt edilebilir zaman-frekans oruntuleri bulunmasidir. Kronolojik deprem oncesi tahmin deneylerinde ise sinif dengesizligi ve dagilim kaymasi nedeniyle F1 ve precision metrikleri sinirli kalmistir. Bu nedenle calisma, "kesin deprem tahmini" olarak degil, "sismik anomali ve olay cevresi davranis analizi" olarak yorumlanmalidir.

## Lisans ve Kullanim Notu

Bu repo akademik arastirma ve bitirme tezi kapsaminda hazirlanmistir. Buyuk veri dosyalari ve ara egitim ciktlari GitHub'a dahil edilmemistir; deneylerin birebir tekrar edilebilmesi icin ilgili ham/islenmis veri dosyalarinin yerel ortamda bulunmasi gerekir.
