# DeepSeis 🌍

**Transformer Tabanlı Zaman Serisi Modelleri ile Sismik Sinyallerde Anomali Tespiti ve Deprem Öncesi Davranış Analizi**

> Recep Tayyip Erdoğan Üniversitesi - Bilgisayar Mühendisliği Bitirme Tezi  
> Can Ahmedi Yaşar PARLAK | Danışman: Dr. Öğr. Üyesi Tuğçem PARTAL

---

## 📋 Proje Hakkında

DeepSeis, Transformer tabanlı derin öğrenme modellerini kullanarak sürekli sismik sinyallerdeki anomalileri tespit etmeyi ve bu anomalilerin deprem olaylarıyla ilişkisini istatistiksel olarak analiz etmeyi amaçlayan bir araştırma projesidir.

**⚠️ Not:** Bu çalışma deterministik deprem tahmini iddiası taşımamaktadır.

## 🏗️ Proje Yapısı

```
DeepSeis/
├── configs/
│   └── station_config.yaml          # Proje konfigürasyonu
├── data/
│   ├── raw/
│   │   ├── mseed/                   # Ham MiniSEED dosyaları
│   │   └── catalog/                 # Deprem kataloğu (CSV)
│   ├── interim/                     # Ara veriler
│   └── processed/
│       ├── windows/                 # Pencerelenmiş veriler (.npy)
│       └── stft/                    # Spektrogramlar (.npy)
├── src/
│   ├── data_ingestion/
│   │   ├── download_mseed.py        # FDSN sismik veri indirme
│   │   └── download_catalog.py      # Deprem kataloğu indirme
│   ├── preprocessing/
│   │   ├── read_mseed.py            # MiniSEED okuma ve doğrulama
│   │   ├── filter_and_window.py     # Filtreleme ve pencereleme
│   │   └── normalize.py             # Z-score normalizasyonu
│   ├── features/
│   │   ├── spectral_features.py     # STFT spektral öznitelikler
│   │   └── time_domain_features.py  # Zaman alanı öznitelikleri
│   ├── visualization/
│   │   ├── plot_waveforms.py        # Dalga formu grafikleri
│   │   └── plot_spectrograms.py     # Spektrogram grafikleri
│   └── utils/
│       └── labeling.py              # Pencere etiketleme
├── notebooks/                       # Jupyter notebook'lar
├── artifacts/
│   ├── figures/                     # Üretilen görseller
│   ├── logs/                        # Eğitim logları
│   └── reports/                     # Raporlar
├── requirements.txt                 # Python bağımlılıkları
└── README.md
```

## 🚀 Kurulum

```bash
# Sanal ortam oluştur ve aktifleştir
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows PowerShell

# Bağımlılıkları kur
pip install -r requirements.txt
```

## 📡 Sprint 1 - Veri Hattı Kullanımı

### 1. Deprem kataloğu indir
```bash
python -m src.data_ingestion.download_catalog
python -m src.data_ingestion.download_catalog --test   # 1 aylık test
```

### 2. Sismik veri indir
```bash
python -m src.data_ingestion.download_mseed
python -m src.data_ingestion.download_mseed --test     # 1 günlük test
```

### 3. Veri ön işleme
```bash
python -m src.preprocessing.read_mseed --test
```

## 🔬 Kullanılan Teknolojiler

| Kategori | Teknoloji |
|---|---|
| Dil | Python 3.10 |
| Sismik Veri | ObsPy |
| Veri İşleme | NumPy, Pandas, SciPy |
| Görselleştirme | Matplotlib |
| Makine Öğr. | Scikit-learn |
| Derin Öğrenme | PyTorch *(Sprint 2)* |

## 📊 Metodoloji

1. **Veri Toplama**: FDSN/KOERI üzerinden MiniSEED sismik kayıtlar
2. **Ön İşleme**: Butterworth bant-geçiren filtre (1-45 Hz) → Pencereleme (60s, %50 örtüşme)
3. **Öznitelik Çıkarma**: STFT spektrogramlar + zaman alanı özellikleri
4. **Modelleme**: TST, Informer, TFT (Transformer tabanlı) *(Sprint 2+)*
5. **Değerlendirme**: Anomali skoru ↔ Deprem kataloğu korelasyonu
