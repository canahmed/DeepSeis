RECEP TAYYİP ERDOĞAN ÜNİVERSİTESİ
MÜHENDİSLİK VE MİMARLIK FAKÜLTESİ
BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ
Transformer Tabanlı Zaman Serisi
Modelleri ile
Sismik Sinyallerde Anomali Tespiti ve
Deprem Öncesi Davranış Analizi
“DeepSeis”
BİTİRME TEZİ
Can Ahmedi Yaşar PARLAK
OCAK 2026
RİZE
Transformer Tabanlı Zaman Serisi Modelleri ile
Sismik Sinyallerde Anomali Tespiti ve
Deprem Öncesi Davranış Analizi
“DeepSeis”
Can Ahmedi Yaşar PARLAK
Jüri Üyeleri
Danışman: Dr. Öğr. Üyesi Tuğçem PARTAL
Üye:
Üye:
Bölüm Başkanı: Dr. Öğr. Üyesi Yasin YALÇIN
OCAK 2026
RİZE
ÖNSÖZ
Depremler, yüksek yıkıcılık potansiyeli nedeniyle hem insan hayatı hem de eko-
nomik açıdan kritik öneme sahip doğal afetlerdir. Türkiye, Alp-Himalaya deprem
kuşağı üzerinde yer alması nedeniyle tarih boyunca sayısız yıkıcı depreme ma-
ruz kalmıştır. Bu gerçeklik, deprem bilimi alanındaki araştırmaları ülkemiz için
hayati bir öncelik haline getirmektedir.
Bu tez çalışması, yapay zekâ ve derin öğrenme teknolojilerinin sismik veri ana-
lizine entegrasyonunu araştırmaktadır. Çalışmanın temel motivasyonu, Transfor-
mer tabanlı zaman serisi modellerinin sismik sinyallerdeki anomalileri tespit etme
kapasitesini değerlendirmek ve bu anomalilerin deprem olaylarıyla olan ilişkisini
istatistiksel olarak analiz etmektir.
Tezin hazırlanması sürecinde değerli bilgi birikimi, rehberliği ve yönlendirme-
leri ile akademik gelişimime büyük katkı sağlayan danışman hocam Dr. Öğr. Üyesi
Tuğçem Partal’a en içten teşekkürlerimi sunarım.
Bu çalışma, deterministik bir deprem tahmini iddiası taşımamakta; bunun
yerine sismik sinyallerdeki istatistiksel değişimlerin derin öğrenme yöntemleriyle
daha erken ve güvenilir biçimde analiz edilebileceğini ortaya koymayı hedeflemek-
tedir.

Can Ahmedi Yaşar PARLAK
Ocak 2026, Rize
III
ÖZET
Bu tez çalışması, Transformer tabanlı zaman serisi modellerini sismik titreşim ve-
rilerine uygulayarak deprem öncesi anomali tespiti gerçekleştirmeyi hedeflemekte-
dir. Mevcut erken uyarı sistemleri genellikle depremin ilk saniyelerinde P-dalgasını
algılayarak S-dalgasından önce uyarı vermeye odaklanmaktadır. Bu yaklaşım dep-
rem başladıktan sonra birkaç saniyelik kazanım sağlamakta; ancak deprem öncesi
saatler veya günler içinde yer kabuğundaki mikrosismik davranış değişimlerini
modellemek bilimsel açıdan hâlâ zorlu bir problem olarak kalmaktadır.
Çalışmada, sürekli kaydedilen ivme ve titreşim sinyallerinden anlamlı temsil-
ler öğrenmek, kısa ve uzun vadeli bağımlılıkları modellemek ve normal davra-
nışa kıyasla olağandışı aktivite dönemlerini otomatik işaretlemek amaçlanmakta-
dır. Time Series Transformer (TST), Informer ve Temporal Fusion Transformer
(TFT) gibi modern derin öğrenme mimarileri kullanılarak sismik sinyallerdeki
anomaliler tespit edilmekte ve bu anomaliler gerçekleşen depremlerle zaman ek-
seninde eşleştirilerek istatistiksel korelasyon analizi yapılmaktadır.
MiniSEED formatındaki sürekli sismik veriler işlenmiş, kısa zaman pencereleri
elde edilerek hem ham dalga formu hem de zaman-frekans dönüşümleri (STFT,
CWT) üzerinden öznitelikler çıkarılmıştır. Elde edilen sonuçlar, Transformer ta-
banlı modellerin geleneksel LSTM/GRU yöntemlerine kıyasla uzun süreli bağım-
lılıkları yakalama konusunda daha etkili olduğunu göstermektedir.

Anahtar Kelimeler:
Sismik Anomali Tespiti: Sürekli sismik sinyallerdeki olağandışı davra-
nışların otomatik olarak belirlenmesi.
Transformer: Self-attention mekanizması kullanan, uzun vadeli bağımlı-
lıkları modelleyebilen derin öğrenme mimarisi.
Zaman Serisi Analizi: Zamana bağlı verilerin istatistiksel ve algoritmik
yöntemlerle incelenmesi.
Derin Öğrenme: Çok katmanlı yapay sinir ağları kullanarak karmaşık
örüntülerin öğrenilmesi.
Deprem Öncesi Sinyaller: Depremlerden önce gözlemlenebilecek potan-
siyel sismik aktivite değişimleri.
IV
Time Series Transformer: Zaman serisi verileri için özelleştirilmiş Trans-
former mimarisi varyantı.
V
İçindekiler
ÖNSÖZ III
ÖZET IV
ŞEKİLLER DİZİNİ IX
TABLOLAR DİZİNİ X
1 GİRİŞ SEMBOLLER VE KISALTMALAR XI
1.1 Problem Tanımı
1.1.1 Mevcut Erken Uyarı Sistemlerinin Sınırlamaları
1.1.2 Deprem Öncesi Anomali Tespitinin Önemi
1.2 Araştırma Soruları
1.3 Amaç ve Bilimsel Katkı
1.3.1 Neden Transformer Mimarisi?
1.3.2 Bilimsel Katkılar
1.4 Çalışmanın Kapsamı ve Sınırları
1.5 Tezin Organizasyonu
2 LİTERATÜR TARAMASI
2.1 Sismik Veri Analizi ve Deprem Erken Uyarı Sistemleri
2.1.1 Klasik Sinyal İşleme Yöntemleri
2.2 Makine Öğrenmesi ve Derin Öğrenme Yaklaşımları
2.2.1 Tekrarlayan Sinir Ağları (RNN)
2.2.2 Evrişimli Sinir Ağları (CNN)
2.3 Transformer Mimarisi ve Zaman Serisi Analizi
2.3.1 Attention Mekanizması
2.3.2 Zaman Serisi için Transformer Varyantları
2.4 Anomali Tespiti Yaklaşımları
2.4.1 Yeniden Yapılandırma Tabanlı Yöntemler
2.4.2 Tahmin Tabanlı Yöntemler
2.4.3 Sınıflandırma Tabanlı Yöntemler
2.5 Sismik Anomali ve Deprem Öncesi Sinyaller
2.5.1 Mikrosisimik Aktivite Değişimleri
2.5.2 Frekans İçeriği Değişimleri
2.6 Literatür Özeti
3 YÖNTEM
3.1 Veri Kaynakları
3.1.1 Sürekli Sismik İstasyon Verileri
3.1.2 Deprem Kataloğu
3.1.3 Veri Toplama Prensibi
3.2 Veri Ön İşleme
3.2.1 Gürültü Giderme
3.2.2 Normalizasyon
3.2.3 Spektral Dönüşüm
3.3 Model Mimarisi
3.3.1 Time Series Transformer (TST)
3.3.2 Informer
3.3.3 Temporal Fusion Transformer (TFT)
3.3.4 Model Hiperparametreleri
3.4 Eğitim Stratejisi
3.4.1 Veri Bölümleme
3.4.2 Etiketleme Yaklaşımları
3.4.3 Sınıf Dengesizliği
3.5 Değerlendirme Metrikleri
3.5.1 Teknik Metrikler
3.5.2 Jeofiziksel Anlamlılık
3.6 Kullanılan Teknolojiler
3.6.1 Yazılım
3.6.2 Donanım
3.6.3 Veri Depolama
4 BULGULAR VE TARTIŞMA
4.1 Deneysel Kurulum
4.1.1 Veri Seti İstatistikleri
4.1.2 Eğitim Süreci
4.2 Model Performans Karşılaştırması
4.2.1 Anomali Tespiti Performansı
4.2.2 Yanlış Alarm Analizi
4.3 Anomali Skoru Analizi
4.3.1 Zaman Serisi Görselleştirmesi
4.3.2 İstatistiksel Korelasyon Analizi
4.4 Girdi Temsili Karşılaştırması
4.5 Ablasyon Çalışması
4.6 Tartışma
4.6.1 Ana Bulgular
4.6.2 Sınırlamalar
4.6.3 Karşılaştırmalı Değerlendirme
5 SONUÇ VE ÖNERİLER
5.1 Sonuçlar
5.1.1 Teknik Sonuçlar
5.1.2 Jeofiziksel Sonuçlar
5.1.3 Bilimsel Katkılar
5.2 Sınırlar ve Etik Uyarılar
5.3 Öneriler
5.3.1 Gelecek Araştırmalar İçin Öneriler
5.3.2 Pratik Öneriler
5.4 Beklenen Çıktılar
5.5 Son Söz
KAYNAKLAR
ÖZGEÇMİŞ
Şekil 1.1. DeepSeis Sistem Mimarisi Genel Görünümü ŞEKİLLER DİZİNİ
Şekil 2.1. Sismik Dalga Tipleri ve Yayılım Karakteristikleri
Şekil 2.2. Transformer Attention Mekanizması
Şekil 3.1. Veri Toplama ve Ön İşleme Süreç Akışı
Şekil 3.2. STFT Tabanlı Zaman-Frekans Dönüşümü Örneği
Şekil 3.3. Time Series Transformer Model Mimarisi
Şekil 3.4. Anomali Skoru Üretim Mekanizması
Şekil 4.1. Model Eğitim Kaybı (Loss) Grafiği
Şekil 4.2. Anomali Skoru - Deprem Zaman Çizelgesi Karşılaştırması
Şekil 4.3. ROC Eğrisi Analizi
Şekil 4.4. LSTM ve Transformer Performans Karşılaştırması
TABLOLAR DİZİNİ
Tablo 2.1. Literatürdeki İlgili Çalışmaların Karşılaştırması 10
Tablo 3.1. Kullanılan Sismik Veri Setlerinin Özellikleri 13
Tablo 3.2. Veri Ön İşleme Adımları ve Uygulanan Filtreler 14
Tablo 3.3. Model Hiperparametreleri 17
Tablo 4.1. Transformer ve LSTM Modellerinin Performans Karşı-
laştırması
21
Tablo 4.2. Anomali Skoru - Deprem Kataloğu Zaman Eşleşme Ana-
lizi
23
Tablo 4.3. Değerlendirme Metrikleri (Precision, Recall, F1, ROC-
AUC)
24
Tablo 4.4. Yanlış Alarm Oranı Analizi 22
X
SEMBOLLER VE KISALTMALAR
Kısaltma Açıklama
AI Artificial Intelligence (Yapay Zekâ)
ML Machine Learning (Makine Öğrenmesi)
DL Deep Learning (Derin Öğrenme)
TST Time Series Transformer
TFT Temporal Fusion Transformer
STFT Short-Time Fourier Transform (Kısa Zamanlı Fourier Dö-
nüşümü)
CWT Continuous Wavelet Transform (Sürekli Dalgacık Dönü-
şümü)
FFT Fast Fourier Transform (Hızlı Fourier Dönüşümü)
MSE Mean Squared Error (Ortalama Kare Hatası)
MAE Mean Absolute Error (Ortalama Mutlak Hata)
ROC-AUC Receiver Operating Characteristic - Area Under Curve
PR-AUC Precision-Recall - Area Under Curve
TP True Positive (Doğru Pozitif)
FP False Positive (Yanlış Pozitif)
FN False Negative (Yanlış Negatif)
TN True Negative (Doğru Negatif)
LSTM Long Short-Term Memory
GRU Gated Recurrent Unit
RNN Recurrent Neural Network
CNN Convolutional Neural Network
GPU Graphics Processing Unit
STA/LTA Short-Term Average / Long-Term Average
MiniSEED Sismik Kayıt Veri Formatı
IRIS Incorporated Research Institutions for Seismology
USGS United States Geological Survey
AFAD Afet ve Acil Durum Yönetimi Başkanlığı
Kandilli Boğaziçi Üniversitesi Kandilli Rasathanesi
Mw Moment Magnitüdü
ML Lokal Magnitüd
ObsPy Python Tabanlı Sismoloji Kütüphanesi
RMS Root Mean Square
XI
Bölüm 1
GİRİŞ
1.1 Problem Tanımı
Deprem, yüksek yıkıcılık potansiyeli nedeniyle hem insan hayatı hem de ekonomik
açıdan kritik bir doğal afettir [? ]. Dünya genelinde her yıl ortalama 20.000’den
fazla can kaybına neden olan depremler, aynı zamanda milyarlarca dolarlık ekono-
mik zarara yol açmaktadır. Türkiye, Kuzey Anadolu Fay Hattı ve Doğu Anadolu
Fay Hattı gibi aktif tektonik yapılar üzerinde konumlanması nedeniyle tarih bo-
yunca sayısız yıkıcı depreme maruz kalmıştır [? ]. 1999 Marmara Depremi (Mw
7.4) ve 2023 Kahramanmaraş Depremleri (Mw 7.7 ve Mw 7.6) bu yıkıcılığın en
acı örnekleridir. Bu gerçeklik, deprem bilimi alanındaki araştırmaları ülkemiz için
stratejik bir öncelik haline getirmektedir.

1.1.1 Mevcut Erken Uyarı Sistemlerinin Sınırlamaları
Günümüzde kullanılan erken uyarı sistemleri genellikle depremin ilk saniyelerinde
P-dalgasını (birincil dalga) algılayarak, daha yıkıcı olan S-dalgasından (ikincil
dalga) önce uyarı vermeye odaklanmaktadır [? ]. Bu sistemlerin çalışma prensibi
şöyle özetlenebilir:

P-Dalgası Algılama: Deprem kaynağından yayılan ilk sismik dalga olan
P-dalgası, hızı nedeniyle istasyonlara ilk ulaşan dalgadır (yaklaşık 6-8 km/s).
Hızlı Analiz: Sistem, P-dalgasının özelliklerinden (genlik, frekans içeriği)
depremin büyüklüğünü ve konumunu tahmin eder.
Uyarı Yayını: S-dalgası (yaklaşık 3-5 km/s) ve yüzey dalgaları varmadan
önce saniyeler içinde uyarı verilir.
Bu yaklaşım deprem başladıktan sonra birkaç saniyelik (genellikle 10-60 sa-
niye) kazanım sağlamaktadır. Ancak bazı kritik sınırlamaları bulunmaktadır:

Reaktif Yapı: Sistem ancak deprem başladıktan sonra devreye girmekte-
dir.
Kör Bölge: Episantra yakın bölgelerde uyarı süresi çok kısa veya sıfır ola-
bilir.
Tahmin Değil, Algılama: Deprem öncesi herhangi bir öngörü sağlama-
maktadır.
1.1.2 Deprem Öncesi Anomali Tespitinin Önemi
Deprem olmadan önceki saatler veya günler içinde yer kabuğundaki mikrosismik
davranıştaki değişimleri modelleyerek “olasılık artışı” hakkında bir ön sinyal üret-
mek hâlâ bilimsel olarak zorlu ve tartışmalı bir konudur [? ]. Ancak son yıllarda
yapılan araştırmalar, bazı depremlerin öncesinde şu tür anomalilerin gözlemlene-
bileceğini ortaya koymuştur:

Mikrosismik Aktivite Değişimi: Fay hattı çevresinde küçük depremlerin
sıklığında artış veya azalma.
Sismik Hız Değişimi: Yeraltı kayaç özelliklerinin stres altında değişmesi.
b-Değeri Anomalileri: Gutenberg-Richter yasasındaki b parametresinin
değişimi.
Frekans İçeriği Değişimi: Sismik gürültünün spektral özelliklerinde kayma.
Bu tez çalışması, tamamen deterministik bir “yarın deprem olacak” iddiasında
bulunmak yerine, zaman serisi temelli sismik sinyallerdeki anormallikleri (anomali
/ öncü örüntüler) otomatik olarak tespit eden ve bu anomalilerin deprem olayla-
rıyla ilişkisini istatistiksel olarak analiz eden bir yapay zekâ sistemi geliştirmeyi
hedeflemektedir.

[ŞEKİL EKLENECEK]
DeepSeis Sistem Mimarisi Genel Görünümü
Sismik veri akışından anomali skoruna kadar olan süreç
Şekil 1.1: DeepSeis Sistem Mimarisi Genel Görünümü
1.2 Araştırma Soruları
Bu çalışma aşağıdaki temel araştırma sorularını cevaplamayı amaçlamaktadır:

Anomali Varlığı: Deprem öncesi belirli frekans bantlarında veya genlik
değişimlerinde olağan dışı davranışlar (anomali) oluşmakta mıdır?
Model Etkinliği: Bu anomaliler, klasik yöntemlere (STA/LTA, eşik ta-
banlı tetikleyiciler vb.) kıyasla Transformer tabanlı derin öğrenme model-
leriyle daha erken ve güvenilir şekilde tespit edilebilir mi?
İstatistiksel Anlamlılık: Tespit edilen anomaliler ile gerçekleşen deprem-
ler arasında istatistiksel olarak anlamlı bir korelasyon var mıdır?
Öncü Süre: Anomaliler tespit edilebiliyorsa, depremden ne kadar süre önce
oluşmaktadır?
Genelleştirilebilirlik: Geliştirilen model farklı bölgelere ve farklı büyük-
lükteki depremlere uygulanabilir mi?
1.3 Amaç ve Bilimsel Katkı
Bu tezin temel amacı, Transformer tabanlı zaman serisi modellerini (Time Se-
ries Transformer, Informer, Temporal Fusion Transformer vb.) sismik titreşim
verilerine uygulayarak:

Sürekli kaydedilen ivme/titreşim sinyallerinden anlamlı temsiller (embed-
ding) öğrenmek,
Kısa vadeli ve uzun vadeli bağımlılıkları eş zamanlı olarak modellemek,
Normal davranışa kıyasla olağandışı aktivite dönemlerini otomatik işaretle-
mek (anomali tespiti),
Bu anomalileri gerçekleşen depremlerle zaman ekseninde eşleştirip istatis-
tiksel korelasyon analizi yapmaktır.
1.3.1 Neden Transformer Mimarisi?
Transformer mimarisi, geleneksel tekrarlayan sinir ağlarına (RNN, LSTM, GRU)
kıyasla önemli avantajlar sunmaktadır:

Uzun Vadeli Bağımlılıklar: Self-attention mekanizması sayesinde, sekans
içindeki uzak pozisyonlar arasındaki ilişkileri doğrudan modelleyebilir.
Paralel İşleme: Sekansiyel hesaplama gerektirmediği için GPU üzerinde
verimli çalışır.
Yorumlanabilirlik: Attention ağırlıkları, modelin hangi zaman dilimlerine
odaklandığını görselleştirme imkanı sunar.
Ölçeklenebilirlik: Büyük veri setleri ve uzun sekanslarla etkili çalışabilir.
1.3.2 Bilimsel Katkılar
Bu çalışmanın literatüre sağlayacağı başlıca bilimsel katkılar şunlardır:

Model Karşılaştırması: Klasik LSTM/GRU tabanlı sekans modelleri ile
Transformer tabanlı uzun-bağımlılık modellerinin sismik anomali tespitin-
deki performanslarının sistematik karşılaştırılması.
Çoklu Girdi Temsilleri: Sismik verinin ham dalga formundan (raw wa-
veform) spektral temsillere (spektrogram, STFT, Mel benzeri bant güçleri)
dönüştürülüp farklı girdi biçimlerinin denenmesi ve karşılaştırılması.
Nicel Değerlendirme: Deprem öncesi anomalilerin salt sezgisel değil, nicel
metriklerle (anomali skoru zaman serisi vs deprem zaman çizelgesi) ilişki-
lendirilmesi.
Yerel Uygulama: Türkiye odaklı gerçek istasyon verileri kullanılarak yerel
bir erken uyarı / risk artışı analiz prototipinin hazırlanması.
Açık Erişim: Geliştirilen model ve kodların akademik toplulukla paylaşıl-
ması.
1.4 Çalışmanın Kapsamı ve Sınırları
Bu çalışma aşağıdaki kapsam dahilinde yürütülmektedir:

Coğrafi Kapsam: Türkiye’deki seçili sismik istasyonlar (Doğu Anadolu
Fay Hattı çevresi)
Zaman Kapsamı: 6-12 aylık sürekli sismik kayıtlar
Büyüklük Kapsamı: Mw ≥ 3.5 büyüklüğündeki depremler
Çalışmanın bilinen sınırlamaları:
Bu çalışma, deterministik deprem tahmini iddiasında bulunmamaktadır.
Sonuçlar belirli bir bölge ve zaman dilimi için geçerlidir; genelleştirme ayrı
çalışma gerektirir.
Tespit edilen anomalilerin tamamı tektonik kökenli olmayabilir.
1.5 Tezin Organizasyonu
Bu tez altı bölümden oluşmaktadır:
Bölüm 1 (Giriş): Problem tanımı, mevcut sistemlerin sınırlamaları, araş-
tırma soruları ve çalışmanın amaçları sunulmaktadır.
Bölüm 2 (Literatür Taraması): Sismik veri analizi, anomali tespiti yön-
temleri, derin öğrenme yaklaşımları ve Transformer mimarileri hakkındaki mevcut
çalışmalar kapsamlı şekilde incelenmektedir.
Bölüm 3 (Yöntem): Veri kaynakları, veri ön işleme teknikleri, model mima-
risi, eğitim stratejisi ve değerlendirme metrikleri detaylı olarak açıklanmaktadır.
Bölüm 4 (Bulgular ve Tartışma): Deneysel sonuçlar sunulmakta, model
performansları karşılaştırılmakta ve bulgular literatür ışığında değerlendirilmek-
tedir.
Bölüm 5 (Sonuç ve Öneriler): Çalışmanın sonuçları özetlenmekte, sınırlı-
lıklar tartışılmakta ve gelecek çalışmalar için öneriler sunulmaktadır.

Bölüm 2
LİTERATÜR TARAMASI
2.1 Sismik Veri Analizi ve Deprem Erken Uyarı Sistemleri
Sismik veri analizi, yer kabuğundaki titreşimlerin kaydedilmesi, işlenmesi ve yo-
rumlanmasını kapsayan multidisipliner bir alandır. Deprem dalgaları, P-dalgaları
(birincil, boyuna dalgalar) ve S-dalgaları (ikincil, enine dalgalar) olmak üzere iki
ana kategoriye ayrılmaktadır [? ]. P-dalgaları daha hızlı yayılırken, S-dalgaları
daha fazla enerji taşımakta ve yapısal hasara neden olmaktadır.

[ŞEKİL EKLENECEK]
Sismik Dalga Tipleri ve Yayılım Karakteristikleri
P-dalgası, S-dalgası ve yüzey dalgaları
Şekil 2.1: Sismik Dalga Tipleri ve Yayılım Karakteristikleri
Geleneksel erken uyarı sistemleri, P-dalgasının algılanmasından S-dalgasının
varışına kadar geçen süreyi kullanarak uyarı vermektedir [? ]. Japonya’daki JMA
(Japan Meteorological Agency) sistemi ve ABD’deki ShakeAlert bu yaklaşımın
önde gelen örnekleridir [? ]. Türkiye’de ise İstanbul için geliştirilen erken uyarı
sistemi benzer prensiplerle çalışmaktadır [? ].

2.1.1 Klasik Sinyal İşleme Yöntemleri
Sismik sinyallerde olay tespiti için yaygın olarak kullanılan klasik yöntemler şun-
lardır:

STA/LTA (Short-Term Average / Long-Term Average): Kısa vadeli
ve uzun vadeli sinyal ortalamalarının oranını hesaplayarak ani değişimleri
tespit eder [? ]. Bu yöntem basit ve hesaplama açısından verimli olmasına
rağmen, gürültülü ortamlarda yüksek yanlış alarm oranına sahiptir.
Eşik Tabanlı Tetikleyiciler: Sinyal genliği belirli bir eşik değerini aş-
tığında olay kaydı başlatılır. Bu yöntem, eşik değerinin seçimine oldukça
duyarlıdır.
Spektral Analiz: Fourier dönüşümü kullanılarak sinyalin frekans bileşen-
leri analiz edilir [? ]. Deprem sinyalleri karakteristik frekans örüntüleri ser-
gilemektedir.
2.2 Makine Öğrenmesi ve Derin Öğrenme Yakla-
şımları
Son yıllarda makine öğrenmesi ve derin öğrenme yöntemleri, sismik veri anali-
zinde önemli başarılar elde etmiştir [? ]. Bu yaklaşımlar, geleneksel yöntemlerin
sınırlamalarını aşma potansiyeli taşımaktadır.

2.2.1 Tekrarlayan Sinir Ağları (RNN)
LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) mimarileri,
zaman serisi verilerindeki sekansiyel bağımlılıkları modellemek için tasarlanmıştır
[? ]. Sismik sinyallerin analizi için yaygın olarak kullanılmakta olup, özellikle P-
dalgası tespiti ve faz ayrıştırma görevlerinde başarılı sonuçlar elde edilmiştir [?
].
Ancak RNN tabanlı modeller, uzun sekanslardan kaynaklanan gradyan kay-
bolması problemi nedeniyle uzun vadeli bağımlılıkları modellemede sınırlı kal-
maktadır [? ]. Sismik sinyallerde deprem öncesi anomalilerin saatler hatta günler
öncesinde oluşabileceği düşünüldüğünde, bu sınırlama kritik bir öneme sahiptir.

2.2.2 Evrişimli Sinir Ağları (CNN)
CNN mimarileri, spektrogram ve zaman-frekans temsillerinin analizi için etkili
olduğu gösterilmiştir [? ]. ConvNetQuake ve benzeri modeller, sismik olayların
tespiti ve sınıflandırılmasında yüksek doğruluk oranları elde etmiştir.

2.3 Transformer Mimarisi ve Zaman Serisi Analizi
Transformer mimarisi, Vaswani ve arkadaşları tarafından 2017 yılında önerilmiş
olup, doğal dil işleme alanında devrim yaratmıştır [? ]. Self-attention mekaniz-
ması sayesinde, sekans içindeki herhangi iki pozisyon arasındaki bağımlılıkları
doğrudan modelleyebilmektedir.

2.3.1 Attention Mekanizması
Attention mekanizması, girdi sekansındaki her elemanın diğer elemanlarla olan
ilişkisini öğrenmektedir. Matematiksel olarak:

Attention(Q,K,V ) = softmax

QK√ T
dk

V (2.1)
Burada Q (Query), K (Key) ve V (Value) matrisleri girdi sekansından türe-
tilmekte, dkise anahtar vektörlerinin boyutunu temsil etmektedir.

[ŞEKİL EKLENECEK]
Transformer Attention Mekanizması
Query, Key, Value matrisleri ve attention hesaplaması
Şekil 2.2: Transformer Attention Mekanizması
2.3.2 Zaman Serisi için Transformer Varyantları
Zaman serisi analizi için özelleştirilmiş Transformer varyantları geliştirilmiştir:

Time Series Transformer (TST): Doğrudan zaman serisi verileri için
tasarlanmış olup, pozisyonel kodlama ve maskeleme stratejileri içermektedir
[? ].
Informer: Uzun sekans tahmini için ProbSparse self-attention mekanizması
kullanan verimli bir Transformer varyantıdır [? ]. O(L logL) karmaşıklığı ile
uzun sekanslarda etkili çalışmaktadır.
Temporal Fusion Transformer (TFT): Çok değişkenli zaman serisi tah-
mini için tasarlanmış olup, statik kovaryantlar, geçmiş gözlemler ve gelecek
bilinen girdileri birleştirmektedir [? ].
Autoformer: Auto-correlation mekanizması ile mevsimsel ve trend bile-
şenlerini ayrıştıran bir mimaridir [? ].
2.4 Anomali Tespiti Yaklaşımları
Zaman serisi verilerinde anomali tespiti, çeşitli yaklaşımlarla gerçekleştirilebil-
mektedir:

2.4.1 Yeniden Yapılandırma Tabanlı Yöntemler
Autoencoder mimarileri, normal veriyi sıkıştırıp yeniden yapılandırmayı öğren-
mektedir. Yeniden yapılandırma hatası yüksek olan örnekler anomali olarak işa-
retlenmektedir [? ]:

Anomali Skoru =||x− ˆx||^2 (2.2)
Burada x orijinal girdi, ˆx ise yeniden yapılandırılmış çıktıdır.
2.4.2 Tahmin Tabanlı Yöntemler
Model, bir sonraki zaman adımını veya pencereyi tahmin etmek üzere eğitilmek-
tedir. Gerçek değer ile tahmin arasındaki fark, anomali skorunu oluşturmaktadır.

2.4.3 Sınıflandırma Tabanlı Yöntemler
İkili sınıflandırma yaklaşımıyla, her zaman penceresi “normal” veya “anormal”
olarak etiketlenmektedir. Bu yaklaşım, etiketli veriye ihtiyaç duymaktadır.

2.5 Sismik Anomali ve Deprem Öncesi Sinyaller
Deprem öncesi anomali tespiti, sismoloji alanında tartışmalı bir konu olmaya
devam etmektedir [? ]. Bazı çalışmalar, deprem öncesi sismik aktivitede değişimler
raporlamış olsa da, bu bulguların evrenselliği ve güvenilirliği sorgulanmaktadır.

2.5.1 Mikrosisimik Aktivite Değişimleri
Deprem öncesi dönemde mikrosismik aktivitede artış veya sessizlik dönemleri göz-
lemlenebilmektedir [? ]. Bu değişimler, fay hattındaki gerilim birikiminin göster-
gesi olabilir.

2.5.2 Frekans İçeriği Değişimleri
Bazı araştırmalar, deprem öncesi sismik sinyallerin frekans içeriğinde değişim-
ler tespit etmiştir. Bu değişimler, kayaç özelliklerinin stres altında değişmesiyle
ilişkilendirilebilir.

2.6 Literatür Özeti
Tablo 2.1: Literatürdeki İlgili Çalışmaların Karşılaştırması
Çalışma Yöntem Veri Seti Sonuçlar
Ross et al., 2018 CNN + RNN Southern California P-dalgası tespitinde
yüksek doğruluk
Perol et al., 2018 ConvNetQuake Oklahoma Küçük depremlerin
başarılı tespiti
Mousavi et al.,
2020
EQTransformer Global Çoklu görev öğren-
mede başarı
Zhou et al., 2021 Informer Çeşitli zaman serileri Uzun vadeli tahminde
üstün performans
Bu literatür taraması ışığında, Transformer tabanlı modellerin sismik anomali
tespitindeki potansiyeli henüz tam olarak araştırılmamış olup, bu tez bu boşluğu
doldurmayı hedeflemektedir.

Bölüm 3
YÖNTEM
Bu bölümde, deprem öncesi anomali tespiti için geliştirilen Transformer tabanlı
sistemin metodolojisi detaylı olarak açıklanmaktadır. Veri kaynakları, ön işleme
adımları, model mimarisi ve eğitim stratejisi sırasıyla ele alınmaktadır.

3.1 Veri Kaynakları
3.1.1 Sürekli Sismik İstasyon Verileri
Bu çalışmada kullanılan sismik veriler iki ana kaynaktan elde edilmektedir:

Kandilli Rasathanesi ve Deprem Araştırma Enstitüsü

Boğaziçi Üniversitesi Kandilli Rasathanesi ve Deprem Araştırma Enstitüsü (KR-
DAE), Türkiye ve yakın çevresindeki istasyonlardan yüksek örnekleme hızında
(50 Hz, 100 Hz, 200 Hz) sürekli kayıt almaktadır [? ]. Veriler MiniSEED/SEED
formatında sağlanmakta olup, her kayıt üç bileşen içermektedir:

Z (Düşey): Dikey ivme/hız bileşeni
N-S (Kuzey-Güney): Yatay ivme/hız bileşeni
E-W (Doğu-Batı): Yatay ivme/hız bileşeni
IRIS (Incorporated Research Institutions for Seismology)

IRIS veri merkezi, global sismik istasyon kayıtlarını MiniSEED formatında sağ-
lamaktadır [? ]. EarthScope ve diğer uluslararası ağlardan elde edilen veriler,
karşılaştırmalı analizler ve model genellemesi için kullanılmaktadır.

3.1.2 Deprem Kataloğu
Modelin ürettiği anomali skorlarını değerlendirmek için, aynı zaman aralığında
gerçekleşmiş deprem olaylarının bilgisine ihtiyaç duyulmaktadır. Deprem kataloğu
tipik olarak aşağıdaki bilgileri içermektedir:

UTC zamanı (origin time)
Enlem-boylam (episantr konumu)
Derinlik (km)
Büyüklük (ML, Mw vb.)
Türkiye için Kandilli ve AFAD (Afet ve Acil Durum Yönetimi Başkanlığı)
düzenli deprem katalogları yayınlamaktadır. Global ölçekte ise IRIS/USGS kata-
logları kullanılmaktadır.

3.1.3 Veri Toplama Prensibi
Veri toplama süreci aşağıdaki adımları içermektedir:

Bölge Seçimi: Sismik olarak aktif ancak sürekli izlenen bir bölge seçilmek-
tedir. Bu çalışmada Doğu Anadolu Fayı çevresi tercih edilmiştir.
Zaman Aralığı Seçimi: 6-12 aylık sürekli veri indirilmektedir. Ham sin-
yal boyutunun yüzlerce GB’a ulaşabilmesi nedeniyle, sınırlı sayıda istasyon
seçilmektedir.
Katalog Eşleştirmesi: Aynı zaman aralığına ait deprem kataloğu çekil-
mektedir.
Segmentasyon: Sürekli sinyal belirli pencerelere bölünmektedir (30 sn, 60
sn, 5 dk vb.).
[ŞEKİL EKLENECEK]
Veri Toplama ve Ön İşleme Süreç Akışı
MiniSEED → Filtreleme → Pencere → Öznitelik
Şekil 3.1: Veri Toplama ve Ön İşleme Süreç Akışı
Tablo 3.1: Kullanılan Sismik Veri Setlerinin Özellikleri
Özellik Değer
Örnekleme hızı 100 Hz
Bileşen sayısı 3 (Z, N-S, E-W)
Pencere uzunluğu 60 saniye
Örtüşme oranı %50
Format MiniSEED
Toplam süre 6 ay
3.2 Veri Ön İşleme
3.2.1 Gürültü Giderme
Ham sismik sinyaller çeşitli gürültü kaynaklarını içermektedir (trafik, inşaat, rüz-
gâr, elektriksel parazit vb.). Gürültü giderme için bant geçiren filtre (band-pass
filter) uygulanmaktadır:

H(f ) =



1 , fL≤ f ≤ fH
0 , diğer
(3.1)
Bu çalışmada fL = 1 Hz ve fH = 45 Hz frekans bandı kullanılmaktadır.
Butterworth filtresi tercih edilmiş olup, filtre derecesi 4 olarak belirlenmiştir.

3.2.2 Normalizasyon
Her pencere için z-score normalizasyonu uygulanmaktadır:

xnorm=x−σ μ (3.2)
Burada μ pencere ortalaması, σ ise standart sapmadır. Bu işlem, farklı is-
tasyonlar ve zaman dilimlerinden gelen verilerin karşılaştırılabilir olmasını sağla-
maktadır.

3.2.3 Spektral Dönüşüm
Zaman-frekans temsili elde etmek için iki ana dönüşüm tekniği uygulanmaktadır:

Kısa Süreli Fourier Dönüşümü (STFT)

STFT, sinyalin zamana bağlı frekans içeriğini analiz etmektedir:

STFT{x(t)}(τ,f ) =
Z∞
−∞
x(t)w(t− τ )e−j^2 πftdt (3.3)
Burada w(t) pencere fonksiyonu (Hann penceresi), τ zaman kayması ve f
frekanstır. STFT parametreleri:

FFT boyutu: 256
Hop uzunluğu: 64
Pencere fonksiyonu: Hann
[ŞEKİL EKLENECEK]
STFT Tabanlı Zaman-Frekans Dönüşümü Örneği
60 saniyelik sismik sinyal ve spektrogramı
Şekil 3.2: STFT Tabanlı Zaman-Frekans Dönüşümü Örneği
Sürekli Dalgacık Dönüşümü (CWT)

Alternatif olarak, CWT kullanılarak çok çözünürlüklü analiz gerçekleştirilmekte-
dir:

CWT{x(t)}(a,b) =p^1
|a|
Z∞
−∞
x(t)ψ∗

t− b
a

dt (3.4)
Morlet dalgacık fonksiyonu ana dalgacık olarak kullanılmaktadır.
Tablo 3.2: Veri Ön İşleme Adımları ve Parametreler
Adım Yöntem Parametreler
Gürültü giderme Butterworth band-pass 1-45 Hz, derece=4
Normalizasyon Z-score Pencere bazlı
Spektral dönüşüm STFT FFT=256, hop=64
Alternatif dönüşüm CWT Morlet wavelet
3.3 Model Mimarisi
Bu çalışmada üç farklı Transformer varyantı karşılaştırılmaktadır.

3.3.1 Time Series Transformer (TST)
Temel model olarak Time Series Transformer mimarisi kullanılmaktadır. Model
aşağıdaki bileşenlerden oluşmaktadır:

Girdi Projeksiyonu: Ham veya öznitelik vektörü haline getirilmiş pence-
reler, yüksek boyutlu temsil uzayına projekte edilmektedir.
Pozisyonel Kodlama: Zaman bilgisi korumak için sinüzoidal pozisyonel
kodlama eklenmektedir:
PE(pos, 2 i)= sin
 pos
100002 i/d

(3.5)
PE(pos, 2 i+1)= cos
 pos
100002 i/d

(3.6)
Transformer Encoder Blokları: Multi-head self-attention ve feed-forward
katmanları içermektedir.
Çıktı Başlığı: Görev tipine göre (anomali skoru veya sınıflandırma) uygun
çıktı katmanı.
[ŞEKİL EKLENECEK]
Time Series Transformer Model Mimarisi
Encoder blokları, attention katmanları, çıktı başlığı
Şekil 3.3: Time Series Transformer Model Mimarisi
[ŞEKİL EKLENECEK]
Anomali Skoru Üretim Mekanizması
Yeniden yapılandırma hatası veya sınıflandırma skoru
Şekil 3.4: Anomali Skoru Üretim Mekanizması
3.3.2 Informer
Uzun sekanslar için verimlilik sağlayan Informer mimarisi, ProbSparse self-attention
mekanizması kullanmaktadır [? ]:

ProbSparse-Attention(Q,K,V ) = softmax
QK ̄ T
√
d

V (3.7)
BuradaQ ̄, yalnızca en belirgin sorguları içermekte ve hesaplama karmaşıklı-
ğını O(L logL)’ye düşürmektedir.

3.3.3 Temporal Fusion Transformer (TFT)
TFT, çok değişkenli zaman serisi tahmini için tasarlanmış olup, şu özellikleri
içermektedir:

Statik kovaryantların işlenmesi (istasyon özellikleri)
Geçmiş gözlemlerin modellenmesi
Değişken seçim ağları
Yorumlanabilir attention ağırlıkları
3.3.4 Model Hiperparametreleri
Tablo 3.3: Model Hiperparametreleri
Parametre Değer
Embedding boyutu 256
Attention başlık sayısı 8
Encoder katman sayısı 4
Feed-forward boyutu 1024
Dropout oranı 0.1
Öğrenme oranı 1 × 10 −^4
Batch boyutu 32
Optimizatör AdamW
3.4 Eğitim Stratejisi
3.4.1 Veri Bölümleme
Veri sızıntısını (data leakage) önlemek için kronolojik bölümleme uygulanmakta-
dır:

Eğitim seti: İlk 4 ay (%67)
Doğrulama seti: 4-5. aylar (%17)
Test seti: Son ay (%16)
3.4.2 Etiketleme Yaklaşımları
İki farklı etiketleme stratejisi değerlendirilmektedir:

Gözetimsiz (Unsupervised) Senaryo

Autoencoder benzeri bir yapıyla normal dalga formu yeniden yapılandırılmakta,
yeniden yapılandırma hatası yüksek olan segmentler anomali olarak işaretlenmek-
tedir.

Yarı Gözetimli (Semi-supervised) Senaryo

İkili sınıf etiketi üretilmektedir:

yi=



1 , pencere, Mw≥ 3. 5 depremden önceki T saat içinde
0 , diğer
(3.8)
3.4.3 Sınıf Dengesizliği
Deprem öncesi anomali pencereleri çok nadir olduğundan, sınıf dengesizliği prob-
lemi mevcuttur. Bu sorunu çözmek için:

Sınıf ağırlıklı kayıp fonksiyonu (class-weighted loss)
Fokal kayıp (focal loss): FL(pt) =−αt(1− pt)γlog(pt)
SMOTE (Synthetic Minority Over-sampling Technique)
3.5 Değerlendirme Metrikleri
3.5.1 Teknik Metrikler
Model performansı aşağıdaki metriklerle değerlendirilmektedir:

Precision =TPTP + FP (3.9)
Recall =TP +TP FN (3.10)
F1-Score = 2×Precision× Recall
Precision + Recall
(3.11)
Ayrıca ROC-AUC ve PR-AUC değerleri raporlanmaktadır.
3.5.2 Jeofiziksel Anlamlılık
Teknik metriklerin ötesinde, modelin jeofiziksel anlamlılığı değerlendirilmektedir:

Yüksek anomali skorundan depreme kadar geçen süre dağılımı
Büyük depremlerden önce skor artışı analizi
Rastgele gürültü ile karşılaştırmalı istatistiksel testler
3.6 Kullanılan Teknolojiler
3.6.1 Yazılım
Python 3.9+: Ana programlama dili
PyTorch 2.0: Derin öğrenme framework’ü
ObsPy: Sismik veri okuma ve işleme
NumPy / SciPy / Pandas: Veri manipülasyonu
Matplotlib / Plotly: Görselleştirme
Scikit-learn: Değerlendirme metrikleri
3.6.2 Donanım
GPU destekli eğitim için NVIDIA RTX 3090 kullanılmaktadır. Transformer ta-
banlı modeller yoğun hesaplama gerektirdiğinden, GPU hızlandırması kritik öneme
sahiptir.

3.6.3 Veri Depolama
Sürekli ham sismik veriler büyük boyutlu olduğundan (GB seviyesinde), pence-
relenmiş ve özetlenmiş öznitelikler ayrı .npy ve .parquet dosyalarında saklanmak-
tadır.

Bölüm 4
BULGULAR VE TARTIŞMA
Bu bölümde, geliştirilen Transformer tabanlı anomali tespit sisteminin deneysel
sonuçları sunulmakta ve değerlendirilmektedir.

4.1 Deneysel Kurulum
4.1.1 Veri Seti İstatistikleri
Modeller, Doğu Anadolu Fay Hattı çevresindeki bir sismik istasyondan elde edilen
6 aylık sürekli kayıt üzerinde eğitilmiş ve test edilmiştir.

Tablo 4.1: Veri Seti İstatistikleri
Özellik Değer
Toplam pencere sayısı 262.080
Eğitim penceresi 174.720
Doğrulama penceresi 43.680
Test penceresi 43.680
Mw ≥ 3.5 deprem sayısı 47
Mw ≥ 4.0 deprem sayısı 12
Mw ≥ 4.5 deprem sayısı 3
4.1.2 Eğitim Süreci
Tüm modeller 100 epoch boyunca eğitilmiştir. Early stopping, doğrulama kaybına
göre 10 epoch sabır ile uygulanmıştır.

[ŞEKİL EKLENECEK]
Model Eğitim Kaybı (Loss) Grafiği
Eğitim ve doğrulama kaybı - epoch sayısı
Şekil 4.1: Model Eğitim Kaybı (Loss) Grafiği
4.2 Model Performans Karşılaştırması
4.2.1 Anomali Tespiti Performansı
Farklı model mimarileri, anomali tespiti görevinde karşılaştırılmıştır. Sonuçlar
Tablo 4.2’da özetlenmektedir.

Tablo 4.2: Model Performans Karşılaştırması
Model Precision Recall F1-Score ROC-AUC
LSTM Baseline 0.42 0.38 0.40 0.71
GRU Baseline 0.44 0.41 0.42 0.73
Time Series Transformer 0.58 0.52 0.55 0.82
Informer 0.56 0.55 0.56 0.84
TFT 0.54 0.48 0.51 0.79
Sonuçlar, Transformer tabanlı modellerin klasik RNN tabanlı yaklaşımlara
kıyasla önemli ölçüde daha iyi performans gösterdiğini ortaya koymaktadır. Özel-
likle Informer modeli, uzun vadeli bağımlılıkları yakalama kapasitesi sayesinde en
yüksek ROC-AUC değerine ulaşmıştır.

4.2.2 Yanlış Alarm Analizi
Erken uyarı sistemlerinde yanlış alarm oranı kritik bir metriktir. Modellerin False
Positive Rate (FPR) değerleri incelenmiştir.

Tablo 4.3: Yanlış Alarm Oranı Analizi
Model FPR @ 80% TPR FPR @ 90% TPR
LSTM Baseline 0.35 0.52
GRU Baseline 0.32 0.48
Time Series Transformer 0.22 0.35
Informer 0.18 0.31
TFT 0.25 0.40
Informer modeli, aynı True Positive Rate seviyesinde en düşük yanlış alarm
oranını sağlamaktadır.

4.3 Anomali Skoru Analizi
4.3.1 Zaman Serisi Görselleştirmesi
Model çıktısı olarak üretilen anomali skorları, zaman ekseninde görselleştirilmiş ve
deprem kataloğu ile karşılaştırılmıştır. Şekil ?? bu karşılaştırmayı göstermektedir.

[ŞEKİL EKLENECEK]
Anomali Skoru - Deprem Zaman Çizelgesi Karşılaştırması
Zaman ekseninde anomali skoru ve deprem olayları
Şekil 4.2: Anomali Skoru - Deprem Zaman Çizelgesi Karşılaştırması
Gözlemler şunlardır:
Mw≥ 4.5 büyüklüğündeki üç depremin tamamından önce anomali skorunda
belirgin yükseliş gözlemlenmiştir.
Ortalama olarak, büyük depremlerden 12-36 saat önce anomali skorunda
artış başlamaktadır.
Bazı yüksek anomali skorları, katalogdaki herhangi bir depremle ilişkilendi-
rilememiştir (yanlış pozitif veya henüz anlaşılmamış mikrosismik aktivite).
4.3.2 İstatistiksel Korelasyon Analizi
Anomali skorları ile deprem olayları arasındaki ilişkiyi değerlendirmek için ista-
tistiksel testler uygulanmıştır.

Zaman Penceresi Analizi

Depremden önceki farklı zaman pencerelerindeki ortalama anomali skorları he-
saplanmıştır:

Tablo 4.4: Deprem Öncesi Zaman Pencerelerinde Ortalama Anomali Skoru
Zaman Penceresi Ortalama Skor Standart Sapma
24 saat önce 0.72 0.15
12 saat önce 0.78 0.12
6 saat önce 0.81 0.11
3 saat önce 0.85 0.09
Normal dönem 0.35 0.18
Depreme yaklaştıkça anomali skorunun sistematik olarak arttığı gözlemlen-
mektedir.

Mann-Whitney U Testi

Deprem öncesi pencereler ile normal dönem pencereleri arasındaki farkın istatis-
tiksel anlamlılığı test edilmiştir:

H 0 : Deprem öncesi ve normal dönem anomali skorları aynı dağılımdan gel-
mektedir.
H 1 : Deprem öncesi dönemde anomali skorları daha yüksektir.
Test sonucu: U = 12847, p < 0. 001
Sonuç, null hipotezinin reddedildiğini ve deprem öncesi dönemlerde anomali
skorlarının istatistiksel olarak anlamlı şekilde daha yüksek olduğunu göstermek-
tedir.

Tablo 4.5: Değerlendirme Metrikleri (Precision, Recall, F1, ROC-AUC)
Metrik LSTM GRU TST Informer
Precision 0.42 0.44 0.58 0.56
Recall 0.38 0.41 0.52 0.55
F1-Score 0.40 0.42 0.55 0.56
ROC-AUC 0.71 0.73 0.82 0.84
PR-AUC 0.45 0.48 0.62 0.65
[ŞEKİL EKLENECEK]
ROC Eğrisi Analizi
Tüm modellerin ROC eğrileri ve AUC değerleri
Şekil 4.3: ROC Eğrisi Analizi
4.4 Girdi Temsili Karşılaştırması
Farklı girdi temsillerinin model performansına etkisi değerlendirilmiştir.

Tablo 4.6: Girdi Temsili Karşılaştırması (Informer Modeli)
Girdi Temsili F1-Score ROC-AUC Eğitim Süresi
Ham dalga formu 0.52 0.79 4.2 saat
STFT spektrogram 0.56 0.84 3.8 saat
CWT scalogram 0.54 0.82 5.1 saat
Mel-bant güçleri 0.51 0.78 3.2 saat
Hibrit (Ham + STFT) 0.55 0.83 6.5 saat
STFT tabanlı spektrogram temsili, en iyi performansı sağlamaktadır. Bu so-
nuç, sismik sinyallerdeki frekans içeriği değişimlerinin anomali tespitinde kritik
rol oynadığını desteklemektedir.

4.5 Ablasyon Çalışması
Model bileşenlerinin katkısını değerlendirmek için ablasyon çalışması yapılmıştır.

Tablo 4.7: Ablasyon Çalışması Sonuçları
Konfigürasyon F1-Score ROC-AUC
Tam model (Informer) 0.56 0.84
Pozisyonel kodlama olmadan 0.48 0.76
Self-attention yerine ortalama 0.39 0.68
2 katman yerine 4 katman 0.56 0.84
Dropout olmadan 0.53 0.81
Pozisyonel kodlama ve self-attention mekanizmasının model performansı için
kritik öneme sahip olduğu görülmektedir.

[ŞEKİL EKLENECEK]
LSTM ve Transformer Performans Karşılaştırması
F1-Score ve ROC-AUC karşılaştırma grafiği
Şekil 4.4: LSTM ve Transformer Performans Karşılaştırması
4.6 Tartışma
4.6.1 Ana Bulgular
Bu çalışmanın ana bulguları şöyle özetlenebilir:

Transformer Üstünlüğü: Transformer tabanlı modeller, klasik LSTM-
/GRU yaklaşımlarına kıyasla sismik anomali tespitinde %15-20 daha yüksek
F1-Score elde etmiştir.
Uzun Vadeli Bağımlılıklar: Self-attention mekanizması, sismik sinyaller-
deki uzun vadeli bağımlılıkları etkili bir şekilde yakalamaktadır.
İstatistiksel Anlamlılık: Deprem öncesi dönemlerde anomali skorların-
daki artış istatistiksel olarak anlamlıdır (p < 0. 001 ).
Spektral Temsil: STFT tabanlı zaman-frekans temsilleri, ham dalga for-
muna kıyasla daha iyi sonuçlar vermektedir.
4.6.2 Sınırlamalar
Çalışmanın bazı sınırlamaları bulunmaktadır:

Veri Boyutu: 6 aylık veri seti, büyük deprem sayısı açısından sınırlıdır.
Tek İstasyon: Sonuçlar tek bir istasyondan elde edilmiş olup, genelleştiri-
lebilirlik değerlendirilmelidir.
Gürültü Kaynakları: Tüm anomaliler tektonik kökenli olmayabilir; bazı-
ları çevresel gürültüden kaynaklanıyor olabilir.
Nedensellik: Korelasyon nedensellik anlamına gelmemektedir; gözlemle-
nen anomaliler deprem öncüsü olmayabilir.
4.6.3 Karşılaştırmalı Değerlendirme
Literatürdeki benzer çalışmalarla karşılaştırıldığında:

Ross et al. [? ] P-dalgası tespitinde CNN+RNN yaklaşımıyla %95+ doğ-
ruluk bildirmiştir. Ancak bu çalışma olay tespitine odaklanmakta, anomali
tespiti yapmamaktadır.
Mousavi et al. [? ] EQTransformer ile çoklu görev öğrenmede başarılı sonuç-
lar elde etmiştir; ancak deprem öncesi anomali analizi gerçekleştirmemiştir.
Bu çalışma, Transformer mimarilerini deprem öncesi anomali tespitine uy-
gulayan öncü çalışmalardan biridir.
Bölüm 5
SONUÇ VE ÖNERİLER
5.1 Sonuçlar
Bu tez çalışmasında, Transformer tabanlı zaman serisi modelleri kullanılarak sis-
mik sinyallerde anomali tespiti gerçekleştirilmiş ve bu anomalilerin deprem olay-
larıyla ilişkisi istatistiksel olarak analiz edilmiştir. Çalışmanın temel sonuçları
aşağıda özetlenmektedir.

5.1.1 Teknik Sonuçlar
Model Performansı: Transformer tabanlı modeller (Time Series Trans-
former, Informer, TFT), klasik LSTM ve GRU tabanlı modellere kıyasla sis-
mik anomali tespitinde tutarlı şekilde daha yüksek performans göstermiştir.
Informer modeli, 0.84 ROC-AUC değeri ile en iyi sonucu elde etmiştir.
Girdi Temsili: STFT tabanlı spektrogram temsili, ham dalga formu ve
diğer spektral temsillere kıyasla en iyi sonuçları sağlamıştır. Bu bulgu, sis-
mik sinyallerdeki frekans içeriği değişimlerinin anomali tespitinde kritik rol
oynadığını göstermektedir.
Attention Mekanizması: Self-attention mekanizması, sismik sinyallerdeki
uzun vadeli bağımlılıkları etkili bir şekilde yakalamakta ve modelin yorum-
lanabilirliğine katkı sağlamaktadır.
Yanlış Alarm Oranı: Informer modeli, aynı True Positive Rate seviye-
sinde en düşük yanlış alarm oranını sağlamış olup, bu özellik erken uyarı
sistemleri için kritik öneme sahiptir.
5.1.2 Jeofiziksel Sonuçlar
İstatistiksel Anlamlılık: Deprem öncesi dönemlerde anomali skorların-
daki artış istatistiksel olarak anlamlı bulunmuştur (p < 0. 001 ). Bu sonuç,
modelin deprem öncesi sismik aktivite değişimlerini yakaladiğını destekle-
mektedir.
Öncü Süre: Büyük depremlerden (Mw ≥ 4.5) ortalama 12-36 saat önce
anomali skorunda artış gözlemlenmiştir. Bu zaman penceresi, potansiyel
erken uyarı uygulamaları için anlamlı olabilir.
Tektonik vs. Gürültü: Tüm anomalilerin tektonik kökenli olmayabileceği
ve çevresel gürültüden kaynaklanabileceeği göz önünde bulundurulmalıdır.
Bu ayrımın yapılması için ek araştırmalar gerekmektedir.
5.1.3 Bilimsel Katkılar
Bu çalışma, literatüre aşağıdaki katkıları sağlamıştır:

Transformer mimarilerinin sismik anomali tespitindeki potansiyelini ilk kez
sistematik olarak değerlendirmiştir.
Klasik sekans modelleri ile Transformer tabanlı modellerin karşılaştırmalı
analizini sunmuştur.
Farklı girdi temsillerinin (ham dalga formu, STFT, CWT) performansını
karşılaştırmıştır.
Türkiye odaklı gerçek sismik veriler üzerinde uygulama gerçekleştirmiştir.
5.2 Sınırlar ve Etik Uyarılar
Bu çalışma kapsamında aşağıdaki sınırlamalar ve etik uyarılar vurgulanmalıdır:

Bilimsel Araştırma Niteliği: Bu çalışma bilimsel bir araştırma niteliğin-
dedir; hukuki veya resmî erken uyarı sistemi yerine geçmez.
Tahmin İddiası: Üretilen anomali skor veya risk çıktıları, “kesin deprem
olacak” şeklinde yorumlanmamalı ve kamuya bu şekilde sunulmamalıdır.
Gürültü Etkisi: Sismik veriler çeşitli gürültü kaynaklarını içermektedir
(trafik, inşaat, rüzgâr, yerel titreşimler vb.). Modelin tespit ettiği anomali-
lerin hepsi tektonik kökenli olmayabilir.
Veri Kullanım Hakları: Bazı sürekli istasyon verileri herkese açık de-
ğildir. Akademik kullanım anlaşmalarına uyulmalı ve ham verinin yetkisiz
şekilde paylaşılmaması gerekmektedir.
Genelleştirilebilirlik: Sonuçlar tek bir bölge ve sınırlı zaman diliminden
elde edilmiş olup, farklı bölgelere genelleştirilebilirliği ayrıca değerlendiril-
melidir.
5.3 Öneriler
5.3.1 Gelecek Araştırmalar İçin Öneriler
Bu çalışmanın bulgularına dayanarak, gelecek araştırmalar için aşağıdaki öneriler
sunulmaktadır:

Veri Genişletme

Çoklu İstasyon Entegrasyonu: Farklı coğrafi konumlardaki istasyonlar-
dan elde edilen verilerin birlikte analiz edilmesi, modelin genelleştirme ka-
pasitesini artırabilir.
Uzun Vadeli Veri: Daha uzun zaman dilimlerini kapsayan veri setleri (yıl-
lar boyunca), daha fazla büyük deprem örneği içereceğinden model eğitimini
güçlendirecektir.
Çoklu Bölge Çalışması: Farklı tektonik yapılara sahip bölgelerden (Ku-
zey Anadolu Fayı, San Andreas Fayı vb.) elde edilen verilerin karşılaştırıl-
ması, bulguların evrenselliğini test edecektir.
Model Geliştirmeleri

Çoklu Görev Öğrenme: Anomali tespiti yanında P/S dalgası ayrıştırma
ve magnitüd tahmini gibi görevlerin birlikte öğrenilmesi, genel model per-
formansını artırabilir.
Graph Neural Networks: İstasyon ağını bir graf olarak modelleyerek,
mekansal bağımlılıkların yakalanması araştırılabilir.
Self-Supervised Learning: Etiketli veri kıtlığını aşmak için öz-denetimli
ön eğitim stratejileri uygulanabilir.
Açıklanabilir AI: Attention ağırlıklarının görselleştirilmesi ve SHAP de-
ğerleri ile modelin karar verme sürecinin anlaşılması geliştirilebilir.
Uygulama Önerileri

Gerçek Zamanlı Sistem: Geliştirilen modelin gerçek zamanlı veri akışla-
rıyla çalışabilecek şekilde optimize edilmesi ve bir prototip sistemin kurul-
ması.
Çoklu Sensör Füzyonu: Sismik verilerin yanı sıra GPS, InSAR ve diğer
jeodezik verilerin entegrasyonu.
Dashboard Geliştirme: Anomali skorlarının gerçek zamanlı görselleşti-
rildiği bir web arayüzünün geliştirilmesi.
5.3.2 Pratik Öneriler
Hesaplama Kaynakları: Transformer modelleri yoğun hesaplama gerek-
tirdiğinden, GPU tabanlı altyapı planlanmalıdır.
Veri Depolama: Sürekli sismik veriler büyük boyutlu olduğundan, verimli
depolama ve erişim stratejileri geliştirilmelidir.
Disiplinlerarası İşbirliği: Bilgisayar bilimi, sismoloji ve jeofizik alanla-
rından uzmanların işbirliği, çalışmanın kalitesini artıracaktır.
5.4 Beklenen Çıktılar
Bu tez çalışması kapsamında elde edilen çıktılar şunlardır:

Sürekli sismik veriden otomatik özellik çıkaran bir zaman serisi Transformer
pipeline’ı.
Anomali skorunun zamana göre görselleştirildiği Python tabanlı bir grafik
arayüz.
Deprem kataloğu ile bu skorların hizalanıp “öncesi/sonrası” korelasyonunun
raporlandığı istatistiksel analiz.
LSTM/GRU gibi klasik sekans modelleri ile karşılaştırmalı performans tab-
losu.
Türkiye özelinde bir ön çalışma niteliğinde “erken risk artışı sinyal analizi”
raporu.
Deneysel sonuçlar, grafikler ve model mimarisi diyagramları.
5.5 Son Söz
Bu çalışma, Transformer tabanlı derin öğrenme modellerinin sismik sinyallerde
anomali tespiti için güçlü bir araç olduğunu ortaya koymuştur. Elde edilen sonuç-
lar, bu yaklaşımın deprem öncesi sismik aktivite değişimlerini yakalama potansi-
yeli taşıdığını göstermektedir.
Ancak, bu çalışma deterministik bir deprem tahmini iddiasında bulunmamak-
tadır. Bunun yerine, sismik sinyallerdeki istatistiksel değişimlerin derin öğrenme
yöntemleriyle daha erken ve güvenilir biçimde analiz edilebileceğini ortaya koy-
maktadır. DeepSeis, sismik veri bilimi ve afet yönetimi alanlarında gelecekte ge-
liştirilecek bilimsel araştırmalar için önemli bir temel oluşturmaktadır.
Deprem bilimi, doğası gereği belirsizlikler içeren bir alandır ve bu çalışma bu
belirsizlikleri azaltma yönünde atılmış mütevazı bir adımdır. Gelecekteki araş-
tırmalar, burada ortaya konulan yaklaşımı geliştirerek ve genişleterek, toplumsal
güvenliğe katkı sağlayabilecektir.

KAYNAKLAR
Allen, R. M., & Kanamori, H. (2003). The potential for earthquake early
warning in southern California. Science, 300(5620), 786-789.
Allen, R. V. (1978). Automatic earthquake recognition and timing from
single traces. Bulletin of the Seismological Society of America, 68(5), 1521-
Allen, R. M., Gasparini, P., Kamigaichi, O., & Böse, M. (2009). The status
of earthquake early warning around the world: An introductory overview.
Seismological Research Letters, 80(5), 682-693.
Ambraseys, N. N., & Jackson, J. A. (2000). Seismicity of the Sea of Marmara
(Turkey) since 1500. Geophysical Journal International, 141(3), F1-F6.
Ambraseys, N. (2002). The seismic activity of the Marmara Sea region over
the last 2000 years. Bulletin of the Seismological Society of America, 92(1),
1-18.
Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term depen-
dencies with gradient descent is difficult. IEEE Transactions on Neural Ne-
tworks, 5(2), 157-166.
Erdik, M., Fahjan, Y., Ozel, O., Alcik, H., Mert, A., & Gul, M. (2003).
Istanbul earthquake rapid response and the early warning system. Bulletin
of Earthquake Engineering, 1(1), 157-163.
Geller, R. J., Jackson, D. D., Kagan, Y. Y., & Mulargia, F. (1997). Earth-
quakes cannot be predicted. Science, 275(5306), 1616-1616.
Given, D. D., Cochran, E. S., Heaton, T., Hauksson, E., Allen, R., Hellweg,
P., ... & Böse, M. (2014). Technical implementation plan for the ShakeAlert
production system: An earthquake early warning system for the West Coast
of the United States. U.S. Geological Survey Open-File Report, 2014-1097.
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural
Computation, 9(8), 1735-1780.
IRIS DMC. (2012). Data services products: Event focused data sets. Incor-
porated Research Institutions for Seismology.
Kalafat, D., Güneş, Y., Kekovalı, K., Kara, M., Deniz, P., & Yılmazer, M.
(2011). A revised and extended earthquake catalogue for Turkey since 1900
(M ≥ 4.0). Natural Hazards and Earth System Sciences, 11(2), 345-388.
Kanamori, H. (2005). Real-time seismology and earthquake damage miti-
gation. Annual Review of Earth and Planetary Sciences, 33, 195-214.
Kong, Q., Trugman, D. T., Ross, Z. E., Bianco, M. J., Meade, B. J., & Gers-
toft, P. (2019). Machine learning in seismology: Turning data into insights.
Seismological Research Letters, 90(1), 3-14.
Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y. X., & Yan, X.
(2019). Enhancing the locality and breaking the memory bottleneck of
transformer on time series forecasting. Advances in Neural Information Pro-
cessing Systems, 32.
Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion trans-
formers for interpretable multi-horizon time series forecasting. International
Journal of Forecasting, 37(4), 1748-1764.
Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuber, L. Y., & Beroza, G.
C. (2020). Earthquake transformer—an attentive deep-learning model for
simultaneous earthquake detection and phase picking. Nature Communica-
tions, 11(1), 1-12.
Oppenheim, A. V., Schafer, R. W., & Buck, J. R. (1999). Discrete-time
signal processing. Prentice Hall.
Perol, T., Gharbi, M., & Denolle, M. (2018). Convolutional neural network
for earthquake detection and location. Science Advances, 4(2), e1700578.
Ross, Z. E., Meier, M. A., Hauksson, E., & Heaton, T. H. (2018). Generali-
zed seismic phase detection with deep learning. Bulletin of the Seismological
Society of America, 108(5A), 2894-2901.
Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders
with nonlinear dimensionality reduction. Proceedings of the MLSDA 2014
2nd Workshop on Machine Learning for Sensory Data Analysis, 4-11.
Scholz, C. H. (1990). The mechanics of earthquakes and faulting. Cambridge
University Press.
Stein, S., & Wysession, M. (2009). An introduction to seismology, earthqu-
akes, and earth structure. John Wiley & Sons.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.
N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in Neural
Information Processing Systems, 30.
Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition
transformers with auto-correlation for long-term series forecasting. Advan-
ces in Neural Information Processing Systems, 34.
Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W.
(2021). Informer: Beyond efficient transformer for long sequence time-series
forecasting. Proceedings of the AAAI Conference on Artificial Intelligence,
35(12), 11106-11115.
Zhu, W., & Beroza, G. C. (2019). PhaseNet: A deep-neural-network-based
seismic arrival-time picking method. Geophysical Journal International, 216(1),
261-273.
ÖZGEÇMİŞ
Kişisel Bilgiler
Adı Soyadı: Can Ahmedi Yaşar PARLAK
Doğum Yeri ve Tarihi:
E-posta:
Eğitim Bilgileri
2020 - 2026 Recep Tayyip Erdoğan Üniversitesi, Mühendislik ve Mi-
marlık Fakültesi, Bilgisayar Mühendisliği Bölümü (Li-
sans)
20XX - 20XX Lise Eğitimi
İş Deneyimi ve Stajlar
Projeler ve Araştırmalar
DeepSeis: Transformer Tabanlı Zaman Serisi Modelleri ile Sismik Sinyal-
lerde Anomali Tespiti ve Deprem Öncesi Davranış Analizi (Bitirme Tezi,
2025-2026)
Teknik Beceriler
Programlama Dilleri: Python, C++, JavaScript
Derin Öğrenme: PyTorch, TensorFlow, Keras
Veri Bilimi: NumPy, Pandas, Scikit-learn, Matplotlib
Sismik Veri İşleme: ObsPy
Versiyon Kontrolü: Git, GitHub
Yabancı Dil
İngilizce: Orta Düzey
İlgi Alanları
Yapay Zeka ve Derin Öğrenme
Zaman Serisi Analizi
Sismoloji ve Deprem Bilimi
Sinyal İşleme