DeepSeis Bitirme Tezi Uygulama Aşaması Yol Haritası

Güncel rapor incelemesine dayanarak hazırlanan uygulama ve geliştirme planı

Bu metin, bitirme tezinin mevcut rapor sürümü esas alınarak hazırlanmış uygulama odaklı yol haritasını içermektedir. Çalışmanın amacı yalnızca teorik çerçeveyi korumak değil, raporda tanımlanan yöntemi adım adım çalışan bir araştırma ve geliştirme sürecine dönüştürmektir.

Rapor incelendiğinde problem tanımı, veri kaynakları, ön işleme yaklaşımı, model adayları ve değerlendirme mantığı büyük ölçüde belirlenmiş durumdadır. Buna karşılık bazı şekillerin, tabloların ve sonuç bölümlerinin proje tamamlandığında gerçek deney çıktıları ile güncellenmesi gerekmektedir. Bu nedenle aşağıda verilen plan, tez yazımından çok uygulama geliştirme sürecine odaklanmaktadır.

# 1\. Genel değerlendirme

Mevcut rapora göre proje uygulamaya geçmeye hazır bir olgunluğa ulaşmıştır. Araştırma soruları, veri toplama mantığı, zaman serisi temelli yaklaşım, önişleme adımları ve Transformer tabanlı modellerin kullanım gerekçesi açık biçimde tanımlanmıştır.

Bununla birlikte raporda yer alan bazı görseller henüz şema düzeyindedir. Bulgular kısmında yer alacak tablolar, başarı karşılaştırmaları, ROC eğrileri, anomali skor grafikleri ve deneysel yorumlar ancak gerçek veri üzerinde yapılacak uygulama sonrasında nihai hâline kavuşacaktır.

Bu nedenle bundan sonraki süreçte öncelik, tez metnini genişletmekten çok veri hattını kurmak, deneyleri çalıştırmak, sonuçları toplamak ve daha sonra bu çıktıları tez metnine bilimsel düzen içinde yerleştirmek olmalıdır.

# 2\. Uygulama sürecinin temel hedefi

Uygulama sürecinin temel hedefi, sürekli sismik veriler üzerinden pencereleme ve önişleme yaparak anlamlı özellik temsilleri üretmek, bu veriler üzerinde karşılaştırmalı model eğitimleri gerçekleştirmek ve elde edilen anomali skorlarını deprem katalogları ile ilişkilendirerek yorumlanabilir bir sonuç seti oluşturmaktır.

Başka bir ifadeyle, bu aşamada geliştirilecek yapı yalnızca bir model eğitimi süreci olmayacaktır. Aynı zamanda veri toplama, veri doğrulama, etiketleme, modelleme, değerlendirme, görselleştirme ve tez çıktısı üretme adımlarının tamamını kapsayan bütüncül bir araştırma hattı kurulacaktır.

# 3\. Yol haritasının ana fazları

Çalışma, uygulama açısından birbirini izleyen on bir ana faz altında yürütülmelidir. Her fazın sonunda ölçülebilir bir çıktı üretilmesi, sonraki adıma geçişte belirsizliği azaltacaktır.

## 3.1. Faz 1 - Kapsamın uygulama açısından netleştirilmesi

İlk adımda tezde anlatılan yöntem, kodlanabilir ve yönetilebilir bir teknik plana dönüştürülmelidir. Bu aşamada çalışma bölgesi, istasyon sayısı, zaman aralığı, minimum deprem büyüklüğü ve ilk aşamada kullanılacak görev tanımı kesinleştirilmelidir.

İlk sürüm için tek istasyon ile başlanması daha sağlıklı olacaktır. Veri hacmini, işlem maliyetini ve hata ayıklama sürecini kontrol altında tutmak açısından bu tercih önemlidir. Aynı şekilde ilk deneylerde altı aylık sürekli veri, hem yeterli hacim sağlayacak hem de proje yönetimini zorlaştırmayacaktır.

Bu fazın sonunda proje kapsamı yazılı hâle getirilmeli; hangi verinin, hangi eşiklerle, hangi amaçla kullanılacağı belgelendirilmelidir.

## 3.2. Faz 2 - Veri erişimi ve veri düzeninin kurulması

Bu fazda ham veri kaynaklarına erişim güvence altına alınmalıdır. Kandilli, AFAD, IRIS veya uygun diğer açık kaynaklardan MiniSEED formatındaki kayıtların indirilmesi için tekrar çalıştırılabilir komut dosyaları hazırlanmalıdır.

Aynı anda deprem katalog verileri de indirilmeli ve zaman damgaları bakımından ana veri hattı ile uyumlu hâle getirilmelidir. İstasyon kodu, kanal bilgisi, örnekleme frekansı, zaman aralığı ve veri bütünlüğü gibi alanları içeren düzenli bir metadata yapısı oluşturulmalıdır.

Bu fazın sonunda ham veri klasör yapısı, katalog yapısı ve veri bütünlük kontrolü çalışan bir düzene kavuşmalıdır.

## 3.3. Faz 3 - Ön işleme hattının geliştirilmesi

Ön işleme adımı, projenin bilimsel güvenilirliğini belirleyen temel aşamalardan biridir. MiniSEED verisinin okunması, kanal kontrolü, gerekirse yeniden örnekleme, bant geçiren süzgeç uygulanması, bozuk segmentlerin elenmesi ve ardından kayan pencereleme işlemleri bu aşamada tamamlanmalıdır.

Her pencere için yalnızca ham sinyal değil, aynı zamanda STFT gibi frekans-zaman temsilleri de üretilmelidir. Böylece modellerin hem zaman alanı hem de spektral alan üzerinden denenebilmesi sağlanacaktır.

Ön işleme hattı tamamlandığında, seçilen örnek pencereler hem ham dalga biçimi olarak hem de spektrogram görüntüsü olarak doğrulanmalıdır.

## 3.4. Faz 4 - Etiketleme stratejisinin oluşturulması

Uygulama safhasında en kritik kararlardan biri, pencere etiketlerinin hangi mantıkla üretileceğidir. İlk aşamada yarı gözetimli bir yaklaşım izlenmesi uygun görünmektedir. Buna göre belirli büyüklüğün üzerindeki depremlerden önceki belirli zaman aralıklarında kalan pencereler pozitif örnek olarak kabul edilecektir.

Bu aşamada üç saat, altı saat, on iki saat ve yirmi dört saat gibi farklı zaman pencereleri ayrı deney grupları olarak tasarlanmalıdır. Böylece deprem öncesi davranışların hangi zaman ölçeğinde daha anlamlı temsil edildiği sayısal olarak karşılaştırılabilir.

Etiketleme tamamlandığında pozitif ve negatif örnek sayıları, sınıf dengesizliği ve veri dağılımı açık biçimde raporlanmalıdır.

## 3.5. Faz 5 - Temel karşılaştırma modellerinin kurulması

Transformer tabanlı modellerden önce güçlü temel modeller oluşturulmalıdır. Bu amaçla klasik sinyal tabanlı bir eşik yöntemi ile LSTM ve GRU tabanlı zaman serisi modelleri ilk karşılaştırma omurgasını oluşturmalıdır.

Bu adım, daha sonra elde edilecek Transformer sonuçlarının anlamlı bir referans ile yorumlanmasını sağlayacaktır. Aksi durumda yalnızca tek model ailesi üzerinde ilerlemek, tez çalışmasının karşılaştırmalı niteliğini zayıflatabilir.

Bu fazın sonunda aynı veri bölünmesi ve aynı metrikler ile çalışan en az üç temel model hazır olmalıdır.

## 3.6. Faz 6 - Transformer çekirdeğinin kurulması

Temel modellerin ardından Transformer tabanlı mimarilerin uygulanmasına geçilmelidir. Öncelik sırası olarak önce daha yalın bir zaman serisi Transformer yapısı kurulmalı, ardından uzun dizilerde verimli çalışan mimariler eklenmelidir.

Bu aşamada ortak eğitim, doğrulama ve test akışı olan modüler bir yapı kurulması gerekir. Böylece farklı model mimarileri arasında yalnızca yapılandırma değiştirerek deney yapmak mümkün olur.

Model eğitimi sırasında kayıp eğrileri, doğrulama metrikleri ve kayıt dosyaları düzenli biçimde saklanmalıdır.

## 3.7. Faz 7 - Deney matrisinin hazırlanması

Çalışma, rastgele denemeler yerine sistematik bir deney planı ile yürütülmelidir. Girdi türü, model ailesi, etiketleme penceresi ve istasyon sayısı gibi değişkenler kontrollü biçimde değiştirilmelidir.

İlk aşamada tek istasyon ve STFT temsili ile sınırlı kalmak mantıklıdır. Daha sonra ham sinyal ile STFT karşılaştırması, farklı zaman pencereleri ve çoklu istasyon senaryosu ayrı deney setleri olarak ele alınmalıdır.

Bu fazın sonunda hangi deneyin hangi veri ile, hangi ayarlarla ve hangi amaçla çalıştırıldığı açık biçimde izlenebilir hâle gelmelidir.

## 3.8. Faz 8 - Değerlendirme ve istatistiksel analiz

Model başarısı yalnızca doğruluk veya F1 skoru ile değerlendirilmemelidir. Çalışmanın doğası gereği, anomali skorlarının gerçek deprem zamanları ile ne kadar anlamlı ilişkiler kurduğu da ayrıca incelenmelidir.

Bu nedenle precision, recall, F1, ROC-AUC ve PR-AUC gibi standart ölçütlerin yanında yüksek skorların deprem olaylarına göre zaman farkı dağılımı, büyük depremler öncesindeki skor değişimi ve rastgele dönemlerle yapılan karşılaştırmalar da değerlendirmeye dâhil edilmelidir.

Bu aşama tamamlandığında hem pencere düzeyinde hem de olay düzeyinde sonuç üreten bir analiz altyapısı hazır olmalıdır.

## 3.9. Faz 9 - Görsellerin ve tez çıktılarının üretilmesi

Tez raporunda daha sonra gerçek çıktılar ile doldurulması gereken birçok şekil bulunmaktadır. Uygulama süreci tamamlandıkça bu görsellerin tamamı doğrudan proje çıktılarından üretilmelidir.

Sistem mimarisi, veri akış şeması, örnek STFT görselleri, eğitim-doğrulama kayıp eğrileri, anomali skor zaman çizelgeleri ve ROC eğrileri bu aşamada hazırlanacaktır. Bu yaklaşım, tezde yer alacak şekillerin yalnızca temsili değil, gerçek deneylere dayalı olmasını sağlayacaktır.

Bu fazın sonunda tez metnine doğrudan eklenebilecek nitelikte düzenli, açıklamalı ve tekrar üretilebilir görseller elde edilmelidir.

## 3.10. Faz 10 - Prototip arayüzün hazırlanması

Tez çalışmasının uygulama yönünü görünür kılmak için temel bir gösterim arayüzü hazırlanması yararlı olacaktır. Bu arayüz, seçilen zaman aralığı içinde üretilen anomali skorlarını, ilgili deprem olaylarını ve istasyon seçimini görsel olarak sunmalıdır.

Başlangıç düzeyinde Python tabanlı hafif bir arayüz yeterli olacaktır. Amaç ticari bir ürün geliştirmek değil, geliştirilen yöntemin nasıl çalıştığını açık biçimde gösterebilen araştırma amaçlı bir prototip oluşturmaktır.

## 3.11. Faz 11 - Sonuçların tez metnine entegrasyonu

Uygulama tamamlandıktan sonra tez metninde yer alan yöntem, bulgular, tartışma ve sonuç bölümleri gerçek verilerle yeniden gözden geçirilmelidir. Özellikle performans tabloları, yorumlar ve model karşılaştırmaları deney sonuçlarına göre güncellenmelidir.

Bu aşamada veri seti istatistikleri, eğitim süreci, başarı metrikleri, yanlış alarm örnekleri, anomali-gözlem ilişkileri ve sınırlılıklar bilimsel biçimde yeniden yazılmalıdır.

Son düzenleme aşamasında tez metninin anlatımı ile uygulama çıktılarının tam uyumlu olması sağlanmalıdır.

# 4\. Öncelik sırasına göre uygulanacak çalışma düzeni

Uygulama aşamasında bütün adımları aynı anda yürütmeye çalışmak yerine, birbirine bağımlı görevleri doğru sırayla ilerletmek gerekmektedir. Buna göre izlenecek ana sıra şu şekilde olmalıdır: veri erişimi ve istasyon seçimi, veri indirme betiklerinin hazırlanması, ön işleme hattının kurulması, etiketleme düzeninin oluşturulması, temel modellerin geliştirilmesi, Transformer mimarilerinin eklenmesi, deneylerin yürütülmesi, sonuçların analiz edilmesi, görsellerin üretilmesi ve son olarak tez güncellemesinin yapılması.

Bu sıra korunursa her adım bir öncekinin ürettiği çıktıyı kullanır ve süreç boyunca belirsizlik azalır. Özellikle veri hattı tam çalışmadan model eğitimine geçilmemesi büyük önem taşımaktadır.

# 5\. Sprint 1 için ayrıntılı başlangıç planı

İlk sprintin amacı, tüm projeyi bir anda tamamlamak değil; sistemin omurgasını kuracak ilk çalışan veri hattını ortaya çıkarmaktır. Bu nedenle ilk sprint olabildiğince dar kapsamlı ancak tamamen çalışır yapıda tasarlanmalıdır.

İlk sprintte öncelikle tek bir istasyon seçilecek, ardından bu istasyona ait sınırlı bir zaman aralığındaki sürekli veri indirilecektir. Aynı zaman aralığı için deprem katalog verisi de toplanacak ve iki veri kaynağının zaman uyumu doğrulanacaktır.

Sonraki adımda MiniSEED verisi okunacak, kanal bilgileri kontrol edilecek, temel filtreleme yapılacak ve veri pencerelenecektir. Üretilen örnek pencereler için hem ham sinyal grafikleri hem de STFT temsilleri elde edilecektir.

Bu sprintin sonunda amaç model eğitmek değil, veri hattının güvenilir şekilde çalıştığını göstermektir. Başka bir ifadeyle Sprint 1 sonunda elimizde çalışan bir veri okuma, önişleme ve görselleştirme akışı bulunmalıdır.

1\. Çalışmada ilk kullanılacak istasyonun belirlenmesi.

2\. Seçilen istasyona ait kısa süreli sürekli verinin indirilmesi.

3\. Aynı tarih aralığı için deprem katalog verisinin elde edilmesi.

4\. MiniSEED verisinin okunması ve kanal yapısının doğrulanması.

5\. Temel filtreleme ve pencereleme hattının kurulması.

6\. STFT temsillerinin üretilmesi ve örnek görsellerin alınması.

7\. Üretilen çıktılar için klasör düzeninin ve kayıt yapısının oluşturulması.

# 6\. Son değerlendirme

Mevcut rapor, teorik çerçeve bakımından yeterli düzeye ulaşmış durumdadır. Bundan sonraki asıl ihtiyaç, kurulan yöntemin çalışan bir araştırma hattına dönüştürülmesidir. Bu nedenle süreç, önce veri ve deney altyapısını kurmaya; ardından sonuçları üretmeye; en son da bu çıktıları tez metnine sistemli biçimde yerleştirmeye dayanmalıdır.

Bu yol haritası izlendiği takdirde proje, dağınık bir geliştirme sürecinden çıkıp kontrol edilebilir, ölçülebilir ve akademik olarak savunulabilir bir yapıya kavuşacaktır. Özellikle Sprint 1, sonraki bütün adımların temelini oluşturacağı için dikkatli ve düzenli biçimde yürütülmelidir.