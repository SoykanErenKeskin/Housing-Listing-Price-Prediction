# Listing Outlier Cleaning Pipeline

Bu paket, listing portal ilan CSV dosyasını model eğitiminden önce temizlemek için oluşturuldu.

Amaç:
- Eksik veya gerçekçi olmayan fiyat ve m² değerlerini ayıklamak
- Brüt m² fiyatı uç değerlerini temizlemek
- Aynı ilan tekrar geldiyse duplicate satırları işaretlemek
- Temiz veri, silinen outlier verisi ve rapor üretmek

## Girdi

```text
data/input/listing_dataset.csv
```

## Çıktılar

```text
data/output/listing_dataset_cleaned.csv
data/output/listing_dataset_removed_outliers.csv
data/output/listing_dataset_marked_with_outlier_reason.csv
reports/outlier_cleaning_report.json
```

## Kullanım

```bash
python src/clean_outliers.py
```

## Temizleme mantığı

Pipeline şu adımları uygular:

1. Sayısal dönüşüm:
   - `price`
   - `gross_m2`
   - `unit_price_gross`

2. Eğer `unit_price_gross` boşsa hesaplar:

```text
unit_price_gross = price / gross_m2
```

3. Temel geçerlilik filtreleri:
   - çok düşük/yüksek brüt m²
   - çok düşük/yüksek m² fiyatı
   - çok düşük/yüksek toplam fiyat
   - eksik temel değerler

4. Duplicate filtre:
   - aynı `classified_id` tekrar ediyorsa ilk satır kalır, tekrar edenler silinenlere gider.

5. Global quantile filtre:
   - m² fiyatında alt %1 ve üst %1 uç değerleri işaretler.

6. Mahalle bazlı IQR filtre:
   - `city + county + district` gruplarında yeterli gözlem varsa mahalle içi uç m² fiyatlarını işaretler.

7. İlçe bazlı IQR filtre:
   - `city + county` düzeyinde daha geniş outlier kontrolü yapar.

## Bu veriyle hızlı sonuç

Ham satır: 200  
Temiz kalan: 190  
Silinen/işaretlenen: 10  
Silinen oran: 5.00%

## Ayar değiştirme

Limitleri ve filtre sertliğini `src/clean_outliers.py` içindeki `CONFIG` bölümünden değiştirebilirsin.

Örneğin daha yumuşak temizlik istersen:

```python
"lower_quantile": 0.005,
"upper_quantile": 0.995
```

Daha sert temizlik istersen:

```python
"lower_quantile": 0.02,
"upper_quantile": 0.98
```

## Tez için not

Bu pipeline outlier silmeyi şeffaf yapar. Silinen her satır için `outlier_reason` tutulur. Bu sayede tezde hangi kriterle kaç satırın çıkarıldığını açıkça raporlayabilirsin.

Önerilen tez cümlesi:

> Model eğitiminden önce ilan verisi üzerinde aykırı değer temizliği uygulanmıştır. Bu süreçte eksik veya gerçekçi olmayan fiyat/m² değerleri, tekrar eden ilanlar ve brüt m² fiyatı dağılımında uçta kalan gözlemler işaretlenmiştir. Aykırı değer temizliği hem genel dağılım düzeyinde hem de mahalle/ilçe bazlı IQR kontrolleriyle gerçekleştirilmiştir.
