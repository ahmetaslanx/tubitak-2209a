import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

def dinamik_matris_olustur(mesaj_bit_uzunlugu, ortak_tohum):
    np.random.seed(ortak_tohum)
    katsayilar_0 = np.random.uniform(1.01, 1.09, mesaj_bit_uzunlugu)
    katsayilar_1 = np.random.uniform(0.91, 0.99, mesaj_bit_uzunlugu)
    return katsayilar_0, katsayilar_1

# =============================================================================
# 1. AŞAMA: TEORİK ALTYAPI (NOTA SÖZLÜĞÜ) VE GİZLİ VERİ ŞİFRELEME
# =============================================================================
# Gelecekte M-ary FSK kapasite artırımı için tüm nota referansları tanımlanmıştır.
nota_frekans = {
    'C4': 261.63,  # Do (Mevcut sistemde '0' bitini temsil eder)
    'D4': 293.66,  # Re 
    'E4': 329.63,  # Mi
    'F4': 349.23,  # Fa
    'G4': 392.00,  # Sol (Mevcut sistemde '1' bitini temsil eder)
    'A4': 440.00,  # La
    'B4': 493.88   # Si
}

gizli_mesaj = "TEST MESAJI - 1,TEST MESAJI - 1,TEST MESAJI - 1"

# Metni karakter karakter ASCII tablosu üzerinden 8-bitlik ikilik (binary) koda çeviriyoruz
ikilik_kod = ""
for harf in gizli_mesaj:
    ikilik_kod += format(ord(harf), '08b')

# Ortak tohumu ve uzunluğu belirliyoruz
ortak_tohum = 35
mesaj_bit_uzunlugu = len(ikilik_kod)

# Dinamik matrisi oluşturuyoruz (girdi ve parametre isimleri aynı)
katsayilar_0, katsayilar_1 = dinamik_matris_olustur(
    mesaj_bit_uzunlugu=mesaj_bit_uzunlugu, 
    ortak_tohum=ortak_tohum
)

# Ses Sinyali Sentezleme Parametreleri
ornekleme_frekansi = 44100
bit_suresi = 0.1 # Veri kapasitesini artırmak için optimize edilmiş süre
t = np.linspace(0, bit_suresi, int(ornekleme_frekansi * bit_suresi), endpoint=False)
uretilen_ses_sinyali = np.array([])

# 0 ve 1'leri trigonometrik dalgalara dönüştürme döngüsü (i indeksi eklendi)
for i, bit in enumerate(ikilik_kod):
    # O anki bit'e özel frekansı dinamik matristen çekiyoruz
    sifreli_frekans_0 = nota_frekans['C4'] * katsayilar_0[i]
    sifreli_frekans_1 = nota_frekans['G4'] * katsayilar_1[i]
    
    if bit == '0':
        dalga = np.sin(2 * np.pi * sifreli_frekans_0 * t)
    else:
        dalga = np.sin(2 * np.pi * sifreli_frekans_1 * t)
    uretilen_ses_sinyali = np.concatenate((uretilen_ses_sinyali, dalga))

# =============================================================================
# 2. AŞAMA: STEGANOGRAFİK ÖRTME İŞLEMİ
# =============================================================================
# Örtü müziğini ses dosyasını içe aktar ve formatını eşitle
ornekleme_muzik, muzik_sinyali = wavfile.read("ses2 109bpm.wav")
if len(muzik_sinyali.shape) == 2:
    muzik_sinyali = muzik_sinyali.mean(axis=1)

# Müziği normalize et ve gizli verinin genliğini %5 (0.05) seviyesine düşür
muzik_sinyali = muzik_sinyali / np.max(np.abs(muzik_sinyali))
kisik_sifreli_ses = uretilen_ses_sinyali * 0.05               

# Boyutları eşitleyip iki dalgayı topluyoruz (Superpozisyon)
min_uzunluk = min(len(muzik_sinyali), len(kisik_sifreli_ses))
muzik_sinyali = muzik_sinyali[:min_uzunluk]
kisik_sifreli_ses = kisik_sifreli_ses[:min_uzunluk]

temiz_sifreli_muzik = muzik_sinyali + kisik_sifreli_ses
temiz_sifreli_muzik = temiz_sifreli_muzik / np.max(np.abs(temiz_sifreli_muzik)) # Yeniden Normalizasyon

# Kusursuz stego-sinyali kaydet
wavfile.write("son_sifreli_muzik.wav", ornekleme_muzik, temiz_sifreli_muzik.astype(np.float32))

# =============================================================================
# 3. AŞAMA: RİSK SİMÜLASYONLARI (STRES TESTİ DOSYALARININ ÜRETİMİ)
# =============================================================================
# Risk 1: Yüksek Seviyeli AWGN (İlaveli Beyaz Gauss Gürültüsü)
noise_level = 0.3
gurultu = np.random.normal(0, noise_level, temiz_sifreli_muzik.shape)
gurultulu_muzik = temiz_sifreli_muzik + gurultu
gurultulu_muzik = gurultulu_muzik / np.max(np.abs(gurultulu_muzik))
wavfile.write("gurultulu_sifreli_muzik.wav", ornekleme_muzik, gurultulu_muzik.astype(np.float32))

# Risk 2: Zaman Kayması / Kırpma (Desenkronizasyon Testi)
# Dosyanın en başından tam 0.05 saniye (yarım bit) kesip atıyoruz.
kirpilacak_ornek_sayisi = int(ornekleme_muzik * 0.05)
zaman_kaymali_muzik = temiz_sifreli_muzik[kirpilacak_ornek_sayisi:]
wavfile.write("zaman_kaymali_muzik.wav", ornekleme_muzik, zaman_kaymali_muzik.astype(np.float32))

# =============================================================================
# 4. AŞAMA: SİNYAL ÇÖZÜMLEME (FFT DECODING) VE İSTATİSTİKSEL BER ANALİZİ
# =============================================================================
def test_et_ve_raporla(dosya_adi, ortak_tohum, bit_suresi=0.1, orijinal_ikilik_kod=ikilik_kod):
    print(f"\n{'='*60}\nTEST EDİLİYOR: {dosya_adi}\n{'='*60}")
    
    try:
        _, analiz_edilecek_ses = wavfile.read(dosya_adi)
    except FileNotFoundError:
        print(f"HATA: {dosya_adi} bulunamadı!")
        return 1.0 # Hata durumunda döngü kırılmasın diye

    if len(analiz_edilecek_ses.shape) == 2: 
        analiz_edilecek_ses = analiz_edilecek_ses.mean(axis=1)

    ornek_sayisi_per_bit = int(ornekleme_muzik * bit_suresi)
    toplam_iletilen_bit = len(analiz_edilecek_ses) // ornek_sayisi_per_bit
    cozulen_ikilik_kod = ""

    # Alıcı taraf aynı tohumla matrisi yeniden inşa ediyor
    cozum_katsayilari_0, cozum_katsayilari_1 = dinamik_matris_olustur(
        mesaj_bit_uzunlugu=toplam_iletilen_bit, 
        ortak_tohum=ortak_tohum
    )

    # Sesi pencerelere bölüp her birine FFT uyguluyoruz
    for i in range(toplam_iletilen_bit):
        baslangic_indeksi = i * ornek_sayisi_per_bit
        bitis_indeksi = baslangic_indeksi + ornek_sayisi_per_bit
        ses_penceresi = analiz_edilecek_ses[baslangic_indeksi : bitis_indeksi]
        
        frekanslar = fftfreq(len(ses_penceresi), 1/ornekleme_muzik)
        genlikler = np.abs(fft(ses_penceresi))
        
        pozitif_frekanslar = frekanslar[:len(frekanslar)//2]
        pozitif_genlikler = genlikler[:len(genlikler)//2]
        
        anlik_hedef_frekans_0 = nota_frekans['C4'] * cozum_katsayilari_0[i]
        anlik_hedef_frekans_1 = nota_frekans['G4'] * cozum_katsayilari_1[i]
        
        idx_0 = np.argmin(np.abs(pozitif_frekanslar - anlik_hedef_frekans_0))
        idx_1 = np.argmin(np.abs(pozitif_frekanslar - anlik_hedef_frekans_1))
        
        if pozitif_genlikler[idx_0] > pozitif_genlikler[idx_1]:
            cozulen_ikilik_kod += '0'
        else:
            cozulen_ikilik_kod += '1'

    cozulmus_metin = ""
    for i in range(0, len(cozulen_ikilik_kod), 8):
        byte = cozulen_ikilik_kod[i:i+8]
        if len(byte) == 8:
            cozulmus_metin += chr(int(byte, 2))
            
    print(f"Çözülen Metin: {cozulmus_metin}")
    
    # BER (Bit Error Rate) Hesaplama
    hatali_bit_sayisi = 0
    min_uzunluk = min(len(orijinal_ikilik_kod), len(cozulen_ikilik_kod))
    for i in range(min_uzunluk):
        if orijinal_ikilik_kod[i] != cozulen_ikilik_kod[i]:
            hatali_bit_sayisi += 1
            
    hatali_bit_sayisi += abs(len(orijinal_ikilik_kod) - len(cozulen_ikilik_kod))
    ber_orani = hatali_bit_sayisi / len(orijinal_ikilik_kod)
    
    print(f"Toplam İletilen Bit : {len(orijinal_ikilik_kod)}")
    print(f"Hatalı Çözülen Bit  : {hatali_bit_sayisi}")
    print(f"Bit Hata Oranı (BER): {ber_orani:.4f} (%{ber_orani*100:.2f})")
    
    return ber_orani

# =============================================================================
# 5. AŞAMA: İSTATİSTİKSEL GÜRÜLTÜ ANALİZİ VE GÖRSELLEŞTİRME
# =============================================================================

# Temel düzeyde çizim yapan fonksiyonumuz (Toplam bit sayısını başlığa yansıtır)
def ber_grafigi_ciz(test_edilecek_gurultu_seviyeleri, elde_edilen_ber_degerleri, toplam_bit_sayisi):
    plt.figure(figsize=(8, 5))
    # Gürültü arttıkça hatanın nasıl değiştiğini görmek için çizgi grafiği (plot) kullanıyoruz
    plt.plot(test_edilecek_gurultu_seviyeleri, elde_edilen_ber_degerleri, marker='o', color='red', linewidth=2)
    
    # Başlığa toplam iletilen bit miktarını ekliyoruz
    plt.title(f"Gürültü Seviyesi ve Hata Oranı (BER) İlişkisi\n(Toplam İletilen Veri Paketi: {toplam_bit_sayisi} Bit)")
    plt.xlabel("Gürültü Seviyesi (Standart Sapma - σ)")
    plt.ylabel("Bit Hata Oranı (BER)")
    plt.grid(True)
    plt.show(block=True)


# Raporda belirttiğimiz 3 farklı test senaryosu
test_edilecek_gurultu_seviyeleri = [0.05, 0.30, 0.50]
elde_edilen_ber_degerleri = []

# Temiz müziği referans olarak içe aktarıyoruz
ornekleme, temiz_sinyal = wavfile.read("son_sifreli_muzik.wav")
if len(temiz_sinyal.shape) == 2:
    temiz_sinyal = temiz_sinyal.mean(axis=1)

print("\n--- OTOMATİK STRES TESTİ BAŞLIYOR ---")

# Döngü sırayla her bir gürültü seviyesini uygulayıp test edecek
for gurultu_seviyesi in test_edilecek_gurultu_seviyeleri:
    
    # 1. Gürültüyü oluştur ve müziğe ekle
    np.random.seed(42) # Testlerin hep aynı matematiksel gürültüyle standart olması için
    gurultu = np.random.normal(0, gurultu_seviyesi, temiz_sinyal.shape)
    bozulmus_sinyal = temiz_sinyal + gurultu
    bozulmus_sinyal = bozulmus_sinyal / np.max(np.abs(bozulmus_sinyal))
    
    # 2. Bozulmuş dosyayı geçici olarak kaydet
    gecici_dosya_adi = f"gurultu_test_{gurultu_seviyesi}.wav"
    wavfile.write(gecici_dosya_adi, ornekleme, bozulmus_sinyal.astype(np.float32))
    
    # 3. Test fonksiyonunu çağır (Girdi ve parametre isimleri aynı)
    anlik_ber = test_et_ve_raporla(
        dosya_adi=gecici_dosya_adi,
        ortak_tohum=ortak_tohum
    )
    
    # 4. Çıkan sonucu grafikte göstermek için listeye ekle
    elde_edilen_ber_degerleri.append(anlik_ber)


# Testler bitti, şimdi grafik fonksiyonunu çağırıyoruz (Girdi ve parametre isimleri aynı)
ber_grafigi_ciz(
    test_edilecek_gurultu_seviyeleri=test_edilecek_gurultu_seviyeleri,
    elde_edilen_ber_degerleri=elde_edilen_ber_degerleri,  # Doğru liste adı buraya yazıldı
    toplam_bit_sayisi=len(ikilik_kod)  # Otomatik olarak 376 değerini başlığa işler
)

# =============================================================================
# KAPASİTE (BAUD RATE) TEST OTOMASYONU VE GÖRSELLEŞTİRME
# =============================================================================
# Veri İletim Hızı (Baud Rate) ve Kapasite Grafiği Çizdirme Fonksiyonu
def kapasite_grafigi_ciz(test_edilen_bit_sureleri, elde_edilen_ber_ucretleri, toplam_bit_sayisi):
    # Bit süresini Baud oranına (saniyedeki bit sayısına: 1 / bit_suresi) çeviriyoruz
    baud_oranlari = [1 / sure for sure in test_edilen_bit_sureleri]
    
    plt.figure(figsize=(8, 5))
    # Hız arttıkça hatanın değişimini görmek için çizgi ve işaretçiler (marker) kullanıyoruz
    plt.plot(baud_oranlari, elde_edilen_ber_ucretleri, marker='s', color='blue', linestyle='-', linewidth=2)
    
    # Başlığa toplam iletilen bit miktarını dinamik olarak ekliyoruz
    plt.title(f"Veri İletim Hızı (Baud Rate) ve Hata Oranı (BER) İlişkisi\n(Toplam İletilen Veri Paketi: {toplam_bit_sayisi} Bit)")
    plt.xlabel("Kapasite / Hız (Baud / bit-sn)")
    plt.ylabel("Bit Hata Oranı (BER)")
    plt.grid(True)
    
    # Grafiğin ekranda anında açılıp kalmasını sağlayan komut
    plt.show(block=True)

def kapasite_testi_yap(test_edilecek_bit_sureleri, orijinal_ikilik_kod, ortak_tohum):
    print("\n" + "="*60)
    print("FARKLI KAPASİTE (BAUD RATE) TESTLERİ BAŞLIYOR")
    print("="*60)
    
    kaydedilen_ber_degerleri = []
    
    for bit_suresi in test_edilecek_bit_sureleri:
        baud_orani = 1 / bit_suresi
        print(f"\n--- TEST EDİLEN BİT SÜRESİ: {bit_suresi} sn (Kapasite: {baud_orani:.1f} Baud / bit-sn) ---")
        
        t = np.linspace(0, bit_suresi, int(ornekleme_frekansi * bit_suresi), endpoint=False)
        uretilen_ses_sinyali = np.array([])
        
        katsayilar_0, katsayilar_1 = dinamik_matris_olustur(
            mesaj_bit_uzunlugu=len(orijinal_ikilik_kod), 
            ortak_tohum=ortak_tohum
        )
        
        for i, bit in enumerate(orijinal_ikilik_kod):
            sifreli_frekans_0 = nota_frekans['C4'] * katsayilar_0[i]
            sifreli_frekans_1 = nota_frekans['G4'] * katsayilar_1[i]
            
            if bit == '0':
                dalga = np.sin(2 * np.pi * sifreli_frekans_0 * t)
            else:
                dalga = np.sin(2 * np.pi * sifreli_frekans_1 * t)
            uretilen_ses_sinyali = np.concatenate((uretilen_ses_sinyali, dalga))
            
        ornekleme_muzik, muzik_sinyali = wavfile.read("ses2 109bpm.wav")
        if len(muzik_sinyali.shape) == 2:
            muzik_sinyali = muzik_sinyali.mean(axis=1)
            
        muzik_sinyali = muzik_sinyali / np.max(np.abs(muzik_sinyali))
        kisik_sifreli_ses = uretilen_ses_sinyali * 0.05
        
        min_uzunluk = min(len(muzik_sinyali), len(kisik_sifreli_ses))
        muzik_sinyali = muzik_sinyali[:min_uzunluk]
        kisik_sifreli_ses = kisik_sifreli_ses[:min_uzunluk]
        
        test_muzigi = muzik_sinyali + kisik_sifreli_ses
        test_muzigi = test_muzigi / np.max(np.abs(test_muzigi))
        
        gecici_dosya_adi = f"kapasite_test_{bit_suresi}sn.wav"
        wavfile.write(gecici_dosya_adi, ornekleme_muzik, test_muzigi.astype(np.float32))
        
        # Test fonksiyonundan dönen BER oranını değişkene alıyoruz
        anlik_ber = test_et_ve_raporla(
            dosya_adi=gecici_dosya_adi, 
            ortak_tohum=ortak_tohum, 
            bit_suresi=bit_suresi, 
            orijinal_ikilik_kod=orijinal_ikilik_kod
        )
        kaydedilen_ber_degerleri.append(anlik_ber)

# Testler bittiğinde otomatik olarak kapasite grafiğini çizdiriyoruz
    kapasite_grafigi_ciz(
        test_edilen_bit_sureleri=test_edilecek_bit_sureleri,
        elde_edilen_ber_ucretleri=kaydedilen_ber_degerleri,
        toplam_bit_sayisi=len(orijinal_ikilik_kod)  # Otomatik olarak 376 değerini başlığa işler
    )

# ---- ÇALIŞTIRMA ÖRNEĞİ ----
# Test edilecek bit sürelerini belirleyip fonksiyonu çağırıyoruz:
test_edilecek_bit_sureleri = [0.1, 0.05, 0.02]

kapasite_testi_yap(
    test_edilecek_bit_sureleri=test_edilecek_bit_sureleri,
    orijinal_ikilik_kod=ikilik_kod,
    ortak_tohum=ortak_tohum
)