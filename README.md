# Laporan Proyek Machine Learning - Fajar

## Domain Proyek: Hospitality & Tourism (Perhotelan dan Pariwisata)

![image](https://github.com/user-attachments/assets/5193f0d3-5d4b-4c56-82ac-39aa02845235)


Industri perhotelan merupakan salah satu sektor ekonomi yang sangat dipengaruhi oleh fluktuasi permintaan pasar, musim, dan perilaku pelanggan. Dalam upaya untuk meningkatkan efisiensi operasional dan pendapatan, banyak hotel mulai mengadopsi teknologi berbasis data, salah satunya adalah penerapan machine learning (ML) untuk memprediksi harga kamar (Average Daily Rate/ADR) dan total biaya menginap. Prediksi yang akurat terhadap ADR dan total_cost dapat membantu manajer hotel dalam mengoptimalkan strategi penetapan harga serta meningkatkan pendapatan dan kepuasan pelanggan (Chen & Kuo, 2015).

Seiring dengan meningkatnya volume data historis pemesanan hotel dan kemajuan algoritma prediktif, metode machine learning menjadi solusi yang relevan dalam membantu pengambilan keputusan strategis di sektor ini. Penggunaan algoritma seperti Random Forest dan XGBoost dalam konteks regresi multi-target terbukti mampu memberikan hasil yang kompetitif dan lebih adaptif terhadap variasi data dibanding metode konvensional (Li et al., 2016).

## Business Understanding

### Problem Statements

Rumusan masalah dari latar belakang di atas adalah:

- Kapan waktu terbaik untuk memesan hotel agar mendapatkan harga terbaik?
- Berapa lama idealnya menginap untuk mengoptimalkan biaya?

### Goals

Tujuan dari analisis ini adalah:

- Mengetahui pola musiman dan waktu dalam setahun yang paling tepat untuk melakukan pemesanan hotel dengan harga terbaik.
- Mengidentifikasi durasi menginap yang memberikan efisiensi biaya tertinggi bagi pelanggan berdasarkan data historis.
 
### Solution statements
- Melakukan eksplorasi data (EDA) untuk melihat tren harga harian berdasarkan waktu pemesanan (bulan, hari, musim) dan lama tinggal.Menggunakan visualisasi data (line chart, boxplot, heatmap) untuk menggambarkan hubungan antara waktu dan harga, serta durasi menginap dengan biaya per malam.
- Membangun model regresi sederhana (jika diperlukan) untuk memprediksi harga harian berdasarkan waktu pemesanan dan durasi menginap.Memberikan rekomendasi praktis bagi konsumen dan manajemen hotel berdasarkan hasil analisis tren harga.

## Data Understanding
**Informasi Dataset**

Dataset **Hotel Booking Demand** digunakan untuk menganalisis dan memprediksi pola pemesanan hotel. Dataset ini memuat informasi terkait pemesanan dari dua jenis hotel (Resort Hotel dan City Hotel), serta mencakup atribut seperti tanggal pemesanan, durasi menginap, jumlah tamu, tipe kamar, harga harian (ADR), total biaya, dan status pembatalan. Dataset ini sangat relevan untuk pengembangan model prediksi harga dan analisis perilaku pelanggan di industri perhotelan.

Dataset ini berasal dari situs **Kaggle**, dan telah banyak digunakan dalam penelitian serta kompetisi data science karena kelengkapan dan kompleksitas fitur-fiturnya.

**Tabel Informasi Dataset**

| Jenis         | Keterangan                                                                                      |
|---------------|-------------------------------------------------------------------------------------------------|
| **Title**     | Hotel Booking Demand                                                                            |
| **Source**    | [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)                    |
| **Owner**     | Jesse Mostipak                                                                                  |
| **License**   | Unknown                                                                                         |
| **Visibility**| Publik                                                                                          |
| **Tags**      | Hotel, Booking, Demand, Cancellations, Data Science, Machine Learning                          |

### Penjelasan Variabel Dataset Hotel Booking Demand

| Variabel                         | Deskripsi Singkat                                                              |
|----------------------------------|---------------------------------------------------------------------------------|
| `hotel`                         | Jenis hotel: *Resort Hotel* atau *City Hotel*.                                 |
| `is_canceled`                   | Status pembatalan (1 = dibatalkan, 0 = tidak).                                 |
| `lead_time`                     | Jumlah hari antara pemesanan dan tanggal kedatangan.                           |
| `arrival_date_year`            | Tahun kedatangan tamu.                                                          |
| `arrival_date_month`           | Bulan kedatangan tamu.                                                          |
| `arrival_date_week_number`     | Minggu keberapa dalam tahun saat kedatangan.                                   |
| `arrival_date_day_of_month`    | Tanggal kedatangan tamu.                                                        |
| `stays_in_weekend_nights`      | Lama inap pada akhir pekan (Sabtu/Minggu).                                     |
| `stays_in_week_nights`         | Lama inap pada hari kerja (Senin–Jumat).                                       |
| `adults`                        | Jumlah orang dewasa.                                                            |
| `children`                      | Jumlah anak-anak.                                                               |
| `babies`                        | Jumlah bayi.                                                                    |
| `meal`                          | Tipe paket makan yang dipesan.                                                  |
| `country`                       | Negara asal pelanggan.                                                          |
| `market_segment`               | Segmen pemasaran, misalnya *Online TA*, *Direct*, dll.                         |
| `distribution_channel`        | Saluran distribusi pemesanan.                                                   |
| `is_repeated_guest`            | Apakah pelanggan pernah menginap sebelumnya (1 = ya, 0 = tidak).               |
| `previous_cancellations`       | Jumlah pembatalan sebelumnya oleh pelanggan tersebut.                           |
| `previous_bookings_not_canceled`| Jumlah pemesanan sebelumnya yang tidak dibatalkan.                             |
| `reserved_room_type`           | Tipe kamar yang dipesan.                                                        |
| `assigned_room_type`           | Tipe kamar yang diberikan.                                                      |
| `booking_changes`              | Jumlah perubahan yang dilakukan pada pemesanan.                                 |
| `deposit_type`                 | Jenis deposit: *No Deposit*, *Non Refund*, *Refundable*.                       |
| `agent`                         | ID agen yang melakukan pemesanan.                                               |
| `company`                       | ID perusahaan yang melakukan pemesanan.                                         |
| `days_in_waiting_list`         | Jumlah hari pemesanan menunggu konfirmasi.                                      |
| `customer_type`                | Tipe pelanggan: *Transient*, *Group*, dll.                                     |
| `adr`                           | *Average Daily Rate*: harga kamar per malam.                                   |
| `required_car_parking_spaces` | Jumlah tempat parkir yang diminta.                                             |
| `total_of_special_requests`   | Jumlah permintaan khusus (seperti tempat tidur bayi, dll.).                     |
| `reservation_status`          | Status akhir reservasi (*Check-Out*, *Canceled*, dll.).                        |
| `reservation_status_date`     | Tanggal ketika status reservasi ditentukan.                                     |


### Struktur Dataset

Berikut adalah informasi awal mengenai struktur dataset Hotel Booking Demand berdasarkan hasil `df.info()`:

| No | Kolom                          | Non-Null Count | Dtype     | Keterangan                                 |
|----|--------------------------------|----------------|-----------|--------------------------------------------|
| 1  | hotel                          | 119,390        | object    | Jenis hotel (*Resort Hotel* atau *City Hotel*) |
| 2  | is_canceled                    | 119,390        | int64     | Status pembatalan reservasi (1 = dibatalkan) |
| 3  | lead_time                      | 119,390        | int64     | Jumlah hari antara pemesanan dan kedatangan |
| 4  | arrival_date_year             | 119,390        | int64     | Tahun kedatangan                            |
| 5  | arrival_date_month            | 119,390        | object    | Bulan kedatangan                            |
| 6  | arrival_date_week_number      | 119,390        | int64     | Minggu keberapa dalam tahun saat kedatangan |
| 7  | arrival_date_day_of_month     | 119,390        | int64     | Tanggal kedatangan                          |
| 8  | stays_in_weekend_nights       | 119,390        | int64     | Lama inap akhir pekan                       |
| 9  | stays_in_week_nights          | 119,390        | int64     | Lama inap hari kerja                        |
| 10 | adults                         | 119,390        | int64     | Jumlah orang dewasa                         |
| 11 | children                       | 119,386        | float64   | Jumlah anak-anak                            |
| 12 | babies                         | 119,390        | int64     | Jumlah bayi                                 |
| 13 | meal                           | 119,390        | object    | Jenis paket makan                           |
| 14 | country                        | 118,902        | object    | Negara asal tamu                            |
| 15 | market_segment                 | 119,390        | object    | Segmen pasar (misal: Online TA, Direct)    |
| 16 | distribution_channel           | 119,390        | object    | Saluran distribusi pemesanan                |
| 17 | is_repeated_guest              | 119,390        | int64     | Apakah tamu pernah menginap sebelumnya      |
| 18 | previous_cancellations         | 119,390        | int64     | Jumlah pembatalan sebelumnya                |
| 19 | previous_bookings_not_canceled| 119,390        | int64     | Jumlah pemesanan sebelumnya yang jadi       |
| 20 | reserved_room_type             | 119,390        | object    | Tipe kamar yang dipesan                     |
| 21 | assigned_room_type             | 119,390        | object    | Tipe kamar yang diberikan                   |
| 22 | booking_changes                | 119,390        | int64     | Jumlah perubahan pada reservasi             |
| 23 | deposit_type                   | 119,390        | object    | Jenis deposit (*No Deposit*, dll)           |
| 24 | agent                          | 103,050        | float64   | ID agen pemesanan (banyak missing)          |
| 25 | company                        | 6,797          | float64   | ID perusahaan pemesan (banyak missing)      |
| 26 | days_in_waiting_list           | 119,390        | int64     | Lama menunggu di waiting list               |
| 27 | customer_type                  | 119,390        | object    | Jenis tamu (*Transient*, *Group*, dll)      |
| 28 | adr                            | 119,390        | float64   | Harga rata-rata per malam (*Average Daily Rate*) |
| 29 | required_car_parking_spaces    | 119,390        | int64     | Jumlah tempat parkir yang diminta           |
| 30 | total_of_special_requests      | 119,390        | int64     | Jumlah permintaan khusus                    |
| 31 | reservation_status             | 119,390        | object    | Status reservasi akhir (Check-Out, dll)     |
| 32 | reservation_status_date        | 119,390        | object    | Tanggal status reservasi ditentukan         |

**Catatan:**
- Beberapa kolom memiliki nilai **missing** (null), seperti:
  - `children` (4 nilai kosong)
  - `country` (~488 nilai kosong)
  - `agent` (~16 ribu kosong)
  - `company` (~112 ribu kosong, sangat spars)

- Tipe data `object` menunjukkan data kategorikal, sedangkan `int64` dan `float64` adalah numerik.

Berikut ini adalah ringkasan statistik dari variabel numerik dalam dataset:

| Variabel                         | Count     | Mean     | Std Dev | Min   | 25%   | Median | 75%   | Max     | Penjelasan Singkat                                                                 |
|----------------------------------|-----------|----------|---------|-------|-------|--------|-------|---------|------------------------------------------------------------------------------------|
| is_canceled                      | 119390    | 0.370    | 0.483   | 0     | 0     | 0      | 1     | 1       | 37% pemesanan dibatalkan                                                          |
| lead_time                        | 119390    | 104.01   | 106.86  | 0     | 18    | 69     | 160   | 737     | Rata-rata tamu memesan 104 hari sebelum kedatangan                                |
| arrival_date_year                | 119390    | 2016.16  | 0.71    | 2015  | 2016  | 2016   | 2017  | 2017    | Data mencakup tahun 2015–2017                                                      |
| arrival_date_week_number         | 119390    | 27.17    | 13.61   | 1     | 16    | 28     | 38    | 53      | Sebaran minggu kedatangan selama setahun                                          |
| arrival_date_day_of_month        | 119390    | 15.80    | 8.78    | 1     | 8     | 16     | 23    | 31      | Tanggal kedatangan tersebar merata                                                 |
| stays_in_weekend_nights          | 119390    | 0.93     | 0.99    | 0     | 0     | 1      | 2     | 19      | Rata-rata tamu menginap 1 malam akhir pekan                                        |
| stays_in_week_nights             | 119390    | 2.50     | 1.91    | 0     | 1     | 2      | 3     | 50      | Rata-rata menginap 2–3 malam hari kerja                                            |
| adults                           | 119390    | 1.86     | 0.58    | 0     | 2     | 2      | 2     | 55      | Umumnya terdiri dari 2 orang dewasa                                                |
| children                         | 119386    | 0.10     | 0.40    | 0     | 0     | 0      | 0     | 10      | Mayoritas tamu tidak membawa anak                                                  |
| babies                           | 119390    | 0.008    | 0.097   | 0     | 0     | 0      | 0     | 10      | Sangat sedikit yang membawa bayi                                                   |
| is_repeated_guest                | 119390    | 0.032    | 0.176   | 0     | 0     | 0      | 0     | 1       | Hanya 3.2% tamu adalah pelanggan yang kembali                                      |
| previous_cancellations           | 119390    | 0.087    | 0.844   | 0     | 0     | 0      | 0     | 26      | Mayoritas tamu belum pernah membatalkan sebelumya                                 |
| previous_bookings_not_canceled   | 119390    | 0.137    | 1.497   | 0     | 0     | 0      | 0     | 72      | Sedikit tamu memiliki riwayat pemesanan sebelumnya                                 |
| booking_changes                  | 119390    | 0.22     | 0.65    | 0     | 0     | 0      | 0     | 21      | Umumnya tidak banyak perubahan pada reservasi                                      |
| agent                            | 103050    | 86.69    | 110.77  | 1     | 9     | 14     | 229   | 535     | Variasi besar ID agen (banyak missing values)                                      |
| company                          | 6797      | 189.27   | 131.66  | 6     | 62    | 179    | 270   | 543     | Hanya sedikit pemesanan melalui perusahaan (banyak missing values)                |
| days_in_waiting_list             | 119390    | 2.32     | 17.59   | 0     | 0     | 0      | 0     | 391     | Umumnya tidak ada masa tunggu, tapi ada kasus ekstrem                              |
| adr                              | 119390    | 101.83   | 50.54   | -6.38 | 69.29 | 94.58  | 126   | 5400    | Harga per malam rata-rata €101, ada outlier ekstrem                                |
| required_car_parking_spaces      | 119390    | 0.063    | 0.245   | 0     | 0     | 0      | 0     | 8       | Jarang tamu meminta tempat parkir                                                  |
| total_of_special_requests        | 119390    | 0.571    | 0.793   | 0     | 0     | 0      | 1     | 5       | Mayoritas tamu tidak memiliki permintaan khusus                                    |

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

| Jumlah Baris | Jumlah Kolom |
|--------------|--------------|
| 119390       | 32           |

**Data duplikat**
  
| Data Duplikat |
|---------------|
| 31994         |

### Explore Data

![image](https://github.com/user-attachments/assets/3f2f5bcd-6f9f-4650-8b09-f4889cdbe1fa)

**1. Hotel**
Dataset terdiri dari dua jenis hotel, yaitu:
- **City Hotel** sebanyak **79.330** sampel (**66,45%**), menunjukkan mayoritas data berasal dari hotel yang berada di area perkotaan.
- **Resort Hotel** sebanyak **40.060** sampel (**33,55%**), menunjukkan proporsi yang lebih kecil dibanding City Hotel.
  
---
![image](https://github.com/user-attachments/assets/8612f8bc-7b7d-4a25-ac28-a5c908c6e16a)

**2. Arrival Date Month**
Distribusi kedatangan tamu berdasarkan bulan menunjukkan pola musiman yang cukup jelas:
- Bulan **Agustus (11,62%)**, **Juli (10,60%)**, dan **Mei (9,88%)** merupakan bulan dengan jumlah kedatangan tertinggi.
- Sementara bulan dengan jumlah kedatangan terendah adalah **Januari (4,97%)**, **Desember (5,68%)**, dan **November (5,69%)**.

---
![image](https://github.com/user-attachments/assets/956f3f45-31e2-4dd4-bb7a-ad5363e3f227)

**3. Meal**
Tipe paket makan yang paling umum dipesan oleh tamu adalah:
- **BB (Bed & Breakfast)** sebanyak **92.310** sampel (**77,32%**), menunjukkan bahwa sebagian besar tamu memilih paket menginap dengan sarapan.
- Diikuti oleh **HB (Half Board)** sebanyak **14.463** sampel (**12,11%**) dan **SC (Self Catering)** sebanyak **10.650** sampel (**8,92%**).
- Kategori **Undefined** dan **FB (Full Board)** sangat sedikit, menunjukkan pilihan yang jarang digunakan.

---

![image](https://github.com/user-attachments/assets/a44ca117-88d2-446a-adee-69d6ea75cb4a)

**4. Market Segment**
Distribusi segmen pasar menunjukkan bagaimana tamu memesan hotel:
- **Online Travel Agent (OTA)** mendominasi dengan **56.477** sampel (**47,30%**), menunjukkan tren pemesanan hotel melalui platform digital.
- Diikuti oleh **Offline TA/TO (20,29%)**, **Groups (16,59%)**, dan **Direct booking (10,56%)**.
- Segmen **Corporate**, **Complementary**, dan **Aviation** memiliki proporsi sangat kecil, mengindikasikan pasar yang lebih spesifik atau terbatas.

---

![image](https://github.com/user-attachments/assets/8e592f91-1d41-4175-934d-865eb0c49bb5)

**5. Customer Type**
Tipe pelanggan yang paling dominan adalah:
- **Transient** sebanyak **89.613** sampel (**75,06%**) – pelanggan individu yang menginap untuk waktu singkat dan tidak dalam grup atau kontrak.
- Diikuti oleh **Transient-Party (21,04%)**, yaitu tamu perorangan yang bepergian dalam kelompok kecil.
- **Contract (3,41%)** dan **Group (0,48%)** menunjukkan porsi yang lebih kecil, menggambarkan segmen pelanggan dengan kontrak perusahaan atau perjalanan grup.

## Data Preparation
Menghapus Data duplikat 

![image](https://github.com/user-attachments/assets/c4375b23-c2a9-47ea-a913-73afd23ae119)

Berdasarkan output diatas data duplikat berhasil dihapus

**Menghapus beberapa kolom yang tidak relevan**

![image](https://github.com/user-attachments/assets/dccb4fbf-22fa-450c-9176-50fdc053357a)

Berikut adalah kolom-kolom yang dianggap tidak relevan untuk analisis fokus **waktu pemesanan dan durasi menginap terhadap harga**:

| Kolom                          | Alasan Penghapusan |
|-------------------------------|---------------------|
| reservation_status            | Status akhir reservasi, tidak relevan untuk fokus pada harga dan durasi. |
| reservation_status_date       | Tanggal status akhir, tidak relevan. |
| is_canceled                   | Sudah difilter hanya untuk data yang tidak dibatalkan. |
| adults, children, babies      | Tidak berpengaruh langsung terhadap waktu atau lama menginap. |
| country                       | Relevan hanya untuk analisis geografis. |
| agent, company                | Banyak missing values dan kurang relevan. |
| assigned_room_type            | Tidak berkaitan langsung dengan waktu atau harga. |
| reserved_room_type            | Sama seperti di atas. |
| distribution_channel          | Tidak berpengaruh langsung pada harga atau lama tinggal. |
| deposit_type                  | Tidak secara langsung memengaruhi harga. |
| days_in_waiting_list          | Lebih relevan untuk analisis permintaan, bukan harga. |
| previous_cancellations        | Relevan untuk prediksi pembatalan, tidak untuk harga. |
| previous_bookings_not_canceled| Sama seperti di atas. |
| booking_changes               | Fokus pada perubahan, bukan durasi atau harga. |
| is_repeated_guest             | Bisa relevan untuk loyalitas, tapi bukan fokus analisis ini. |
| required_car_parking_spaces   | Lebih ke permintaan khusus, bukan harga. |
| total_of_special_requests     | Sama seperti di atas. |

---
**Struktur Data Setelah Menghapus Kolom Tidak Relevan**

| No. | Kolom                      | Tipe Data | Jumlah Non-Null | Deskripsi (Opsional)                    |
|-----|----------------------------|-----------|------------------|-----------------------------------------|
| 1   | hotel                      | object    | 87396            | Nama hotel                              |
| 2   | lead_time                  | int64     | 87396            | Waktu tunggu sebelum check-in (hari)    |
| 3   | arrival_date_year          | int64     | 87396            | Tahun kedatangan                        |
| 4   | arrival_date_month         | object    | 87396            | Bulan kedatangan                        |
| 5   | arrival_date_week_number   | int64     | 87396            | Nomor minggu kedatangan                 |
| 6   | arrival_date_day_of_month  | int64     | 87396            | Tanggal kedatangan                      |
| 7   | stays_in_weekend_nights    | int64     | 87396            | Lama menginap saat akhir pekan (malam)  |
| 8   | stays_in_week_nights       | int64     | 87396            | Lama menginap saat hari kerja (malam)   |
| 9   | meal                       | object    | 87396            | Jenis makanan yang dipesan              |
| 10  | market_segment             | object    | 87396            | Segmen pasar dari tamu                  |
| 11  | customer_type              | object    | 87396            | Tipe pelanggan                          |
| 12  | adr                        | float64   | 87396            | Rata-rata harga per kamar per hari (ADR)|


**Total Data Setelah Pembersihan dan Penghapusan Kolom:**

**Ukuran DataFrame:** `(87396, 12)`

**Mengecek dan Menangani Outlier dengan IQR Method**

![image](https://github.com/user-attachments/assets/4997f947-8b72-48ce-91c2-fa771da2b6e3)
![image](https://github.com/user-attachments/assets/b992e1a4-014d-4bbb-8ef4-f8fbf42fdfe3)
![image](https://github.com/user-attachments/assets/31cf80b3-c165-4a1e-b09c-64cf351b1448)
![image](https://github.com/user-attachments/assets/e96926de-93f3-4eed-bda1-6d7b2963c7af)
![image](https://github.com/user-attachments/assets/ca3439f3-e6e0-48d9-add2-5cf193a62e5c)
![image](https://github.com/user-attachments/assets/c21a711f-d3a1-4e1a-a669-38bb55f846b0)
![image](https://github.com/user-attachments/assets/efd67007-d518-4b07-aabd-93baa08dd552)

Total data setelah penghapusan outlier:
| Jumlah Baris | Jumlah Kolom |
|--------------|--------------|
| 81186        | 12           |


### Pembuatan Fitur Tambahan dan Pemeriksaan Missing Values

Pada tahap ini, dilakukan pembuatan dua fitur baru, yaitu `total_stay` dan `total_cost`.  
- Fitur `total_stay` dihitung dari penjumlahan antara jumlah malam akhir pekan (`stays_in_weekend_nights`) dan jumlah malam hari kerja (`stays_in_week_nights`).  
- Fitur `total_cost` merupakan hasil perkalian antara `adr` (average daily rate) dengan `total_stay`.

![image](https://github.com/user-attachments/assets/b89a3803-ac73-4e63-bcee-5ba2960b2820)

Setelah kedua fitur tersebut dibuat, dilakukan pengecekan terhadap missing values untuk memastikan tidak ada data yang hilang pada kolom-kolom yang relevan, khususnya kolom yang baru ditambahkan. Hasil output menunjukkan bahwa tidak ditemukan nilai kosong (missing values) pada kolom `adr`, `lead_time`, `arrival_date_month`, `market_segment`, `total_stay`, maupun `total_cost`. Ini memastikan bahwa data siap untuk digunakan dalam analisis lanjutan atau pemodelan prediktif.

**Penanganan Outlier**

![image](https://github.com/user-attachments/assets/80f2421e-441d-4176-a671-0cf9e837179d)

Total data setelah penghapusan outlier : **59891**



**Encoding Fitur Kategori**
- Fitur customer_type diubah menjadi format numerik menggunakan LabelEncoder karena model machine learning tidak bisa memproses data kategorikal dalam bentuk string. Setiap jenis pelanggan dikonversi menjadi angka (misal: Contract → 0, Group → 1, dst.) agar bisa digunakan dalam pelatihan model.
  
![image](https://github.com/user-attachments/assets/52198076-706c-4017-98df-889f9481ca53)

- Fitur-fitur kategorikal non-ordinal seperti `hotel`, `meal`, `market_segment`, dan `arrival_date_month` diubah menggunakan *One-Hot Encoding* karena tidak memiliki urutan nilai yang bermakna. Dengan `pd.get_dummies()`, setiap kategori dibuat menjadi kolom biner (0 atau 1), sehingga informasi kategori tetap dipertahankan tanpa memberikan bobot urutan yang salah. Hasilnya disimpan dalam `df_final` dan siap digunakan untuk pelatihan model.

![image](https://github.com/user-attachments/assets/e1b7ee46-70ef-4dd1-947d-5f02addff8b5)

Fitur kategorikal perlu diubah karena model machine learning hanya dapat memproses data numerik. Jika tidak diubah, nilai kategori dalam bentuk teks tidak akan bisa dikenali atau digunakan dalam proses pelatihan model.

**Reduksi dimensi dengan PCA**

teknik PCA digunakan untuk mereduksi variabel asli menjadi sejumlah kecil variabel baru yang tidak berkorelasi linier, disebut komponen utama (PC). Komponen utama ini dapat menangkap sebagian besar varians dalam variabel asli. Sehingga, saat teknik PCA diterapkan pada data, ia hanya akan menggunakan komponen utama dan mengabaikan sisanya.

![image](https://github.com/user-attachments/assets/c2e3f322-a4de-493c-82cf-33521cf80f5e)


**Train-Test-Split**

Pembagian Data untuk Model Prediksi
Sebelum melakukan train-test-split, langkah awalnya adalah memisahkan antara fitur (predictor) dan label (target) yang ingin diprediksi.
- Variabel **X** digunakan untuk menyimpan fitur yang terdiri dari seluruh kolom pada dataset kecuali kolom target yaitu `adr` (Average Daily Rate), `total_cost` (total biaya menginap), serta kolom `customer_type` karena sudah diwakili oleh kolom `customer_type_encoded`.
- Variabel **y_adr** digunakan untuk menyimpan label pertama yaitu harga hotel per malam (`adr`).
- Variabel **y_cost** digunakan untuk menyimpan label kedua yaitu total biaya menginap (`total_cost`).
Selanjutnya, dilakukan pemisahan data menjadi data latih (train) dan data uji (test) menggunakan fungsi `train_test_split` dari sklearn dengan pembagian data sebesar 80:20 (80% untuk data latih dan 20% untuk data uji). Parameter `random_state=42` digunakan untuk memastikan bahwa pembagian data ini dapat direproduksi.
Untuk prediksi harga hotel (`adr`), dari total dataset, data terbagi menjadi 80% untuk data latih dan 20% untuk data uji dengan bentuk sebagai berikut:
- Data latih `X_train_adr` dan `y_train_adr` berisi fitur dan label untuk melatih model.
- Data uji `X_test_adr` dan `y_test_adr` berisi fitur dan label untuk menguji performa model.
Hal yang sama juga dilakukan untuk target kedua yaitu total biaya menginap (`total_cost`), dengan pembagian data latih dan uji yang serupa.

![image](https://github.com/user-attachments/assets/08728870-b876-49e8-aa8b-1c45421488ec)


**Standarisasi Data**

Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.
Selanjutnya dilakukan standarisasi dengan **MinMaxScaler** pada 9 fitur numerik, yaitu:

- `lead_time`
- `arrival_date_year`
- `arrival_date_week_number`
- `arrival_date_day_of_month`
- `customer_type_encoded`
- `stay_combined`
- `arrival_month_pca`
- `market_segment_pca`
- `meal_pca`

Standarisasi dilakukan secara **terpisah** untuk dua target prediksi:

1. **Prediksi Harga Hotel (ADR)**
    - Skala dihitung berdasarkan data latih (`X_train_adr`).
    - Transformasi kemudian diterapkan ke `X_train_adr` dan `X_test_adr`.

2. **Prediksi Total Biaya Menginap (Total Cost)**
    - Skala dihitung berdasarkan data latih (`X_train_cost`).
    - Transformasi kemudian diterapkan ke `X_train_cost` dan `X_test_cost`.

Proses **MinMaxScaler** menghasilkan distribusi data dengan rentang nilai antara **0 dan 1**, membuat algoritma machine learning lebih efisien dalam proses pelatihan.

![image](https://github.com/user-attachments/assets/1da04178-3329-4210-8f25-09a44ea6a842)


**Jadi, mengapa perlu dilakukan data preparation?**

- **Encoding** diperlukan untuk mengubah data kategori menjadi format numerik, karena sebagian besar algoritma machine learning tidak dapat bekerja langsung dengan data non-numerik.
  
- **Reduksi dimensi** (seperti PCA) berguna untuk menyederhanakan representasi data, mempercepat pelatihan model, serta mengurangi risiko overfitting dan multikolinearitas.

- **Split data** ke dalam set pelatihan dan pengujian memungkinkan evaluasi performa model terhadap data yang belum pernah dilihat sebelumnya, memberikan gambaran nyata tentang kemampuan generalisasi model.

- **Standarisasi** memastikan fitur berada pada skala yang sama, meningkatkan efisiensi dan efektivitas algoritma dalam menemukan pola dalam data.

## Modeling
**Model Multi-Output Regression dengan Random Forest**
Tahapan yang dilakukan:
| Parameter           | Nilai    | Fungsi                                                                                                                                                                                                               |
| ------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_estimators`      | `100`    | Jumlah pohon keputusan dalam hutan (*forest*). Semakin banyak pohon, prediksi cenderung lebih stabil, namun komputasi menjadi lebih lambat.                                                                          |
| `max_depth`         | `None`   | Tidak ada batasan kedalaman pohon. Pohon akan tumbuh hingga semua daun murni (*pure leaves*). Tanpa batasan ini bisa menyebabkan *overfitting*.                                                                      |
| `min_samples_split` | `2`      | Minimum jumlah sampel yang dibutuhkan untuk memisahkan sebuah *node*. Nilai yang lebih besar dapat mencegah pohon menjadi terlalu kompleks.                                                                          |
| `min_samples_leaf`  | `1`      | Minimum jumlah sampel yang harus ada di daun pohon. Nilai lebih besar membantu menciptakan pohon yang lebih seimbang dan mampu melakukan generalisasi dengan lebih baik.                                             |
| `max_features`      | `'sqrt'` | Jumlah fitur maksimum yang dipertimbangkan saat membuat split. `'sqrt'` artinya akar dari jumlah total fitur. Strategi ini membantu menghasilkan pohon yang lebih beragam, sehingga mengurangi korelasi antar pohon. |
| `random_state`      | `42`     | Seed untuk keacakan, digunakan agar model menghasilkan output yang sama setiap kali dijalankan (reproducibility).                                                                                                    |

**Kelebihan**
- Bisa memprediksi banyak target sekaligus (multi-output).
- Random Forest kuat terhadap overfitting dan noise.
- Tidak perlu normalisasi fitur.
- Proses pelatihan bisa paralel.

**Kekurangan**
- Tidak menangkap hubungan antar target.
- Boros memori dan waktu untuk data besar.
- Kurang interpretatif.
- Kurang optimal jika target sangat saling terkait.

**XGBoost MultiOutputRegressor**

Berikut adalah parameter default yang digunakan saat XGBRegressor(random_state=42):

| Parameter          | Nilai                          | Fungsi                                                                                                                                  |
| ------------------ | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `objective`        | `'reg:squarederror'` (default) | Fungsi loss untuk regresi, yaitu **mean squared error (MSE)**.                                                                          |
| `n_estimators`     | `100` (default)                | Jumlah pohon (trees) yang akan dibangun. <br>Lebih banyak pohon = prediksi lebih stabil, tapi komputasi meningkat.                      |
| `max_depth`        | `6` (default)                  | Kedalaman maksimum tiap pohon. <br>Mengontrol kompleksitas model. <br>Lebih besar = lebih kompleks, berisiko overfitting.               |
| `learning_rate`    | `0.3` (default)                | Menentukan seberapa besar setiap pohon berkontribusi terhadap prediksi akhir. <br>Nilai kecil = model belajar lebih lambat tapi stabil. |
| `subsample`        | `1.0` (default)                | Persentase sampel yang digunakan untuk melatih setiap pohon. <br>Nilai < 1.0 → membantu mengurangi overfitting.                         |
| `colsample_bytree` | `1.0` (default)                | Proporsi fitur yang dipakai dalam tiap pohon. <br>Mirip dengan `max_features` di Random Forest.                                         |
| `reg_alpha`        | `0` (default)                  | Regularisasi L1 (Lasso). Membantu mengurangi overfitting.                                                                               |
| `reg_lambda`       | `1` (default)                  | Regularisasi L2 (Ridge). Membantu menjaga bobot tetap kecil dan mencegah overfitting.                                                   |
| `random_state`     | `42`                           | Menjamin hasil yang **reproducible** (konsisten saat dijalankan ulang).                                                                 |
| `n_jobs`           | `None` (default)               | Menentukan jumlah core CPU yang digunakan. `None` artinya otomatis pakai semua.                                                         |
| `verbosity`        | `1` (default)                  | Kontrol log yang ditampilkan. 0 = silent, 1 = warning, 2 = info, 3 = debug.                                                             |


**Kelebihan**
- **Akurat dan Efisien**: XGBoost dikenal sebagai salah satu algoritma boosting paling akurat dan cepat.
- **Mampu Menangani Banyak Target**: MultiOutputRegressor memungkinkan prediksi multi-output seperti `adr` dan `total_cost`.
- **Tahan terhadap Overfitting**: Dilengkapi dengan regularisasi L1 dan L2.
- **Fleksibel**: Banyak parameter yang bisa diatur untuk meningkatkan performa.
- **Tidak Perlu Normalisasi**: Seperti Random Forest, XGBoost tidak membutuhkan skala fitur.

**Kekurangan**
- **Tidak Menangkap Hubungan Antar Target**: Setiap target diprediksi oleh model terpisah, tanpa mempertimbangkan korelasi antar target.
- **Konsumsi Memori dan Komputasi Tinggi**: Terutama saat menggunakan banyak estimator dan fitur.
- **Parameter Tuning Kompleks**: Memiliki banyak hyperparameter, butuh waktu untuk optimasi.
- **Kurang Interpretatif**: Sulit menjelaskan alasan di balik setiap prediksi.


## Evaluation
Metrik evaluasi yang digunakan dalam analisis ini adalah Mean Squared Error (MSE), yang berfungsi untuk mengukur seberapa besar kesalahan antara nilai aktual dan nilai yang diprediksi oleh model. MSE dihitung dengan cara mengkuadratkan selisih antara nilai aktual dan nilai prediksi, kemudian menghitung rata-ratanya.

Formula Mean Squared Error:
![image-21](https://github.com/user-attachments/assets/8dc69f20-510b-4be2-aeed-29d116943f7d)

Mean Squared Error (MSE) digunakan untuk menghitung rata-rata dari kuadrat selisih antara nilai prediksi model dengan nilai sebenarnya (ground truth).
Rumus MSE dapat dilihat pada gambar di atas, dengan penjelasan sebagai berikut:
- **N** menyatakan jumlah data dalam dataset,
- **yi** merupakan nilai aktual,
- **y_pred** adalah nilai hasil prediksi model.

**Evaluasi Model Machine Learning**
- Berikut adalah hasil yang diperoleh dari metrik ini, diurutkan dari kesalahan terkecil hingga terbesar:
  
| Model       | Train Error (MSE) | Test Error (MSE) |
|-------------|------------------:|-----------------:|
| RF          |        283.658147 |      1519.951410 |
| boosting_RF |       1227.285113 |      1513.729068 |

![image](https://github.com/user-attachments/assets/01789a4b-bc63-4e8f-b900-420c1f723fcf)

| y_true_adr | y_true_cost | prediksi_RF_adr | prediksi_RF_cost | prediksi_XGB_adr | prediksi_XGB_cost |
|------------|--------------|------------------|-------------------|-------------------|--------------------|
| 145.87     | 437.61       | 153.4            | 452.8             | 136.30            | 392.50             |
| 108.20     | 216.40       | 90.5             | 194.4             | 99.20             | 205.20             |
| 152.10     | 608.40       | 135.0            | 548.8             | 129.90            | 506.70             |
| 156.67     | 470.01       | 132.5            | 400.3             | 145.00            | 453.10             |
| 58.00      | 174.00       | 47.6             | 202.1             | 46.90             | 140.90             |


Dari tabel di atas, dapat dilihat bahwa setiap model memberikan hasil prediksi yang berbeda terhadap nilai aktual untuk `adr` (Average Daily Rate) dan `total_cost`. Secara umum, baik model **Random Forest** maupun **XGBoost** menunjukkan performa prediktif yang cukup baik dan stabil.

Sebagai contoh:
- Pada baris pertama, nilai aktual `adr` adalah **145.87**, sementara prediksi dari Random Forest adalah **153.4** dan XGBoost adalah **136.30**. Perbedaan ini menunjukkan deviasi yang masih dalam batas wajar.
- Untuk `total_cost` pada baris yang sama, nilai aktualnya adalah **437.61**, dengan prediksi dari Random Forest sebesar **452.8** dan dari XGBoost sebesar **392.50**.

Secara umum, dapat diamati bahwa:

- **Model Random Forest** cenderung melakukan sedikit *overestimasi* terhadap nilai aktual.
- **Model XGBoost** pada beberapa kasus tampak sedikit *underestimasi*.
- Namun, selisih keduanya tidak terlalu signifikan, yang menunjukkan bahwa hubungan antara fitur-fitur seperti waktu kedatangan, lama inap, dan tipe segmen pelanggan telah berhasil dipelajari oleh kedua model.

---

### Kesimpulan

Dari hasil analisis dan evaluasi model, dapat disimpulkan bahwa pendekatan ini berhasil menjawab dua rumusan masalah utama secara efektif:

1. **Kapan waktu terbaik untuk memesan hotel agar mendapatkan harga terbaik?**  
   Berdasarkan hasil PCA dan evaluasi model, ditemukan bahwa komponen waktu kedatangan (`arrival_date_month`) berpengaruh terhadap `adr`. Korelasi negatif menunjukkan bahwa harga cenderung lebih rendah pada bulan-bulan sepi seperti **Januari**, **Februari**, dan **November**, yang merupakan *low season*. Maka, waktu terbaik untuk memesan hotel adalah **di luar musim liburan** untuk mendapatkan tarif kamar yang lebih rendah.

2. **Berapa lama idealnya menginap untuk mengoptimalkan biaya?**  
   Fitur `total_cost`, yang merupakan hasil perkalian `adr` dengan durasi inap (`total_stay`), menunjukkan bahwa semakin lama durasi, maka total biaya akan meningkat secara linier. Namun, `adr` relatif stabil. Artinya, **tidak ada diskon signifikan untuk masa inap yang lebih lama**. Oleh karena itu, **durasi ideal menginap adalah sekitar 2 hingga 4 malam**, karena biaya tetap efisien sementara kenyamanan dan tujuan perjalanan masih dapat tercapai secara optimal.



## Referensi
Chen, C., & Kuo, H. (2015). A study of time series models for forecasting the room demand of Taiwan's hotel industry. Journal of Quality Assurance in Hospitality & Tourism, 16(3), 266–293. Link
RJPN

Li, H., Wang, Y., Law, R., & Guillet, B. D. (2016). Impact of the adoption of hotel mobile apps on customers' booking behavior: A case study of a major hotel chain. International Journal of Hospitality Management, 53, 42–49. Link
ScienceDirect
