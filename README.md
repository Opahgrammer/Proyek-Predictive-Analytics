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
## Informasi Dataset

Dataset **Hotel Booking Demand** digunakan untuk menganalisis dan memprediksi pola pemesanan hotel. Dataset ini memuat informasi terkait pemesanan dari dua jenis hotel (Resort Hotel dan City Hotel), serta mencakup atribut seperti tanggal pemesanan, durasi menginap, jumlah tamu, tipe kamar, harga harian (ADR), total biaya, dan status pembatalan. Dataset ini sangat relevan untuk pengembangan model prediksi harga dan analisis perilaku pelanggan di industri perhotelan.

Dataset ini berasal dari situs **Kaggle**, dan telah banyak digunakan dalam penelitian serta kompetisi data science karena kelengkapan dan kompleksitas fitur-fiturnya.

### Tabel Informasi Dataset

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


## Eksplorasi Data Awal (EDA) - Struktur Dataset

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

### Catatan:
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

## EDA Missing Values & Outlier
Mengecek missing values, Duplikat dan data yang dianggap tidak memiliki kontribusi yang signifikan terhadap tujuan atau rumusan masalah
- Data duplikat
  
| Data Duplikat |
|---------------|
| 31994         |

## 📌 Pengecekan Missing Values

Berikut adalah jumlah nilai yang hilang (missing values) untuk setiap kolom:

| Kolom                            | Jumlah Missing |
|----------------------------------|----------------|
| hotel                            | 0              |
| is_canceled                      | 0              |
| lead_time                        | 0              |
| arrival_date_year                | 0              |
| arrival_date_month               | 0              |
| arrival_date_week_number         | 0              |
| arrival_date_day_of_month        | 0              |
| stays_in_weekend_nights          | 0              |
| stays_in_week_nights             | 0              |
| adults                           | 0              |
| children                         | 4              |
| babies                           | 0              |
| meal                             | 0              |
| country                          | 452            |
| market_segment                   | 0              |
| distribution_channel             | 0              |
| is_repeated_guest                | 0              |
| previous_cancellations           | 0              |
| previous_bookings_not_canceled   | 0              |
| reserved_room_type               | 0              |
| assigned_room_type               | 0              |
| booking_changes                  | 0              |
| deposit_type                     | 0              |
| agent                            | 12,193         |
| company                          | 82,137         |
| days_in_waiting_list             | 0              |
| customer_type                    | 0              |
| adr                              | 0              |
| required_car_parking_spaces      | 0              |
| total_of_special_requests        | 0              |
| reservation_status               | 0              |
| reservation_status_date          | 0              |

---

## Data Setelah Penghapusan Kolom Tidak Relevan

**Bentuk DataFrame:**

- Jumlah Baris: `87,396`
- Jumlah Kolom: `12`

**Informasi Kolom yang Tersisa:**

| No | Kolom                       | Tipe Data |
|----|-----------------------------|-----------|
| 1  | hotel                       | object    |
| 2  | lead_time                   | int64     |
| 3  | arrival_date_year           | int64     |
| 4  | arrival_date_month          | object    |
| 5  | arrival_date_week_number    | int64     |
| 6  | arrival_date_day_of_month   | int64     |
| 7  | stays_in_weekend_nights     | int64     |
| 8  | stays_in_week_nights        | int64     |
| 9  | meal                        | object    |
|10  | market_segment              | object    |
|11  | customer_type               | object    |
|12  | adr                         | float64   |

---

## Kolom yang Dihapus dan Alasannya

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

** Total Data Setelah Pembersihan:**
- **Ukuran DataFrame:** `(87396, 12)`

## Mengecek dan Menangani Outlier dengan IQR Method

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

## Exploratory Data Analysis - Univariate Analysis
- Analisis univariat adalah jenis analisis statistik yang mengamati dan menjelaskan satu variabel pada suatu waktu. Tujuannya adalah untuk mendeskripsikan data tersebut, meringkasnya, dan mencari pola-pola yang ada dalam satu variabel tersebut.
### Fitur kategori
![image](https://github.com/user-attachments/assets/69ae2924-d2d6-48c3-933b-4a0e9d2ffba5)


##### hotel                                      
- City Hotel            53428           61.13
- Resort Hotel          33968           38.87

![image](https://github.com/user-attachments/assets/0b0bece8-8d35-4825-9bb4-f7b3b2b0e4da)


##### arrival_date_month                               
- August                      11257           12.88
- July                        10057           11.51
- May                          8355            9.56
- April                        7908            9.05
- June                         7765            8.88
- March                        7513            8.60
- October                      6934            7.93
- September                    6690            7.65
- February                     6098            6.98
- December                     5131            5.87
- November                     4995            5.72
- January                      4693            5.37


![image](https://github.com/user-attachments/assets/9f622ffc-f1ef-4f24-a9a5-fb531b4d2d93)


##### meal                                    
- BB                 67978           77.78
- SC                  9481           10.85
- HB                  9085           10.40
- Undefined            492            0.56
- FB                   360            0.41

![image](https://github.com/user-attachments/assets/373ee402-bf1d-420c-8d40-a68520271a37)

##### market_segment                               
- Online TA               51618           59.06
- Offline TA/TO           13889           15.89
- Direct                  11804           13.51
- Groups                   4942            5.65
- Corporate                4212            4.82
- Complementary             702            0.80
- Aviation                  227            0.26
- Undefined                   2            0.00

![image](https://github.com/user-attachments/assets/83ae2507-de4f-424a-a7d3-10b8eeeda664)

##### customer_type                                 
- Transient                71986           82.37
- Transient-Party          11727           13.42
- Contract                  3139            3.59
- Group                      544            0.62

### Fitur Numerik
![image](https://github.com/user-attachments/assets/6ede20c0-74c1-4ed2-95a6-a2d3cd533c60)
Dari beberapa visualisasi diatas dapat disimpulkan sebagai berikut:

**Lead Time**
- Histogram `lead_time` menunjukkan mayoritas pemesanan dilakukan dalam waktu kurang dari 50 hari sebelum kedatangan.
- Distribusinya **right-skewed**, menandakan ada sebagian kecil tamu yang memesan jauh-jauh hari.

**Tahun Kedatangan**
- Fitur `arrival_date_year` menunjukkan jumlah tertinggi pada tahun **2016**, menandakan data tahun tersebut paling dominan dalam dataset.

**Distribusi Kedatangan**
- Fitur `arrival_date_week_number` dan `arrival_date_day_of_month` menunjukkan variasi kedatangan tamu yang cukup **merata** sepanjang minggu dan bulan.
- Namun, minggu ke-**28** menunjukkan **lonjakan signifikan**—kemungkinan karena libur musim panas atau event khusus.

**Durasi Menginap**
- Fitur `stays_in_week_nights` dan `stays_in_weekend_nights` didominasi oleh nilai **1 dan 2 malam**.
- Menandakan sebagian besar tamu **tidak tinggal terlalu lama**.

**Distribusi Harga per Malam (`adr`)**
- Distribusi `adr` **cukup normal** dengan rata-rata sekitar **100**, namun memiliki **ekor kanan (right-skewed)**.
- Beberapa nilai `adr` sangat tinggi (di atas **200**), kemungkinan merupakan **outlier** atau pemesanan dalam kondisi spesial seperti **event atau high season**.

## Exploratory Data Analysis - Multivariate Analysis
### Fitur Numerik
![image](https://github.com/user-attachments/assets/ff75dfdd-5100-4aa6-bc16-2e99cca94970)
Berdasarkan hasil visualisasi matriks korelasi yang ditampilkan, diperoleh beberapa hubungan penting antar fitur numerik sebagai berikut:

- Korelasi Tinggi antar Fitur Terkait Durasi Inap

  stays_in_week_nights dan total_stay memiliki korelasi sangat kuat (0.92), menunjukkan bahwa jumlah malam di hari kerja adalah komponen utama dari total durasi inap.

  stays_in_weekend_nights dan total_stay juga menunjukkan korelasi yang tinggi sebesar 0.71, yang berarti malam akhir pekan juga memberikan kontribusi signifikan terhadap durasi total menginap.

- Korelasi Lead Time

  lead_time menunjukkan korelasi sedang dengan total_stay (0.38) dan stays_in_week_nights (0.36). Ini mengindikasikan bahwa pemesanan yang dilakukan jauh hari sebelumnya cenderung dikaitkan dengan masa inap yang lebih lama, terutama pada hari kerja.

  Korelasi lead_time terhadap adr (harga per malam) rendah (0.08), yang menunjukkan bahwa waktu pemesanan tidak banyak memengaruhi harga per malam secara langsung.

- Fitur adr (Average Daily Rate)

  adr memiliki korelasi sangat rendah dengan semua fitur lain, termasuk total_stay (0.10) dan lead_time (0.08). Ini menandakan bahwa rata-rata harga per malam kemungkinan besar lebih dipengaruhi oleh variabel non-numerik seperti tipe kamar, jenis pelanggan, musim, atau status reservasi.

- Tanggal Kedatangan

  Fitur-fitur yang berkaitan dengan tanggal kedatangan seperti arrival_date_year, arrival_date_week_number, dan arrival_date_day_of_month memiliki korelasi rendah terhadap fitur lainnya, termasuk adr dan total_stay. Hal ini mengindikasikan bahwa tidak terdapat pola linier kuat antara tanggal kedatangan dengan fitur numerik lainnya.

- Korelasi Negatif

  Satu-satunya korelasi negatif yang mencolok adalah antara arrival_date_year dan arrival_date_week_number (-0.53). Ini dapat mencerminkan distribusi data berdasarkan waktu, misalnya lebih banyak pemesanan di minggu-minggu tertentu pada tahun-tahun tertentu.

![image](https://github.com/user-attachments/assets/22abc74f-6876-45ed-a9be-3abc7bc1d546)
- Membuat kolom baru dan mengecek apakah ada missing values, dari hasil output diatas tidak ada missing values pada kolom yang baru dibuat yaitu total_stay, total_cost

![image](https://github.com/user-attachments/assets/d97b242c-a85f-4cc9-b543-82163031cb3f)
- Mengecek dan menghapus outlier pada fitur yang baru dibuat dengan total data setelah penghapusan outlier **59891**

![image](https://github.com/user-attachments/assets/9ea9e3ec-6295-4aae-948e-287b87abcf2e)
Dari hasil visualisasi tersebut dapat disimpulkan:

- Harga rata-rata harian (ADR) bervariasi setiap bulan dan berbeda antar segmen pasar.

- Segmen Direct dan Online TA cenderung memiliki ADR lebih tinggi dibanding segmen lain hampir di seluruh bulan.

- ADR tertinggi terjadi di bulan Juli, Agustus, September, menunjukkan musim puncak atau permintaan tinggi.

- Bulan Januari dan Februari memiliki ADR paling rendah, menandakan musim sepi atau penurunan permintaan.

- Segmen Complementary memiliki ADR paling rendah secara konsisten.

- Pola musiman ini penting untuk strategi penetapan harga dan promosi sesuai segmen pasar dan waktu kedatangan tamu.

![image](https://github.com/user-attachments/assets/7f06f039-c23b-45ac-847e-b5c84b7d02dc)
Dari visualisasi tersebut dapat dijelaskan:

- Plot menunjukkan hubungan antara Lead Time (jumlah hari sebelum kedatangan) dengan Harga Harian Rata-rata (ADR).

- Sebagian besar reservasi dilakukan dengan Lead Time rendah (konsentrasi data di dekat 0 hari).

- Garis tren (merah) memperlihatkan pola naik dulu sedikit lalu menurun setelah Lead Time sekitar 100-150 hari.

- Artinya, harga ADR cenderung sedikit meningkat jika reservasi dilakukan dengan Lead Time sedang, tetapi menurun untuk Lead Time yang sangat lama.

- Pola ini bisa berarti bahwa booking dengan waktu terlalu lama sebelum kedatangan cenderung mendapatkan harga lebih rendah, mungkin karena promosi awal atau strategi diskon.

- Namun, hubungan ini tidak terlalu kuat, terlihat dari penyebaran titik data yang cukup lebar.

Singkatnya, ada kecenderungan ADR naik dulu lalu turun seiring bertambahnya Lead Time, tapi dengan variasi harga yang besar di setiap rentang waktu.

![image](https://github.com/user-attachments/assets/1215bab4-a9a2-4511-a5f0-5910da522323)
Dari visualisasi tersebut dapat dijelaskan:

- Titik-titik biru mewakili data individual dari setiap reservasi.

- Garis merah menunjukkan rata-rata total biaya untuk setiap lama menginap.

- Terlihat tren bahwa semakin lama menginap, total biaya rata-rata semakin meningkat, yang masuk akal karena biaya dihitung berdasarkan harga per malam dikalikan jumlah malam menginap.

- Namun, kenaikan total biaya mulai melambat setelah lama menginap 4-6 malam, terlihat dari kemiringan garis rata-rata yang mulai mendatar.

- Hal ini bisa mengindikasikan adanya diskon atau harga yang lebih rendah per malam jika menginap lebih lama (efek tarif diskon untuk masa inap panjang).

- Ada juga variasi besar pada total biaya dalam tiap kategori lama menginap, menunjukkan harga per malam yang bervariasi antar reservasi.

Singkatnya, total biaya meningkat seiring lama menginap, namun kenaikan biaya tidak selalu linear, bisa ada penyesuaian harga untuk lama menginap yang lebih panjang.

### Fitur Kategorik
![image](https://github.com/user-attachments/assets/94719373-3677-445a-a68d-c3983f411b78)
Visualisasi tersebut menunjukkan rata-rata harga harian (ADR) hotel per bulan berdasarkan jenis hotel (City Hotel vs Resort Hotel).

- City Hotel cenderung memiliki ADR yang stabil dan lebih tinggi dari Resort Hotel pada sebagian besar bulan, terutama di bulan-bulan non-musim liburan.

- Resort Hotel menunjukkan kenaikan tajam ADR di bulan Juni–Agustus, dengan puncaknya di Agustus (mencapai lebih dari 140), yang kemungkinan besar disebabkan oleh musim liburan atau peak season.

- Setelah Agustus, ADR Resort Hotel turun drastis, kembali lebih rendah dari City Hotel.

- Hal ini menunjukkan bahwa City Hotel lebih stabil sepanjang tahun, sementara Resort Hotel sangat dipengaruhi oleh musim liburan.

Kesimpulan:

- Untuk mencari harga terbaik di Resort Hotel, hindari musim liburan (Juli–Agustus).

- Untuk City Hotel, fluktuasi harga tidak terlalu besar, lebih cocok untuk perjalanan bisnis yang fleksibel.

## Data Preparation
Teknik Data preparation yang dilakukan terdiri dari:
- Label Encoding
- One-Hot Encoding
- Reduksi dimensi dengan PCA
- Pembagian dataset dengan fungsi train_test_split dari library sklearn.

### Encoding Fitur Kategori
![image](https://github.com/user-attachments/assets/99b07a0f-576c-4e47-81fc-11d141bd60f3)
- Dari hasil output diatas Fitur Customer_type diubah menggunakan LabelEncoder
![image](https://github.com/user-attachments/assets/b758f077-01cd-4c29-a8b8-5a1c6310dba3)
- Fitur hotel, meal, market segment, arrival_date_month diubah menggunakan One-Hot Encoder
### Reduksi dimensi dengan PCA
![image](https://github.com/user-attachments/assets/0663ba06-9852-4336-9b63-dc43cfe6f79f)
### Standarisasi
![image](https://github.com/user-attachments/assets/1d2e14af-ec43-434a-b77d-868da459335c)
#### Fitur Numerik yang Dinormalisasi
Fitur-fitur yang dinormalisasi meliputi:
- lead_time (waktu antara reservasi dan tanggal kedatangan),
- arrival_date_year (tahun kedatangan),
- arrival_date_week_number (minggu ke berapa dalam tahun),
- arrival_date_day_of_month (hari kedatangan dalam bulan),
- customer_type_encoded (tipe pelanggan yang sudah diencoding secara numerik),
- stay_combined (fitur gabungan durasi inap, hasil rekayasa fitur),
- arrival_month_pca (hasil PCA dari fitur bulan kedatangan),
- market_segment_pca (hasil PCA dari segmen pasar),
- meal_pca (hasil PCA dari jenis meal plan).
- 
## Modeling
### XGBoost MultiOutputRegressor
Untuk memprediksi dua target sekaligus, yaitu adr (Average Daily Rate) dan total_cost, digunakan MultiOutputRegressor yang membungkus model dasar XGBRegressor. Pendekatan ini memungkinkan model melakukan regresi multi-target secara bersamaan.

- Data fitur (X) dan target (adr dan total_cost) dipisah menjadi data latih dan uji menggunakan train_test_split.
- Model kemudian dilatih (fit) pada data latih dengan dua target output secara paralel.
- Model ini memanfaatkan kekuatan XGBoost yang efisien dan akurat untuk regresi pada masing-masing target secara independen dalam satu pipeline.

### Model Multi-Output Regression dengan Random Forest
Tahapan yang dilakukan:
- Definisi Model Dasar
Model regresi berbasis Random Forest dibuat dengan RandomForestRegressor(random_state=42) untuk menjaga hasil yang reproducible.
- Pembungkusan untuk Multi-Output
Karena target yang ingin diprediksi ada dua (adr dan total_cost), model dasar dibungkus dengan MultiOutputRegressor agar bisa menangani regresi multi-output secara paralel.
- Persiapan Data
Data fitur (X) dan target multi-output (y_multi dengan kolom adr dan total_cost) dipisah menjadi data latih dan data uji menggunakan train_test_split dengan porsi 80% latih dan 20% uji.
- Pelatihan Model
Model multi-output dilatih dengan data latih menggunakan fit(), sehingga model belajar memprediksi kedua target secara bersamaan.



## Evaluation
### Evaluasi Model dengan Mean Squared Error (MSE)
Setelah model Random Forest Multi-Output Regressor selesai dilatih, dilakukan evaluasi performa model pada data uji (X_test_multi) dengan langkah-langkah berikut:
- Prediksi pada Data Uji
Model digunakan untuk memprediksi nilai target adr dan total_cost pada data uji dengan predict().
- Perhitungan Mean Squared Error (MSE)
MSE dihitung secara terpisah untuk masing-masing target:
mse_adr_rf: MSE untuk prediksi adr.
mse_cost_rf: MSE untuk prediksi total_cost.
- MSE mengukur rata-rata kuadrat selisih antara nilai sebenarnya dan prediksi model. Nilai MSE yang lebih kecil menunjukkan model dengan performa yang lebih baik.

### Evaluasi Model XGBoost Multi-Output Regression
Setelah model XGBoost Multi-Output Regressor dilatih, evaluasi performa dilakukan dengan langkah-langkah berikut:

- Prediksi pada Data Uji
Model digunakan untuk memprediksi kedua target, adr dan total_cost, pada data uji (X_test_multi) menggunakan method predict().
- Perhitungan Mean Squared Error (MSE)
MSE dihitung secara terpisah untuk masing-masing target:
mse_adr_xgb untuk prediksi adr.
mse_cost_xgb untuk prediksi total_cost.
MSE mengukur rata-rata kuadrat kesalahan antara nilai asli dan hasil prediksi model. Nilai MSE yang lebih rendah menunjukkan performa model yang lebih baik.
- Pelaporan Hasil Evaluasi
Nilai MSE untuk masing-masing target ditampilkan sebagai metrik evaluasi model.

![image](https://github.com/user-attachments/assets/ee7bb8a5-c6a8-4470-8279-313dd877d569)
Grafik di atas menunjukkan perbandingan nilai Mean Squared Error (MSE) antara dua model machine learning, yaitu **Random Forest** dan **XGBoost**, terhadap dua target prediksi: **ADR (Average Daily Rate)** dan **Total Cost**.

- Baik model Random Forest maupun XGBoost menghasilkan nilai MSE yang relatif rendah untuk prediksi ADR, yaitu sekitar 330.
- Namun, untuk prediksi Total Cost, kedua model menunjukkan nilai MSE yang jauh lebih tinggi, yaitu sekitar 2700.
- Hal ini menunjukkan bahwa kedua model lebih akurat dalam memprediksi ADR dibandingkan Total Cost.

Secara umum, performa kedua model cukup sebanding, dengan XGBoost sedikit lebih baik dalam prediksi Total Cost berdasarkan nilai MSE yang sedikit lebih rendah.

# Perbandingan Model Multi-Output Regression

## 1. Random Forest Multi-Output Regression

**Kelebihan:**

- Robust terhadap overfitting: Dengan banyak pohon keputusan, model ini cenderung lebih stabil dan tahan terhadap overfitting.
- Mudah diinterpretasi: Fitur penting (feature importance) dapat diekstrak untuk analisis.
- Tidak membutuhkan scaling fitur: Random Forest bekerja baik dengan fitur dalam skala asli tanpa perlu normalisasi.
- Cepat dalam pelatihan untuk dataset berukuran sedang.

**Kekurangan:**

- Kurang efisien pada dataset sangat besar: Bisa menjadi lambat dan boros memori saat jumlah data dan fitur sangat besar.
- Tidak optimal untuk fitur numerik yang sangat berkorelasi: Bisa mempengaruhi performa jika banyak fitur saling bergantung.
- Performa bisa lebih rendah dibanding model boosting seperti XGBoost pada beberapa kasus.

---

## 2. XGBoost Multi-Output Regression

**Kelebihan:**

- Performa tinggi dan presisi: Algoritma boosting yang secara iteratif memperbaiki kesalahan prediksi sehingga sering menghasilkan akurasi lebih baik.
- Fleksibel dan dapat disesuaikan: Banyak parameter tuning untuk optimasi performa model.
- Dukungan built-in regularisasi: Membantu mencegah overfitting lebih efektif.
- Mampu menangani data dengan fitur numerik dan kategorikal secara efisien.

**Kekurangan:**

- Proses pelatihan lebih lambat dibanding Random Forest: Karena model dibangun secara berurutan (boosting).
- Lebih kompleks dalam tuning hyperparameter: Membutuhkan eksperimen dan pemahaman lebih dalam.
- Membutuhkan normalisasi atau preprocessing fitur yang lebih hati-hati agar performa optimal.


## Referensi
Chen, C., & Kuo, H. (2015). A study of time series models for forecasting the room demand of Taiwan's hotel industry. Journal of Quality Assurance in Hospitality & Tourism, 16(3), 266–293. Link
RJPN

Li, H., Wang, Y., Law, R., & Guillet, B. D. (2016). Impact of the adoption of hotel mobile apps on customers' booking behavior: A case study of a major hotel chain. International Journal of Hospitality Management, 53, 42–49. Link
ScienceDirect
