# 🚢 Proyek Akhir Data Mining
## Preprocessing Data dan K-Means Clustering Menggunakan Dataset Titanic

---

## 👤 Identitas

**Nama :** Alif Nugraha  
**Mata Kuliah :** Data Mining  
**Topik :** Preprocessing Data dan Unsupervised Learning (K-Means Clustering)

---

# 📌 Deskripsi Proyek

Proyek ini merupakan implementasi proses **Preprocessing Data** dan **Unsupervised Learning** menggunakan algoritma **K-Means Clustering** pada dataset Titanic.

Tahapan yang dilakukan meliputi:

- Membaca dataset
- Data Cleaning
- Penanganan Missing Value
- Menghapus Data Duplikat
- Encoding Data Kategorikal
- Standardisasi Data
- Penentuan Jumlah Cluster menggunakan Metode Elbow
- Clustering menggunakan K-Means
- Evaluasi menggunakan Silhouette Score
- Visualisasi hasil clustering

---

# 📂 Dataset

Dataset yang digunakan adalah **Titanic Dataset**.

Dataset berisi informasi mengenai penumpang Titanic seperti:

- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

Lokasi dataset:

```text
Dataset/Titanic.csv
```

---

# 📁 Struktur Folder

```text
UAS-Alif-Nugraha
│
├── Dataset
│   └── Titanic.csv
│
├── gambar
│   ├── 01_missing_value.png
│   ├── 02_histogram_age.png
│   ├── 03_histogram_fare.png
│   ├── 04_boxplot_age.png
│   ├── 05_boxplot_fare.png
│   ├── 06_heatmap_korelasi.png
│   ├── 07_elbow_method.png
│   ├── 08_scatter_cluster.png
│   ├── 09_bar_cluster.png
│   ├── 10_pie_cluster.png
│   ├── 11_pairplot.png
│   ├── 12_rata_rata_cluster.png
│   ├── 13_boxplot_age_cluster.png
│   └── 14_boxplot_fare_cluster.png
│
├── Hasil
│   └── Titanic_Hasil_Clustering.csv
│
├── clustering.py
├── README.md
└── requirements.txt
```

---

# 🛠 Library yang Digunakan

Program menggunakan library berikut:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install seluruh library dengan:

```bash
pip install -r requirements.txt
```

---

# 🧹 Tahapan Preprocessing

Tahapan preprocessing yang dilakukan yaitu:

- Membaca dataset Titanic
- Mengecek Missing Value
- Menghapus Data Duplikat
- Mengisi Missing Value pada kolom Age menggunakan Median
- Mengisi Missing Value pada kolom Embarked menggunakan Modus
- Menghapus kolom Cabin
- Encoding kolom Sex
- One Hot Encoding kolom Embarked
- Standardisasi data menggunakan StandardScaler

---

# 🤖 Algoritma Clustering

Algoritma yang digunakan adalah:

## K-Means Clustering

Tahapan clustering:

1. Menentukan jumlah cluster menggunakan Metode Elbow
2. Melakukan clustering dengan K-Means
3. Menghitung Silhouette Score
4. Menampilkan hasil cluster

---

# 📊 Hasil Visualisasi

## 1. Visualisasi Missing Value

![Missing Value](gambar/01_missing_value.png)

---

## 2. Histogram Age

![Histogram Age](gambar/02_histogram_age.png)

---

## 3. Histogram Fare

![Histogram Fare](gambar/03_histogram_fare.png)

---

## 4. Boxplot Age

![Boxplot Age](gambar/04_boxplot_age.png)

---

## 5. Boxplot Fare

![Boxplot Fare](gambar/05_boxplot_fare.png)

---

## 6. Heatmap Korelasi

![Heatmap](gambar/06_heatmap_korelasi.png)

---

## 7. Metode Elbow

![Elbow](gambar/07_elbow_method.png)

---

## 8. Scatter Plot Hasil Clustering

![Scatter](gambar/08_scatter_cluster.png)

---

## 9. Bar Chart Jumlah Cluster

![Bar](gambar/09_bar_cluster.png)

---

## 10. Pie Chart Cluster

![Pie](gambar/10_pie_cluster.png)

---

## 11. Pairplot Cluster

![Pairplot](gambar/11_pairplot.png)

---

## 12. Rata-rata Age dan Fare Tiap Cluster

![Rata-rata](gambar/12_rata_rata_cluster.png)

---

## 13. Boxplot Age Berdasarkan Cluster

![Age Cluster](gambar/13_boxplot_age_cluster.png)

---

## 14. Boxplot Fare Berdasarkan Cluster

![Fare Cluster](gambar/14_boxplot_fare_cluster.png)

---

# 📈 Hasil Evaluasi

Evaluasi clustering dilakukan menggunakan **Silhouette Score**.

Hasil yang diperoleh:

```
Silhouette Score : 0.4229
```

Semakin tinggi nilai Silhouette Score maka semakin baik kualitas cluster yang dihasilkan.

---

# 📄 Output Program

Program menghasilkan:

- Dataset yang telah melalui proses preprocessing
- Hasil clustering menggunakan K-Means
- 14 visualisasi dalam folder **gambar**
- File hasil clustering

Lokasi output:

```text
Hasil/Titanic_Hasil_Clustering.csv
```

---

# ▶️ Cara Menjalankan Program

1. Clone repository

```bash
git clone https://github.com/username/UAS-Alif-Nugraha.git
```

2. Masuk ke folder project

```bash
cd UAS-Alif-Nugraha
```

3. Install library

```bash
pip install -r requirements.txt
```

4. Jalankan program

```bash
python clustering.py
```

---

# ✅ Kesimpulan

Berdasarkan hasil preprocessing dan penerapan algoritma K-Means Clustering, data penumpang Titanic berhasil dikelompokkan ke dalam beberapa cluster berdasarkan karakteristik seperti umur, kelas penumpang, tarif, dan jenis kelamin. Tahapan preprocessing, seperti penanganan missing value, encoding data kategorikal, serta standardisasi fitur, membantu meningkatkan kualitas proses clustering sehingga hasil yang diperoleh lebih mudah dianalisis dan divisualisasikan.

---

# 👨‍💻 Dibuat Oleh

**Alif Nugraha**

Program Studi Teknik Informatika

Universitas Pelita Bangsa

Mata Kuliah Data Mining