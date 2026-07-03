# ==========================================================
# PROYEK AKHIR DATA MINING
# PREPROCESSING & K-MEANS CLUSTERING
# NAMA : ALIF NUGRAHA
# ==========================================================

# =======================
# IMPORT LIBRARY
# =======================

import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# =======================
# MEMBUAT FOLDER GAMBAR
# =======================

os.makedirs("gambar", exist_ok=True)

# =======================
# PENGATURAN TAMPILAN
# =======================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# =======================
# MEMBACA DATASET
# =======================

print("="*60)
print("MEMBACA DATASET")
print("="*60)

df = pd.read_csv("Titanic.csv")

print(df.head())

print("\nUkuran Dataset :")
print(df.shape)

print("\nNama Kolom :")
print(df.columns.tolist())

print("\nInformasi Dataset")
print(df.info())

print("\nStatistik Dataset")
print(df.describe())

# =======================
# MISSING VALUE
# =======================

print("\nJumlah Missing Value")

print(df.isnull().sum())

plt.figure(figsize=(12,5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title("Visualisasi Missing Value")

plt.savefig(
    "gambar/01_missing_value.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# =======================
# DUPLIKAT
# =======================

print("\nJumlah Data Duplikat :")

print(df.duplicated().sum())

# =======================
# CLEANING DATA
# =======================

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

if "Cabin" in df.columns:
    df.drop(
        columns=["Cabin"],
        inplace=True
    )

df.drop_duplicates(inplace=True)

print("\nMissing Value Setelah Cleaning")

print(df.isnull().sum())

# =======================
# ENCODING
# =======================

df["Sex"] = df["Sex"].map({
    "male":0,
    "female":1
})

df = pd.get_dummies(
    df,
    columns=["Embarked"],
    drop_first=True
)

print("\nCleaning Data Berhasil")

# =======================
# HISTOGRAM AGE
# =======================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Age"],
    bins=20,
    kde=True,
    color="royalblue"
)

plt.title("Distribusi Umur")

plt.xlabel("Age")

plt.ylabel("Frekuensi")

plt.grid(True)

plt.savefig(
    "gambar/02_histogram_age.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# =======================
# HISTOGRAM FARE
# =======================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Fare"],
    bins=20,
    kde=True,
    color="orange"
)

plt.title("Distribusi Fare")

plt.xlabel("Fare")

plt.ylabel("Frekuensi")

plt.grid(True)

plt.savefig(
    "gambar/03_histogram_fare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# =======================
# BOXPLOT AGE
# =======================

plt.figure(figsize=(8,4))

sns.boxplot(
    x=df["Age"],
    color="skyblue"
)

plt.title("Boxplot Age")

plt.savefig(
    "gambar/04_boxplot_age.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# =======================
# BOXPLOT FARE
# =======================

plt.figure(figsize=(8,4))

sns.boxplot(
    x=df["Fare"],
    color="salmon"
)

plt.title("Boxplot Fare")

plt.savefig(
    "gambar/05_boxplot_fare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# =======================
# HEATMAP KORELASI
# =======================

plt.figure(figsize=(8,6))

sns.heatmap(

    df[
        [
            "Age",
            "Fare",
            "Pclass",
            "SibSp",
            "Parch"
        ]
    ].corr(),

    annot=True,
    cmap="coolwarm"

)

plt.title("Heatmap Korelasi")

plt.savefig(
    "gambar/06_heatmap_korelasi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("\nVisualisasi Awal Selesai")
# ==========================================================
# MENENTUKAN FITUR
# ==========================================================

fitur = [
    "Pclass",
    "Age",
    "Fare",
    "Sex"
]

X = df[fitur]

# ==========================================================
# STANDARDISASI DATA
# ==========================================================

print("\n" + "="*60)
print("STANDARDISASI DATA")
print("="*60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Standardisasi berhasil.")

# ==========================================================
# METODE ELBOW
# ==========================================================

print("\n" + "="*60)
print("METODE ELBOW")
print("="*60)

wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o",
    linewidth=2,
    color="blue"
)

plt.title("Metode Elbow")

plt.xlabel("Jumlah Cluster")

plt.ylabel("WCSS")

plt.grid(True)

plt.savefig(
    "gambar/07_elbow_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# ==========================================================
# K-MEANS
# ==========================================================

print("\n" + "="*60)
print("PROSES K-MEANS")
print("="*60)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Clustering berhasil.")

# ==========================================================
# SILHOUETTE SCORE
# ==========================================================

score = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("\nSilhouette Score :")
print(round(score,4))

# ==========================================================
# SCATTER PLOT
# ==========================================================

plt.figure(figsize=(9,6))

plt.scatter(

    df["Age"],
    df["Fare"],

    c=df["Cluster"],

    cmap="viridis",

    s=70

)

centroid = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(

    centroid[:,1],

    centroid[:,2],

    marker="X",

    s=250,

    color="red",

    label="Centroid"

)

plt.xlabel("Age")

plt.ylabel("Fare")

plt.title("K-Means Clustering")

plt.legend()

plt.grid(True)

plt.savefig(
    "gambar/08_scatter_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# ==========================================================
# BAR CHART
# ==========================================================

plt.figure(figsize=(7,5))

df["Cluster"].value_counts().sort_index().plot(

    kind="bar",

    color=["royalblue","orange","green"]

)

plt.title("Jumlah Data Tiap Cluster")

plt.xlabel("Cluster")

plt.ylabel("Jumlah Data")

plt.grid(axis="y")

plt.savefig(
    "gambar/09_bar_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# ==========================================================
# PIE CHART
# ==========================================================

plt.figure(figsize=(6,6))

df["Cluster"].value_counts().plot(

    kind="pie",

    autopct="%1.1f%%",

    startangle=90,

    colors=["royalblue","orange","green"]

)

plt.ylabel("")

plt.title("Persentase Cluster")

plt.savefig(
    "gambar/10_pie_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# ==========================================================
# PAIRPLOT
# ==========================================================

pair = sns.pairplot(

    df,

    vars=[

        "Age",

        "Fare",

        "Pclass"

    ],

    hue="Cluster",

    palette="viridis"

)

pair.savefig(

    "gambar/11_pairplot.png",

    dpi=300

)

plt.show()

plt.close()

print("\nVisualisasi Clustering Selesai.")
# ==========================================================
# RINGKASAN CLUSTER
# ==========================================================

print("\n" + "="*60)
print("RINGKASAN SETIAP CLUSTER")
print("="*60)

ringkasan = df.groupby("Cluster").agg({

    "Age":["mean","min","max"],

    "Fare":["mean","min","max"],

    "Pclass":["mean"],

    "PassengerId":["count"]

})

print(ringkasan)

# ==========================================================
# RATA-RATA SETIAP CLUSTER
# ==========================================================

cluster_mean = df.groupby("Cluster")[

    ["Age","Fare"]

].mean()

plt.figure(figsize=(8,5))

cluster_mean.plot(

    kind="bar",

    figsize=(8,5),

    colormap="viridis"

)

plt.title("Rata-rata Age dan Fare Tiap Cluster")

plt.xlabel("Cluster")

plt.ylabel("Nilai")

plt.xticks(rotation=0)

plt.grid(axis="y")

plt.savefig(

    "gambar/12_rata_rata_cluster.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

plt.close()

# ==========================================================
# BOXPLOT AGE BERDASARKAN CLUSTER
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(

    data=df,

    x="Cluster",

    y="Age",

    palette="viridis"

)

plt.title("Perbandingan Age Berdasarkan Cluster")

plt.savefig(

    "gambar/13_boxplot_age_cluster.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

plt.close()

# ==========================================================
# BOXPLOT FARE BERDASARKAN CLUSTER
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(

    data=df,

    x="Cluster",

    y="Fare",

    palette="viridis"

)

plt.title("Perbandingan Fare Berdasarkan Cluster")

plt.savefig(

    "gambar/14_boxplot_fare_cluster.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

plt.close()

# ==========================================================
# JUMLAH DATA SETIAP CLUSTER
# ==========================================================

print("\nJumlah Anggota Tiap Cluster")

print(df["Cluster"].value_counts())

# ==========================================================
# MENAMPILKAN 10 DATA HASIL CLUSTERING
# ==========================================================

print("\n" + "="*60)

print("10 DATA HASIL CLUSTERING")

print("="*60)

print(df.head(10).to_string())

# ==========================================================
# MENYIMPAN HASIL
# ==========================================================

output = "Titanic_Hasil_Clustering.csv"

df.to_csv(

    output,

    index=False

)

print("\nFile berhasil disimpan menjadi")

print(output)

# ==========================================================
# MENAMPILKAN SILHOUETTE SCORE
# ==========================================================

print("\n" + "="*60)

print("HASIL EVALUASI")

print("="*60)

print("Silhouette Score :", round(score,4))

# ==========================================================
# INFORMASI AKHIR
# ==========================================================gith

print("\n" + "="*60)

print("INFORMASI DATASET")

print("="*60)

print("Jumlah Data :", len(df))

print("Jumlah Cluster :", df["Cluster"].nunique())

print("\nProgram selesai dijalankan.")

print("Seluruh gambar tersimpan pada folder : gambar")