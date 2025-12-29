# POS Tagging IndoBERT

**Kelompok 5:**  
- Ni Komang Marsyani (2205551052) 
- Ni Kadek Dwita Putri Suastini (2205551074)  
- Ni Putu Putri Maheswari Paramhansa (2205551101)
  
Repositori ini berisi implementasi POS Tagging menggunakan model IndoBERT dengan dataset yang dikumpulkan dari Kompas.com.

## Dataset

Dataset yang digunakan sudah melalui proses preprocessing dan terbagi menjadi data training, validasi, dan testing. Dataset ini berasal dari IndoNLU dan dapat ditemukan pada folder: `/datasets`


## Struktur Kode

- **Scraping dan Preprocessing**  
  Terdapat 3 skrip kode scraping dan preprocessing yang berbeda, masing-masing mengambil data dan melakukan preprocessing dari kategori berita yang berbeda di Kompas.com.

- **Fine-tuning**  
  Kode fine-tuning model tersedia di file: `FineTuning-API.ipynb`
  
  Pada notebook ini terdapat cell yang juga mengimplementasikan API untuk model POS Tagging.

- **API Implementation**  
File `api.py` berisi implementasi API yang dibuat dari cell pada notebook `FineTuning-API.ipynb`.

## Model

Hasil model IndoBERT yang sudah di-fine-tune dapat diakses melalui tautan berikut:

[Link Model](https://drive.google.com/drive/folders/1Y7fZMOQKaMc4IaOb8FY5pn_xjPnDbd-F)
