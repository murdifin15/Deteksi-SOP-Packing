# Product Requirements Document (PRD)
## Sistem Deteksi Kepatuhan SOP Packing Gudang (YOLO11 Real-Time)

---

## 1. Ringkasan Eksekutif (Executive Summary)
Sistem ini adalah program berbasis Python yang memanfaatkan **YOLO11** (Vision AI) dan **OpenCV** untuk menganalisis proses packing barang di gudang secara **Real-Time via Live Webcam** maupun melalui video rekaman. Fokus utama sistem adalah memverifikasi **kepatuhan sekuensial (urutan langkah)** dari **SOP 1: Packing Barang Pecah Belah / Fragile** secara otomatis, menampilkan jendela kamera live dengan penanda visual (HUD) dan mencetak laporan audit di terminal.

---

## 2. Definisi Masalah & Tujuan (Problem Statement & Goals)
- **Masalah:** Inspeksi manual terhadap proses packing di gudang memerlukan waktu lama, rentan kesalahan manusia, dan sulit dipantau secara konsisten.
- **Tujuan:** Menyediakan sistem deteksi otomatis *real-time* langsung dari kamera meja packing:
  $$\text{Live Camera Feed} \longrightarrow \text{Deteksi YOLO11 + Tracking SOP (Live HUD)} \longrightarrow \text{Laporan Audit Terminal + Rekaman Sesi (.avi)}$$

---

## 3. Cakupan SOP (SOP 3 Langkah — Model 3 Class)
Sistem memvalidasi 3 tahapan secara berurutan:
1. **Step 1: Kardus Terdeteksi** — Memastikan kardus pengiriman diletakkan dan terdeteksi di area meja kerja.
2. **Step 2: Lakban & Penyegelan Dus** — Memastikan kardus telah dilakban dan disegel dengan benar.
3. **Step 3: Tempel Resi Pengiriman** — Memastikan label resi pengiriman tertempel pada permukaan kardus.

---

## 4. Fitur Utama Sistem (Core Features)

### A. Real-Time Live Webcam Engine & Video Parser
- Membuka webcam lokal (`python main.py`) secara otomatis.
- Menampilkan jendela GUI live berkecepatan tinggi dengan bounding box dan HUD overlay.
- Tekan `q` atau `ESC` untuk menghentikan pemantauan live dan mencetak laporan akhir.

### B. Engine Deteksi Objek & Fitur Visual (YOLO11 + OpenCV)
- Pendeteksian objek utama (3 Class): `kardus`, `lakban`, `resi`.
- Ekstraksi fitur visual pendukung & Adaptive White Balance.

### C. Evaluator Sekuensial SOP (State Machine Tracker)
- Memantau status 3 step (Pending, In-Progress, Passed, Skipped).
- Mencatat timestamp detik kejadian untuk tiap langkah.
- Mendeteksi *violation rule* (misal: Step 2 dilewati langsung ke Step 3).

### D. Visual Frame Annotator (HUD Overlay)
- Melukis *bounding box* dan label nama objek pada video.
- Menampilkan **Live HUD Panel** pada sudut kiri atas video:
  - Checklist 3 langkah SOP dengan warna status dinamis.
  - Badge Status Akhir: `COMPLIANT (SOP Sesuai)` atau `NON-COMPLIANT (Melanggar SOP)`.

### E. Laporan Audit Terminal & Rekaman Sesi
- Setelah sesi pemantauan selesai, program mencetak ringkasan hasil di terminal (Tabel status per step, timestamp, catatan pelanggaran) dan menyimpan rekaman sesi `.avi`.

---

## 5. Spesifikasi Teknologi (Tech Stack)
- **Bahasa Pemrograman:** Python 3.10+
- **Model Object Detection:** Ultralytics YOLO11 (`yolo11n.pt` / `yolo11s.pt`)
- **Computer Vision & Video Stream:** OpenCV (`cv2`)
- **Deep Learning Framework:** PyTorch (`torch`)
- **Utilitas:** NumPy, tqdm (progress bar terminal)
- **Platform Run:** Terminal / Command Line Interface (Windows/Linux/macOS)

---

## 6. Alur Penggunaan (User Workflow)

```bash
# 1. Jalankan kamera real-time secara langsung (Default):
python main.py

# 2. Atau jalankan analisis pada file video rekaman:
python main.py --input sample_compliant.avi --output hasil_sesuai.avi
```
