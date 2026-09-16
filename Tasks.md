# Checklist Tasks (MVP Scope)
## Tahapan Pengerjaan Sistem Deteksi SOP Packing Gudang (YOLO11 CLI)

---

## Daftar Tugas (Task List Breakdown)

- [x] **Task 1: Inisialisasi Environment & Dependensi (`requirements.txt`)**
  - Buat file `requirements.txt` berisi dependensi: `ultralytics`, `opencv-python`, `torch`, `numpy`, `tqdm`.
  - Pastikan environment Python di lokal terverifikasi.

- [x] **Task 2: Generator Video Simulasi (`generate_demo_video.py`)**
  - Buat script pembuat video sintetis 30 FPS.
  - Skenario 1: `sample_compliant.mp4` (menstimulasikan 5 step lengkap sesuai urutan).
  - Skenario 2: `sample_non_compliant.mp4` (menstimulasikan Step 2 terlewati / out of order).

- [x] **Task 3: State Machine Evaluator SOP (`sop_tracker.py`)**
  - Buat class `SOPSequenceTracker` untuk 3 langkah SOP (Kardus, Lakban, Resi).
  - Tambahkan fungsi pembaruan status step per frame.
  - Tambahkan logika kalkulasi timestamp dan pendeteksi pelanggaran urutan.

- [x] **Task 4: Modul Visual Annotator & HUD Overlay (`annotator.py`)**
  - Buat modul OpenCV untuk menggambar bounding box objek & label (kardus, lakban, resi).
  - Implementasikan panel HUD transparan 3 langkah di sudut kiri atas video sesuai `StyleGuide.md`.
  - Tambahkan penggambar badge status akhir (`COMPLIANT` / `NON-COMPLIANT`).

- [x] **Task 5: Core Detection Pipeline dengan YOLO11 (`sop_detector.py`)**
  - Buat class `SOPDetector` yang mengintegrasikan model YOLO11 3-class dari `models/best.pt`.
  - Buat parser per-frame: ekstrak objek & fitur (kardus, lakban, resi).
  - Hubungkan hasil ekstraksi objek ke `SOPSequenceTracker` dan `SOPAnnotator`.

- [x] **Task 6: Script Utama & Terminal Audit Reporter (`main.py`)**
  - Buat entry point CLI dengan argumen `--input` dan `--output`.
  - Tambahkan *progress bar* (`tqdm`) saat video diproses.
  - Tambahkan fungsi pencetak laporan audit ASCII di terminal saat selesai.

- [x] **Task 7: Panduan Penggunaan & Dokumentasi (`README.md`)**
  - Tulis panduan lengkap langkah eksekusi dari instalasi hingga melihat video hasil.

- [x] **Task 8: Pengujian Akhir (End-to-End Verification)**
  - Eksekusi pengujian pada `sample_compliant.mp4` & `sample_non_compliant.mp4`.
  - Verifikasi file video output dan integritas laporan audit terminal.
