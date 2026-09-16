# Sistem Deteksi Kepatuhan SOP Packing Gudang (YOLO11 Real-Time CLI)

Sistem deteksi otomatis kepatuhan tahapan SOP Packing Gudang berbasis **YOLO11**, **OpenCV**, dan **State Machine Tracker**.

Sistem ini memantau alur kerja pengepakan secara berurutan dan real-time:
```
Live Camera Feed / Video -> Deteksi YOLO11 + Tracking SOP (Live HUD) -> Laporan Audit Terminal + Rekaman Sesi (.avi)
```

---

## Fitur Utama

1. **Mode Live Real-Time Webcam (Default):** Langsung mengaktifkan kamera meja kerja saat program dijalankan (`python main.py`).
2. **Pendeteksian 3 Tahapan SOP:**
   - **Langkah 1:** Deteksi kardus packing pada area meja kerja (`kardus`).
   - **Langkah 2:** Proses lakban dan penyegelan kardus (`lakban`).
   - **Langkah 3:** Penempelan resi / label pengiriman pada kardus (`resi`).
3. **State Machine Evaluator & Anti-Hallucination:** Memvalidasi urutan langkah secara ketat, filter spasial konteks kardus, dan mekanisme debounce stabilitas deteksi.
4. **Visual HUD Overlay:** Menampilkan status tahapan SOP (Pending, In-Progress, Passed, Skipped) serta badge hasil kepatuhan secara langsung pada tampilan visual kamera.
5. **Dukungan Analisis File Video:** Mendukung pemrosesan video rekaman (`.avi` / `.mp4`) dengan progress bar terminal.
6. **Laporan Audit Terminal Otomatis:** Menghasilkan rekapitulasi kepatuhan SOP beserta timestamp detail per tahapan setelah sesi pemantauan selesai.

---

## Persyaratan Sistem dan Instalasi

### 1. Prasyarat
- Python 3.10 atau versi yang lebih baru
- Webcam / Kamera USB (untuk mode live)
- Pip (Python Package Installer)

### 2. Instalasi Dependensi
Buka terminal pada direktori proyek dan jalankan perintah berikut:
```bash
pip install -r requirements.txt
```

---

## Panduan Penggunaan

### 1. Mode Live Webcam Real-Time (Default)
Jalankan perintah berikut untuk memulai pemantauan langsung melalui webcam:
```bash
python main.py
```
- Jendela visual kamera akan terbuka dengan overlay panel status HUD.
- Tekan tombol **'q'** atau **'ESC'** pada jendela kamera untuk menghentikan pemantauan.
- Laporan audit akan langsung dicetak pada terminal dan video sesi disimpan secara otomatis.

### 2. Mode Analisis File Video
Untuk menganalisis file rekaman video yang sudah ada:
```bash
python main.py --input sample_video.avi --output hasil_analisis.avi
```

### 3. Pilihan Parameter CLI
| Parameter | Deskripsi | Default |
|---|---|---|
| `--input`, `-i` | Path file video input (.avi / .mp4) | `None` (Kamera Live) |
| `--output`, `-o` | Path penyimpanan video hasil analisis | `hasil_rekaman.avi` |
| `--cam-id` | Indeks ID kamera/webcam | `0` |
| `--model`, `-m` | Path file model weight YOLO11 | `models/best_sop_packing.pt` |

---

## Struktur Proyek

```
Project-SOP-PACKING/
├── models/
│   └── best_sop_packing.pt       # Bobot model terlatih YOLO11
├── utils/
│   ├── __init__.py
│   └── spatial_filters.py        # Filter spasial & White Balance
├── annotator.py                  # Renderer visual bounding box & HUD overlay
├── debug_detection.py            # Utilitas pengujian deteksi model
├── main.py                       # Titik masuk utama CLI & pelaporan audit
├── sop_detector.py               # Engine pipeline deteksi & video processor
├── sop_tracker.py                # State machine tracker tahapan SOP
├── requirements.txt              # Daftar dependensi Python
├── LICENSE                       # Lisensi MIT atas nama Murdifin
└── README.md                     # Dokumentasi proyek
```

---

## Contoh Output Terminal

```text
===================================================================================
        SISTEM DETEKSI KEPATUHAN SOP PACKING GUDANG (YOLO11 REAL-TIME)              
===================================================================================
 Mode        : Live Camera / Real-Time Webcam (Indeks #0)
 Rekam Sesi  : hasil_rekaman.avi
 Model Weight: models/best_sop_packing.pt
-----------------------------------------------------------------------------------

===================================================================================
                    HASIL AUDIT KEPATUHAN SOP PACKING GUDANG                       
===================================================================================
 File Input  : Live Kamera Gudang (Cam #0)
 Durasi Video: 18.5 Detik
 Total Frame : 555 Frame
 Frame Rate  : 30.0 FPS
 -----------------------------------------------------------------------------------
 DETAIL TAHAPAN SOP (3 LANGKAH PACKING):
 -----------------------------------------------------------------------------------
 [PASS] Step 1: Kardus Terdeteksi          | Timestamp: 00:01.2 - 00:04.5
 [PASS] Step 2: Lakban & Penyegelan Dus    | Timestamp: 00:04.5 - 00:08.1
 [PASS] Step 3: Tempel Resi Pengiriman     | Timestamp: 00:08.5 - 00:11.8
 -----------------------------------------------------------------------------------
 KESIMPULAN AKHIR: [ COMPLIANT - SOP SESUAI ]
 Catatan Audit   : Seluruh 3 tahapan SOP berhasil diselesaikan sesuai urutan.
===================================================================================
 Video Output Tersimpan di: hasil_rekaman.avi
===================================================================================
```

---

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE) atas nama **Murdifin**.
