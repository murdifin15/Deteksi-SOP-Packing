# Visual & Output Style Guide
## Konsep Tampilan Visual Overlay (OpenCV HUD) & Terminal Output

---

## 1. Skema Warna Visual (Color Palette BGR)
OpenCV menggunakan urutan warna **BGR (Blue, Green, Red)**. Berikut skema warna standar yang digunakan pada *video overlay* dan *bounding box*:

| Status / Komponen | Warna | Kode BGR (OpenCV) | Kode RGB (Hex) | Deskripsi |
| :--- | :--- | :--- | :--- | :--- |
| **PASSED / COMPLIANT** | Hijau Emerald | `(113, 204, 46)` | `#2ECC71` | Langkah SOP berhasil / Status SOP sesuai |
| **FAILED / NON-COMPLIANT** | Merah Krimson | `(60, 76, 231)` | `#E74C3C` | Langkah terlewati / Status SOP melanggar |
| **IN PROGRESS** | Kuning Amber | `(15, 196, 241)` | `#F1C40F` | Langkah SOP sedang berlangsung |
| **PENDING** | Abu-Abu Slate | `(166, 165, 149)` | `#95A5A6` | Langkah SOP belum dimulai |
| **HUD BACKGROUND** | Hitam Transparan | `(20, 20, 20)` | Opasitas 75% | Background panel informasi HUD |
| **BOX DETEKSI OBJEK** | Biru Cyan | `(255, 191, 0)` | `#00BFFF` | Bounding box penanda objek yang terdeteksi |

---

## 2. Spesifikasi Visual Video Overlay (HUD Panel)

### A. Layout Panel HUD (Sudut Kiri Atas Frame Video)
- **Ukuran:** Lebar 380px, tinggi dinamis (menyesuaikan 5 baris SOP + Header + Badge Status).
- **Latar Belakang:** Semi-transparan (`cv2.addWeighted` dengan alpha = 0.75).
- **Border:** Garis tipis 1px berwarna putih/cyan pada tepi panel.

### B. Struktur Tampilan HUD:
```text
┌─────────────────────────────────────────┐
│  AI PACKING AUDIT SYSTEM - SOP 1        │  <-- Header Panel (Font: FONT_HERSHEY_SIMPLEX)
├─────────────────────────────────────────┤
│  [OK]  Step 1: Scan Barcode             │  <-- Teks Hijau
│  [OK]  Step 2: Pelapisan Bubble Wrap    │  <-- Teks Hijau
│  [RUN] Step 3: Masuk Kardus             │  <-- Teks Kuning (Sedang aktif)
│  [WAIT]Step 4: Lakban Kardus            │  <-- Teks Abu-Abu
│  [WAIT]Step 5: Resi & Stiker Fragile    │  <-- Teks Abu-Abu
├─────────────────────────────────────────┤
│  STATUS: IN PROGRESS                    │  <-- Status Sementara
└─────────────────────────────────────────┘
```

### C. Bounding Box Objek (Detection Box)
- **Ketebalan Garis:** 2 Pixel.
- **Label Tag:** Latar belakang penuh warna sesuai objek dengan teks putih ukuran `scale=0.5`.
- **Format Label:** `Nama_Objek (Confidence %)` — *Contoh: `Bubble Wrap (94%)`*.

---

## 3. Format Output Ringkasan Terminal (CLI Log)

Setelah proses video selesai, program mencetak laporan visual di terminal menggunakan karakter ASCII/Markdown box:

```text
===================================================================================
                    HASIL AUDIT KEPATUHAN SOP PACKING GUDANG                       
===================================================================================
 File Input  : sample_compliant.mp4
 Durasi Video: 15.4 Detik
 Total Frame : 462 Frame
 Frame Rate  : 30 FPS
 -----------------------------------------------------------------------------------
 DETAIL TAHAPAN SOP (SOP 1 - BARANG FRAGILE):
 -----------------------------------------------------------------------------------
 [PASS] Step 1: Inspeksi & Scan Barcode    │ Timestamp: 00:02.1 - 00:04.3
 [PASS] Step 2: Pelapisan Bubble Wrap      │ Timestamp: 00:05.0 - 00:08.5
 [PASS] Step 3: Masukkan ke Dus            │ Timestamp: 00:09.1 - 00:11.2
 [PASS] Step 4: Lakban & Penyegelan Dus    │ Timestamp: 00:11.8 - 00:13.6
 [PASS] Step 5: Tempel Resi & Stiker Fragile│ Timestamp: 00:14.0 - 00:15.1
 -----------------------------------------------------------------------------------
 KESIMPULAN AKHIR: [ COMPLIANT - SOP SESUAI ]
 Catatan Audit   : Seluruh 5 tahapan SOP berhasil dilakukan sesuai urutan.
===================================================================================
 Video Output Tersimpan di: hasil_sesuai.mp4
===================================================================================
```
