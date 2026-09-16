"""
Core Detection Engine menggunakan YOLO11 & OpenCV
Mendukung Pemrosesan File Video dan Live Webcam Feed Real-Time.

Post-Training — Model Nyata (3 Class: kardus, lakban, resi):
- YOLO inference menggunakan best_sop_packing.pt yang sudah di-training
- Confidence threshold per kelas yang terkalibrasi
- Filter geometri: ROI, minimum box size, aspect ratio kardus
- Filter spasial: lakban/resi harus dekat kardus (kontekstual)
- Temporal smoothing: rolling window buffer anti-flicker
- Debounce + min_duration untuk konfirmasi SOP step
"""

import os
import time
import cv2
from collections import deque
from ultralytics import YOLO

from sop_tracker import SOPSequenceTracker
from annotator import SOPAnnotator
from utils.spatial_filters import (
    filter_by_roi,
    filter_min_box_size,
    filter_aspect_ratio,
    filter_class_size_mismatch,
    apply_spatial_context,
)


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI CONFIDENCE THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

# Threshold dasar per mode
CONF_THRESHOLD_LIVE  = 0.10
CONF_THRESHOLD_VIDEO = 0.25

# Threshold per kelas untuk mode live
CONF_PER_CLASS_LIVE = {
    "kardus": 0.04,   # Sangat peka agar kardus langsung tertangkap di kamera
    "lakban": 0.15,   # Responsif & stabil
    "resi":   0.15,   # Responsif & stabil
}

# Normalisasi nama kelas dari output raw YOLO → nama standar sistem
CLASS_NAME_MAPPING = {
    "kardus": "kardus", "box": "kardus", "cardboard box": "kardus",
    "cardboard_box": "kardus", "container": "kardus",
    "package": "kardus", "packages": "kardus", "carton": "kardus",
    "dus": "kardus", "kotak": "kardus",
    "lakban": "lakban", "tape": "lakban", "duct tape": "lakban",
    "duct_tape": "lakban", "sealer": "lakban", "solasi": "lakban",
    "resi": "resi", "resi_pengiriman": "resi",
    "shipping label": "resi", "shipping_label": "resi",
    "label": "resi", "labels": "resi", "barcode": "resi",
}

VALID_CLASSES = {"kardus", "lakban", "resi"}


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR UTAMA
# ─────────────────────────────────────────────────────────────────────────────

class SOPDetector:
    def __init__(self, model_path="models/best_sop_packing.pt"):
        chosen_path = model_path or "models/best_sop_packing.pt"

        if not os.path.exists(chosen_path):
            raise FileNotFoundError(
                f"[ERROR] File model weight '{chosen_path}' tidak ditemukan!\n"
                f"        Pastikan file model berada di folder: models/best_sop_packing.pt"
            )

        self.model_path = chosen_path
        self.annotator  = SOPAnnotator()
        self._frame_count = 0
        self._inference_error_count = 0  # Untuk log throttle error

        print(f"[INFO] Memuat Model YOLO ({chosen_path})...")

        try:
            self.model = YOLO(chosen_path)
            print(
                f"[SUCCESS] Model YOLO ({chosen_path}) berhasil dimuat. "
                f"Class names: {self.model.names}"
            )
        except Exception as e:
            raise RuntimeError(f"[ERROR] Gagal memuat model YOLO: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    # YOLO INFERENCE + FILTER DASAR (CONFIDENCE, SIZE, ASPECT RATIO)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_frame(self, frame, is_live=False):
        """
        Jalankan YOLO inference pada satu frame.
        Kardus langsung diloloskan tanpa filter pemotongan agar deteksi instan dan konsisten.

        Returns:
            List of (box [x1,y1,x2,y2], cls_name str, conf float)
        """
        conf_thresh = CONF_THRESHOLD_LIVE if is_live else CONF_THRESHOLD_VIDEO
        detections  = []

        try:
            min_yolo_conf = min(CONF_PER_CLASS_LIVE.values()) if is_live else CONF_THRESHOLD_VIDEO
            results = self.model(frame, conf=min_yolo_conf, iou=0.45, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf    = float(box.conf[0])
                cls_id  = int(box.cls[0])
                raw_name = self.model.names[cls_id].lower().replace("_", " ").strip()

                cls_name = CLASS_NAME_MAPPING.get(raw_name, raw_name)
                if cls_name not in VALID_CLASSES:
                    continue

                # ── Filter confidence per kelas ──
                target_conf = CONF_PER_CLASS_LIVE.get(cls_name, conf_thresh) if is_live else CONF_THRESHOLD_VIDEO
                if conf < target_conf:
                    continue

                detections.append(([x1, y1, x2, y2], cls_name, conf))

            # Kardus 100% diloloskan langsung tanpa filter restriktif
            kardus_dets = [d for d in detections if d[1] == "kardus"]
            other_dets  = [d for d in detections if d[1] != "kardus"]

            # Filter spasial & ukuran hanya untuk non-kardus
            other_dets = filter_min_box_size(other_dets, frame.shape)
            other_dets = filter_class_size_mismatch(other_dets, frame.shape)
            other_dets = filter_by_roi(other_dets, frame.shape)

            detections = kardus_dets + other_dets

        except Exception as e:
            # Log throttled — hanya cetak tiap 30 frame agar tidak spam
            self._inference_error_count += 1
            if self._inference_error_count % 30 == 1:
                print(
                    f"[WARNING] YOLO inference error "
                    f"(frame #{self._frame_count}, total error #{self._inference_error_count}): {e}"
                )

        self._frame_count += 1
        return detections

    # ──────────────────────────────────────────────────────────────────────────
    # FILTER SPASIAL KONTEKSTUAL (setelah YOLO + filter dasar)
    # ──────────────────────────────────────────────────────────────────────────

    def apply_contextual_filter(self, detections, step1_passed=False):
        """
        Filter lakban & resi berdasarkan konteks kehadiran kardus di frame.

        - Kardus ada di frame        -> terima semua (tidak dibatasi proximity)
        - Kardus tidak ada, step1 PASSED -> terima semua (konteks sudah terkonfirmasi)
        - Kardus tidak ada, step1 BELUM PASSED -> tolak lakban & resi (background noise)

        Args:
            detections   : output dari detect_frame()
            step1_passed : True jika Step 1 (kardus) sudah dikonfirmasi PASSED

        Returns:
            Filtered list of detections.
        """
        return apply_spatial_context(detections, step1_passed=step1_passed)

    # ──────────────────────────────────────────────────────────────────────────
    # PROCESS WEBCAM (LIVE REAL-TIME)
    # ──────────────────────────────────────────────────────────────────────────

    def process_webcam(self, camera_id=0, save_output_path=None):
        """Menjalankan deteksi real-time menggunakan Live Webcam Stream."""
        # Coba buka menggunakan cv2.CAP_DSHOW (DirectShow di Windows) untuk performa dan stabilitas maksimal
        if os.name == "nt":
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(
                f"Gagal membuka kamera indeks {camera_id}. Pastikan webcam terhubung!"
            )

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        # ── Ukur FPS nyata dari driver kamera, bukan hardcode ──
        driver_fps = cap.get(cv2.CAP_PROP_FPS)
        write_fps  = driver_fps if 5 < driver_fps <= 120 else 30.0

        out = None
        if save_output_path:
            fourcc = (
                cv2.VideoWriter_fourcc(*'MJPG') if save_output_path.endswith('.avi')
                else cv2.VideoWriter_fourcc(*'mp4v')
            )
            out = cv2.VideoWriter(save_output_path, fourcc, write_fps, (width, height))

        # ── Konfigurasi tracker: debounce 3 frame, min 0.4 detik nyata untuk live camera ──
        tracker = SOPSequenceTracker(debounce_threshold=3, min_duration_seconds=0.4)

        # ── Temporal Smoothing Buffer ──
        SMOOTH_WINDOW   = 4
        SMOOTH_MIN_HITS = 1
        class_buffer: dict[str, deque] = {
            cls: deque(maxlen=SMOOTH_WINDOW) for cls in VALID_CLASSES
        }

        # Cache deteksi terakhir per kelas — bounding box persisten (anti-kedip)
        last_seen_detections: dict = {}
        last_seen_frame: dict      = {}

        start_time   = time.time()
        frame_idx    = 0
        WARMUP_SECONDS = 3

        # ── Pengukur FPS Runtime Aktual ──
        fps_measure_start = time.time()
        fps_measure_frames = 0
        current_fps = 0.0

        window_name = "AI SOP Packing Audit System (Live Camera Feed)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n[LIVE WEBCAM BERJALAN]")
        print("Tekan tombol 'q' atau 'ESC' pada jendela kamera untuk menghentikan pemantauan...\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Gagal membaca frame dari webcam.")
                break

            current_time_sec = time.time() - start_time

            # ── Ukur FPS runtime setiap 30 frame ──
            fps_measure_frames += 1
            if fps_measure_frames >= 30:
                elapsed = time.time() - fps_measure_start
                current_fps = fps_measure_frames / max(elapsed, 0.001)
                fps_measure_frames = 0
                fps_measure_start  = time.time()

            # ── WARMUP: countdown, abaikan semua deteksi ──
            if current_time_sec < WARMUP_SECONDS:
                countdown_frame = frame.copy()
                sisa = int(WARMUP_SECONDS - current_time_sec) + 1
                fh, fw = countdown_frame.shape[:2]

                overlay = countdown_frame.copy()
                cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.45, countdown_frame, 0.55, 0, countdown_frame)

                label_ready = "BERSIAP... POSISIKAN KAMERA"
                (tw, _), _ = cv2.getTextSize(label_ready, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.putText(countdown_frame, label_ready,
                            ((fw - tw) // 2, fh // 2 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)

                cnt_str = str(sisa)
                (cw, ch), _ = cv2.getTextSize(cnt_str, cv2.FONT_HERSHEY_SIMPLEX, 4.5, 8)
                cv2.putText(countdown_frame, cnt_str,
                            ((fw - cw) // 2, fh // 2 + ch // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0, 210, 255), 8, cv2.LINE_AA)

                sub = "Deteksi dimulai setelah countdown"
                (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.putText(countdown_frame, sub,
                            ((fw - sw) // 2, fh // 2 + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

                if out:
                    out.write(countdown_frame)
                cv2.imshow(window_name, countdown_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[INFO] Pemantauan kamera dihentikan oleh pengguna.")
                    break
                continue
            # ── AKHIR WARMUP ──

            step1_passed = (tracker.steps[0]["status"] == "PASSED")

            # ── Deteksi + filter dasar (confidence, size, aspect ratio, ROI) ──
            detections = self.detect_frame(frame, is_live=True)

            # ── Filter spasial kontekstual (lakban/resi harus dekat kardus) ──
            detections = self.apply_contextual_filter(detections, step1_passed=step1_passed)

            # ── Update cache ──
            for det in detections:
                cls_name = det[1]
                last_seen_detections[cls_name] = det
                last_seen_frame[cls_name]       = frame_idx

            # ── Temporal smoothing buffer ──
            raw_detected = {d[1] for d in detections}
            for cls in class_buffer:
                class_buffer[cls].append(1 if cls in raw_detected else 0)

            smoothed_detected = {
                cls for cls, buf in class_buffer.items()
                if len(buf) >= SMOOTH_MIN_HITS and sum(buf) >= SMOOTH_MIN_HITS
            }

            # ── Update tracker dengan set yang sudah dismoothing ──
            tracker.update(list(smoothed_detected), current_time_sec)

            # ── Anotasi frame (bounding box langsung dari deteksi frame aktif + HUD) ──
            annotated_frame = self.annotator.annotate_frame(
                frame, detections, tracker, current_fps=current_fps
            )

            cv2.putText(annotated_frame, "Tekan 'q' atau 'ESC' untuk Selesai",
                        (annotated_frame.shape[1] - 360, annotated_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if out:
                out.write(annotated_frame)

            cv2.imshow(window_name, annotated_frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Pemantauan kamera dihentikan oleh pengguna.")
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        duration_sec = time.time() - start_time
        actual_fps   = frame_idx / max(duration_sec, 0.001)

        tracker.finalize()
        summary_report = tracker.get_summary_report(
            input_filename=f"Live Kamera Gudang (Cam #{camera_id})",
            duration_sec=duration_sec,
            total_frames=frame_idx,
            fps=actual_fps,
        )

        return summary_report, tracker.overall_status

    # ──────────────────────────────────────────────────────────────────────────
    # PROCESS VIDEO FILE
    # ──────────────────────────────────────────────────────────────────────────

    def process_video(self, input_video_path, output_video_path, progress_callback=None):
        """Memproses file video input dari awal sampai akhir."""
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"File video '{input_video_path}' tidak ditemukan!")

        cap          = cv2.VideoCapture(input_video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        fourcc = (
            cv2.VideoWriter_fourcc(*'mp4v') if output_video_path.endswith(".mp4")
            else cv2.VideoWriter_fourcc(*'MJPG')
        )
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        tracker   = SOPSequenceTracker(debounce_threshold=12, min_duration_seconds=0.8)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_sec = frame_idx / fps
            step1_passed     = (tracker.steps[0]["status"] == "PASSED")

            # ── Deteksi + filter dasar ──
            detections = self.detect_frame(frame, is_live=False)

            # ── Filter spasial kontekstual ──
            detections = self.apply_contextual_filter(detections, step1_passed=step1_passed)

            detected_classes = [d[1] for d in detections]
            tracker.update(detected_classes, current_time_sec)

            annotated_frame = self.annotator.annotate_frame(
                frame, detections, tracker, current_fps=fps
            )
            out.write(annotated_frame)

            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx, total_frames)

        cap.release()
        out.release()
        tracker.finalize()
        summary_report = tracker.get_summary_report(
            input_filename=os.path.basename(input_video_path),
            duration_sec=duration_sec,
            total_frames=total_frames,
            fps=fps,
        )

        return summary_report, tracker.overall_status
