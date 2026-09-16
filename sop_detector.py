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
import threading
import cv2
from collections import deque
from ultralytics import YOLO
import torch
torch.set_num_threads(4)

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
CONF_THRESHOLD_LIVE  = 0.25
CONF_THRESHOLD_VIDEO = 0.25

# Threshold per kelas untuk mode live — terkalibrasi responsif & presisi anti-halusinasi:
CONF_PER_CLASS_LIVE = {
    "kardus": 0.035,  # Menangkap kardus pada semua orientasi (horizontal/vertikal)
    "lakban": 0.24,   # Lakban nyata (0.25 - 0.90), tolak noise lipatan/tekstur kardus
    "resi":   0.22,   # Resi nyata (0.25 - 0.85), tolak noise teks/stiker kardus
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

        # Inisialisasi Haar Cascade Face Classifier untuk eliminasi false-positive wajah manusia
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    # ──────────────────────────────────────────────────────────────────────────
    # YOLO INFERENCE + FILTER DASAR (CONFIDENCE, SIZE, ASPECT RATIO)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _enhance_frame(frame):
        """
        Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization) pada channel L
        di ruang warna LAB agar kontras tekstur kardus meningkat secara adaptif.
        Sangat membantu pada kondisi pencahayaan redup atau warna kardus yang flat.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced

    def detect_frame(self, frame, is_live=False):
        """
        Jalankan YOLO inference pada satu frame.

        Returns:
            List of (box [x1,y1,x2,y2], cls_name str, conf float)
        """
        conf_thresh = CONF_THRESHOLD_LIVE if is_live else CONF_THRESHOLD_VIDEO
        detections  = []

        try:
            min_yolo_conf = min(CONF_PER_CLASS_LIVE.values()) if is_live else CONF_THRESHOLD_VIDEO
            imgsz = 320 if is_live else 640
            results = self.model(frame, imgsz=imgsz, conf=min_yolo_conf, iou=0.45, agnostic_nms=True, verbose=False)[0]
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

            fh, fw = frame.shape[:2]
            frame_area = max(fh * fw, 1)

            # Deteksi wajah cepat (20ms) pada thumbnail 160x120 untuk eliminasi deteksi wajah sebagai kardus
            detected_faces = []
            if is_live and self.face_cascade is not None and not self.face_cascade.empty():
                try:
                    gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
                    faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.2, minNeighbors=3, minSize=(16, 16))
                    sx, sy = fw / 160.0, fh / 120.0
                    detected_faces = [
                        (int(fx * sx), int(fy * sy), int((fx + fw_b) * sx), int((fy + fh_b) * sy))
                        for (fx, fy, fw_b, fh_b) in faces
                    ]
                except Exception:
                    pass

            kardus_dets = []
            other_dets  = []
            for d in detections:
                (x1, y1, x2, y2), cls_name, conf = d
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # 1. Anti-face filter universal: eliminasi semua objek jika centroid di area wajah
                is_face = False
                for fx1, fy1, fx2, fy2 in detected_faces:
                    if (fx1 - 20 <= cx <= fx2 + 20) and (fy1 - 20 <= cy <= fy2 + 35):
                        is_face = True
                        break
                if is_face:
                    continue

                if cls_name == "kardus":
                    w = max(x2 - x1, 1)
                    h = max(y2 - y1, 1)
                    area_ratio = (w * h) / frame_area
                    aspect = max(w, h) / min(w, h)

                    # 2. Anti-face filter cadangan: zona kepala atas-tengah frame
                    if is_live and (cy < 0.45 * fh and 0.22 * fw < cx < 0.78 * fw and aspect < 2.0 and area_ratio < 0.25):
                        continue

                    if area_ratio >= 0.025 and aspect <= 4.0:
                        kardus_dets.append(d)
                else:
                    other_dets.append(d)

            # Filter spasial & ukuran hanya untuk non-kardus
            other_dets = filter_min_box_size(other_dets, frame.shape)
            other_dets = filter_class_size_mismatch(other_dets, frame.shape)
            other_dets = filter_by_roi(other_dets, frame.shape)

            # Buang deteksi palsu di area panel HUD pojok kiri atas
            combined = kardus_dets + other_dets
            if is_live:
                detections = [
                    d for d in combined
                    if not ((d[0][0] + d[0][2]) / 2 < 285 and (d[0][1] + d[0][3]) / 2 < 140)
                ]
            else:
                detections = combined

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

    def apply_contextual_filter(self, detections, step1_passed=False, step2_passed=False):
        """
        Filter sekuensial & spasial kontekstual SOP:
        - Step 1 belum lulus: hanya terima kardus (lakban & resi ditolak).
        - Step 1 lulus, Step 2 belum lulus: hanya terima kardus & lakban (resi ditolak).
        - Step 2 lulus: terima semua (kardus, lakban, resi).
        """
        return apply_spatial_context(detections, step1_passed=step1_passed, step2_passed=step2_passed)

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

        # Optimasi hardware camera: hilangkan lag buffer & aktifkan streaming cepat
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

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

        # ── Konfigurasi tracker live webcam ──
        tracker = SOPSequenceTracker(debounce_threshold=3, min_duration_seconds=0.4)

        # ── Persistent Box & Tracker Cache (Anti-Kedip & Responsif) ──
        PERSIST_FRAMES = 3
        active_cache: dict[str, dict] = {}

        # ── Threaded Background Inference Worker (Decoupled dari GUI Display) ──
        # Menghilangkan lag & stuttering: GUI camera berjalan di FPS penuh (25-30 FPS),
        # sementara model YOLO memproses frame secara kontinu di thread terpisah.
        latest_frame_lock = threading.Lock()
        latest_frame_copy = None
        latest_dets_lock  = threading.Lock()
        shared_detections = []
        worker_running    = True
        infer_streak      = {"kardus": 0, "lakban": 0, "resi": 0}

        def inference_worker():
            nonlocal shared_detections
            while worker_running:
                frame_to_process = None
                with latest_frame_lock:
                    if latest_frame_copy is not None:
                        frame_to_process = latest_frame_copy

                if frame_to_process is not None:
                    try:
                        raw_dets = self.detect_frame(frame_to_process, is_live=True)

                        # ── Temporal Confirmation Filter (Eliminasi glitch / spike 1 frame) ──
                        seen_classes = {d[1] for d in raw_dets}
                        for c in ("kardus", "lakban", "resi"):
                            if c in seen_classes:
                                infer_streak[c] += 1
                            else:
                                infer_streak[c] = 0

                        confirmed = []
                        for d in raw_dets:
                            box, cls_name, conf = d
                            # Kardus: konfirmasi instan jika conf >= 0.15, butuh 2 siklus (~70ms) jika conf < 0.15
                            if cls_name == "kardus":
                                if conf >= 0.15 or infer_streak["kardus"] >= 2:
                                    confirmed.append(d)
                            # Lakban: konfirmasi instan jika conf >= 0.40, butuh 2 siklus jika 0.24 <= conf < 0.40
                            elif cls_name == "lakban":
                                if conf >= 0.40 or infer_streak["lakban"] >= 2:
                                    confirmed.append(d)
                            # Resi: konfirmasi instan jika conf >= 0.40, butuh 2 siklus jika 0.22 <= conf < 0.40
                            elif cls_name == "resi":
                                if conf >= 0.40 or infer_streak["resi"] >= 2:
                                    confirmed.append(d)

                        with latest_dets_lock:
                            shared_detections = confirmed
                    except Exception:
                        pass
                time.sleep(0.005)

        infer_thread = threading.Thread(target=inference_worker, daemon=True)
        infer_thread.start()

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
        print("Tekan 'r' untuk Reset SOP | Tekan 'q' atau ESC untuk Selesai\n")

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

            # Kirim salinan frame ke background inference worker
            with latest_frame_lock:
                latest_frame_copy = frame.copy()

            # Ambil deteksi terbaru yang dihasilkan inference worker
            with latest_dets_lock:
                raw_detections = list(shared_detections)

            step1_passed = (tracker.steps[0]["status"] == "PASSED")
            step2_passed = (tracker.steps[1]["status"] == "PASSED")
            detections = self.apply_contextual_filter(
                raw_detections, step1_passed=step1_passed, step2_passed=step2_passed
            )

            # ── Update cache persistensi ──
            for det in detections:
                cls_name = det[1]
                active_cache[cls_name] = {"det": det, "last_frame": frame_idx}

            # Proteksi anti-menempel pada cache tampilan:
            # 1. Jika ada lakban, buang kardus yang menempel/tumpang tindih pada lakban (< 2.8x luas lakban)
            if "lakban" in active_cache and "kardus" in active_cache:
                l_box = active_cache["lakban"]["det"][0]
                k_box = active_cache["kardus"]["det"][0]
                lx1, ly1, lx2, ly2 = l_box
                kx1, ky1, kx2, ky2 = k_box
                l_area = max((lx2 - lx1) * (ly2 - ly1), 1)
                k_area = max((kx2 - kx1) * (ky2 - ky1), 1)
                ix1, iy1 = max(lx1, kx1), max(ly1, ky1)
                ix2, iy2 = min(lx2, kx2), min(ly2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if (inter / l_area > 0.15 or inter / k_area > 0.20) and (k_area < 2.8 * l_area):
                        active_cache.pop("kardus", None)

            # 2. Jika ada resi, buang kardus dan lakban yang menempel/tumpang tindih pada resi
            if "resi" in active_cache:
                r_box = active_cache["resi"]["det"][0]
                rx1, ry1, rx2, ry2 = r_box
                r_area = max((rx2 - rx1) * (ry2 - ry1), 1)
                for other_cls in ("kardus", "lakban"):
                    if other_cls in active_cache:
                        o_box = active_cache[other_cls]["det"][0]
                        ox1, oy1, ox2, oy2 = o_box
                        o_area = max((ox2 - ox1) * (oy2 - oy1), 1)
                        ix1, iy1 = max(rx1, ox1), max(ry1, oy1)
                        ix2, iy2 = min(rx2, ox2), min(ry2, oy2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            if (inter / r_area > 0.15 or inter / o_area > 0.20) and (o_area < 2.8 * r_area):
                                active_cache.pop(other_cls, None)

            # Ambil deteksi aktif yang masih dalam batas persistensi
            display_detections = []
            active_classes = []
            for cls_name, item in list(active_cache.items()):
                # Proteksi seketika: jika kelas sudah tidak diizinkan oleh sekuensial SOP, buang seketika dari cache
                if not step1_passed and cls_name != "kardus":
                    active_cache.pop(cls_name, None)
                    continue
                if not step2_passed and cls_name == "resi":
                    active_cache.pop(cls_name, None)
                    continue

                if frame_idx - item["last_frame"] <= PERSIST_FRAMES:
                    display_detections.append(item["det"])
                    active_classes.append(cls_name)
                else:
                    active_cache.pop(cls_name, None)

            # ── Update tracker dengan kelas aktif yang stabil ──
            tracker.update(active_classes, current_time_sec)

            # ── Anotasi frame (bounding box persisten + HUD) ──
            annotated_frame = self.annotator.annotate_frame(
                frame, display_detections, tracker, current_fps=current_fps
            )

            cv2.putText(annotated_frame, "Tekan 'r' Reset  |  'q' / ESC Selesai",
                        (annotated_frame.shape[1] - 380, annotated_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            if out:
                out.write(annotated_frame)

            cv2.imshow(window_name, annotated_frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Pemantauan kamera dihentikan oleh pengguna.")
                break
            elif key == ord('r') or key == ord('R'):
                print("\n[INFO] SOP Sequence Tracker di-reset.")
                tracker = SOPSequenceTracker(debounce_threshold=3, min_duration_seconds=0.4)
                active_cache.clear()
                infer_streak = {"kardus": 0, "lakban": 0, "resi": 0}

        worker_running = False
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
            step2_passed     = (tracker.steps[1]["status"] == "PASSED")

            # ── Deteksi + filter dasar ──
            detections = self.detect_frame(frame, is_live=False)

            # ── Filter spasial kontekstual ──
            detections = self.apply_contextual_filter(
                detections, step1_passed=step1_passed, step2_passed=step2_passed
            )

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
