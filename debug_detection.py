"""
DEBUG TOOL — Tampilkan semua deteksi mentah YOLO tanpa filter apapun.
Gunakan ini untuk mendiagnosis apakah model mendeteksi kardus/lakban/resi.

Jalankan:  python debug_detection.py --cam-id 1
Tekan 'q' / ESC untuk keluar.
"""

import os
import argparse
import time
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Debug Tool Real-Time YOLO Detection")
parser.add_argument("--cam-id", type=int, default=1, help="Indeks ID Kamera (default: 1)")
parser.add_argument("--conf", type=float, default=0.03, help="Confidence threshold minimum (default: 0.03)")
args = parser.parse_args()

MODEL_PATH = "models/best_sop_packing.pt"
CONF_MIN   = args.conf

COLORS = {
    "kardus": (235, 100, 20),
    "lakban": (0, 210, 255),
    "resi":   (50, 50, 255),
}
COLOR_DEFAULT = (180, 180, 180)

print(f"[DEBUG] Memuat model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"[DEBUG] Class names: {model.names}")
print(f"[DEBUG] Confidence minimum: {CONF_MIN:.0%}")
print(f"[DEBUG] Membuka kamera ID #{args.cam_id}...")
print()
print("=" * 60)
print(" Bounding box & confidence ditampilkan di jendela kamera.")
print(" Log terminal dicetak setiap 15 frame.")
print(" Tekan 'q' atau ESC untuk keluar.")
print("=" * 60)
print()

if os.name == "nt":
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam_id)
else:
    cap = cv2.VideoCapture(args.cam_id)

if not cap.isOpened():
    print(f"[ERROR] Tidak bisa membuka kamera ID #{args.cam_id}!")
    exit(1)

cv2.namedWindow("DEBUG -- Raw YOLO Detections", cv2.WINDOW_NORMAL)

MAPPING = {
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

frame_idx = 0
LOG_EVERY = 15

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_MIN, verbose=False)[0]
    display = frame.copy()
    raw_list = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf    = float(box.conf[0])
        cls_id  = int(box.cls[0])
        raw_name = model.names[cls_id].lower().strip()
        sop_name = MAPPING.get(raw_name, raw_name)

        if conf < CONF_MIN:
            continue

        raw_list.append((x1, y1, x2, y2, sop_name, raw_name, conf))

        color = COLORS.get(sop_name, COLOR_DEFAULT)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label_str = f"{sop_name.upper()} ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(display, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(22, y1)), color, -1)
        cv2.putText(display, label_str, (x1 + 4, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Panel overlay info
    frame_h, frame_w = frame.shape[:2]
    cv2.rectangle(display, (8, 8), (350, 125), (15, 15, 15), -1)
    cv2.rectangle(display, (8, 8), (350, 125), (255, 200, 0), 1)
    y_txt = 28
    cv2.putText(display, f"DEBUG RAW YOLO (conf>={CONF_MIN:.0%})", (14, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    y_txt += 22

    for cls_chk, lbl in [("kardus", "Kardus"), ("lakban", "Lakban"), ("resi", "Resi")]:
        matches = [d for d in raw_list if d[4] == cls_chk]
        if matches:
            best_conf = max(d[6] for d in matches)
            color_txt = COLORS.get(cls_chk, COLOR_DEFAULT)
            txt = f"[DETECTED] {lbl}: {best_conf:.0%}"
        else:
            color_txt = (90, 90, 90)
            txt = f"[  ---   ] {lbl}: tidak ada"
        cv2.putText(display, txt, (14, y_txt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color_txt, 1, cv2.LINE_AA)
        y_txt += 18

    cv2.putText(display, f"Frame #{frame_idx} | Total Objek: {len(raw_list)}",
                (14, y_txt + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # Log terminal
    if frame_idx % LOG_EVERY == 0:
        t = time.strftime("%H:%M:%S")
        if raw_list:
            print(f"[{t}] Frame #{frame_idx} | {len(raw_list)} objek terdeteksi:")
            for d in raw_list:
                x1, y1, x2, y2, sop, raw, conf = d
                area_pct = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h) * 100
                print(f"  [{sop:8s}] conf={conf:.2f} ({conf:.0%}) area={area_pct:.1f}%")
        else:
            print(f"[{t}] Frame #{frame_idx} | Tidak ada objek (conf>={CONF_MIN:.0%})")

    cv2.imshow("DEBUG -- Raw YOLO Detections", display)
    frame_idx += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[DEBUG] Selesai. Total frame: {frame_idx}")
