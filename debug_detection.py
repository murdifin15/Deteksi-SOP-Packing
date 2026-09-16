"""
DEBUG TOOL — Tampilkan semua deteksi mentah YOLO tanpa filter apapun.
Gunakan ini untuk mendiagnosis apakah model mendeteksi lakban/resi sama sekali.

Jalankan:  python debug_detection.py
Tekan 'q' / ESC untuk keluar.
"""

import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "models/best_sop_packing.pt"
CONF_MIN   = 0.25   # Threshold sangat rendah — tampilkan SEMUA deteksi

COLORS = {
    "kardus": (255, 200, 0),
    "lakban": (0, 255, 100),
    "resi":   (0, 100, 255),
}
COLOR_DEFAULT = (180, 180, 180)

print(f"[DEBUG] Memuat model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"[DEBUG] Class names: {model.names}")
print(f"[DEBUG] Confidence minimum: {CONF_MIN:.0%}")
print()
print("=" * 60)
print(" Bounding box ditampilkan di jendela kamera.")
print(" Log terminal dicetak setiap 20 frame.")
print(" Tekan 'q' atau ESC untuk keluar.")
print("=" * 60)
print()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Tidak bisa membuka kamera!")
    exit(1)

cv2.namedWindow("DEBUG -- Raw YOLO (No Filter)", cv2.WINDOW_NORMAL)

MAPPING = {
    "kardus": "kardus", "box": "kardus", "cardboard box": "kardus",
    "cardboard_box": "kardus", "container": "kardus",
    "package": "kardus", "packages": "kardus",
    "lakban": "lakban", "tape": "lakban", "duct tape": "lakban",
    "duct_tape": "lakban", "sealer": "lakban",
    "resi": "resi", "resi_pengiriman": "resi",
    "shipping label": "resi", "shipping_label": "resi",
    "label": "resi", "labels": "resi",
}

frame_idx = 0
LOG_EVERY = 20

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
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
        label_str = f"{sop_name} ({conf:.0%}) raw={raw_name}"
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(display, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(22, y1)), color, -1)
        cv2.putText(display, label_str, (x1 + 4, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # Panel overlay
    frame_h, frame_w = frame.shape[:2]
    cv2.rectangle(display, (8, 8), (330, 125), (15, 15, 15), -1)
    cv2.rectangle(display, (8, 8), (330, 125), (255, 200, 0), 1)
    y_txt = 28
    cv2.putText(display, "DEBUG: RAW YOLO (conf>=20%, no filter)", (14, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
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

    cv2.putText(display, f"Frame #{frame_idx} | Total: {len(raw_list)} objek",
                (14, y_txt + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # Log terminal
    if frame_idx % LOG_EVERY == 0:
        t = time.strftime("%H:%M:%S")
        if raw_list:
            print(f"[{t}] Frame #{frame_idx} | {len(raw_list)} objek terdeteksi:")
            for d in raw_list:
                x1,y1,x2,y2,sop,raw,conf = d
                area_pct = ((x2-x1)*(y2-y1)) / (frame_w*frame_h) * 100
                print(f"  [{sop:8s}] conf={conf:.2f} raw='{raw}' area={area_pct:.1f}%")
        else:
            print(f"[{t}] Frame #{frame_idx} | Tidak ada objek (conf>={CONF_MIN:.0%})")

    cv2.imshow("DEBUG -- Raw YOLO (No Filter)", display)
    frame_idx += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[DEBUG] Selesai. Total frame: {frame_idx}")
