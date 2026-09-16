"""
utils/spatial_filters.py
========================
Kumpulan fungsi filter geometri & spasial sebagai lapisan validasi
di atas output YOLO11, untuk mencegah false-positive (halusinasi).

Isi modul:
  - is_near_or_on_kardus()      : Cek proximity / overlap antara item & kardus box
  - filter_by_roi()             : Buang deteksi di luar zona kerja aktif (ROI)
  - filter_min_box_size()       : Buang deteksi dengan bounding box terlalu kecil
  - filter_aspect_ratio()       : Buang kardus dengan rasio aspek tidak wajar
  - filter_class_size_mismatch(): Buang deteksi yang ukurannya tidak masuk akal
                                  untuk kelasnya (mengurangi salah tebak kardus/lakban)
  - apply_spatial_context()     : Filter lakban/resi yang tidak ada konteks kardus
"""


# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA KONFIGURASI FILTER
# ─────────────────────────────────────────────────────────────────────────────

# Zona kerja aktif: (x1_ratio, y1_ratio, x2_ratio, y2_ratio) relatif frame.
# Diberi toleransi luas agar seluruh bidang pandang kamera terdeteksi dengan baik.
ROI_RATIO = (0.01, 0.01, 0.99, 0.99)

# Ukuran minimum bounding box sebagai rasio dari total area frame.
# Hanya menolak noise ekstrem (piksel acak) — sangat toleran.
BOX_MIN_AREA_RATIO = 0.0005

# Rasio aspek maksimum bounding box kardus (lebar/tinggi atau tinggi/lebar)
# Dibuat sangat toleran (20.0) agar semua perspektif / lipatan kardus selalu terdeteksi
KARDUS_MAX_ASPECT  = 20.0

# Batas ukuran bounding box per kelas (min_ratio, max_ratio)
# Dibuat presisi agar item kecil tidak salah tebak sebagai kardus dan sebaliknya
BOX_SIZE_PER_CLASS = {
    "kardus":       (0.020,      0.98),   # Kardus: 2% - 98% area frame
    "lakban":       (0.002,      0.45),   # Lakban: 0.2% - 45% area frame (tidak mungkin sebesar kardus)
    "resi":         (0.002,      0.40),   # Resi: 0.2% - 40% area frame (tidak mungkin sebesar kardus)
}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDASI SPASIAL (PROXIMITY / OVERLAP)
# ─────────────────────────────────────────────────────────────────────────────

def is_near_or_on_kardus(kardus_box, item_box, proximity_px=80, min_overlap=0.03):
    """
    Memeriksa apakah item_box (lakban/resi) berada di dalam, tumpang tindih,
    atau berdekatan (dalam jarak proximity_px piksel) dengan kardus_box.

    Ini penting untuk real-world: lakban yang ditempel di SISI kardus —
    bounding box YOLO-nya mungkin tidak overlap langsung dengan kardus box,
    tapi secara posisi sudah menempel di tepi kardus.

    Args:
        kardus_box   : [x1, y1, x2, y2] bounding box kardus
        item_box     : [x1, y1, x2, y2] bounding box item (lakban/resi)
        proximity_px : jarak piksel toleransi kedekatan
        min_overlap  : rasio minimum overlap item_area agar dianggap menempel

    Returns:
        True jika item berada di dalam / overlap / berdekatan dengan kardus.
    """
    cx1, cy1, cx2, cy2 = kardus_box
    ix1, iy1, ix2, iy2 = item_box

    # Cek 1: Overlap langsung
    ox1, oy1 = max(cx1, ix1), max(cy1, iy1)
    ox2, oy2 = min(cx2, ix2), min(cy2, iy2)
    if ox2 > ox1 and oy2 > oy1:
        overlap_area = (ox2 - ox1) * (oy2 - oy1)
        item_area    = max((ix2 - ix1) * (iy2 - iy1), 1)
        if (overlap_area / item_area) >= min_overlap:
            return True

    # Cek 2: Kedekatan — perluas kardus box sebesar proximity_px ke semua sisi
    if (ix1 <= cx2 + proximity_px and ix2 >= cx1 - proximity_px and
            iy1 <= cy2 + proximity_px and iy2 >= cy1 - proximity_px):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# FILTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_roi(detections, frame_shape):
    """
    Buang deteksi yang centroid-nya berada di luar zona kerja aktif (ROI).

    Args:
        detections  : list of (box [x1,y1,x2,y2], cls_name str, conf float)
        frame_shape : tuple (height, width, ...)

    Returns:
        Filtered list of detections.
    """
    fh, fw = frame_shape[:2]
    rx1 = int(ROI_RATIO[0] * fw)
    ry1 = int(ROI_RATIO[1] * fh)
    rx2 = int(ROI_RATIO[2] * fw)
    ry2 = int(ROI_RATIO[3] * fh)

    valid = []
    for det in detections:
        (x1, y1, x2, y2), cls, conf = det
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            valid.append(det)
    return valid


def filter_min_box_size(detections, frame_shape):
    """
    Buang deteksi dengan bounding box terlalu kecil (objek terlalu jauh / noise).

    Args:
        detections  : list of (box [x1,y1,x2,y2], cls_name str, conf float)
        frame_shape : tuple (height, width, ...)

    Returns:
        Filtered list of detections.
    """
    fh, fw = frame_shape[:2]
    frame_area = fh * fw
    min_area   = BOX_MIN_AREA_RATIO * frame_area

    return [
        det for det in detections
        if ((det[0][2] - det[0][0]) * (det[0][3] - det[0][1])) >= min_area
    ]


def filter_aspect_ratio(detections):
    """
    Buang bounding box kardus dengan rasio aspek tidak wajar (terlalu memanjang).
    Kardus nyata umumnya mendekati persegi atau persegi panjang normal.

    Args:
        detections : list of (box [x1,y1,x2,y2], cls_name str, conf float)

    Returns:
        Filtered list of detections (non-kardus selalu diloloskan).
    """
    valid = []
    for det in detections:
        (x1, y1, x2, y2), cls, conf = det
        if cls == "kardus":
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            if max(w, h) / min(w, h) > KARDUS_MAX_ASPECT:
                continue   # Terlalu memanjang — kemungkinan bukan kardus
        valid.append(det)
    return valid


def filter_class_size_mismatch(detections, frame_shape):
    """
    Tolak deteksi yang ukuran bounding box-nya tidak masuk akal untuk kelasnya.

    Ini membantu mengurangi salah tebak antara kardus dan lakban:
    - Model menebak 'kardus' tapi ukuran box sangat kecil (seperti lakban) → ditolak
    - Model menebak 'lakban' tapi ukuran box sangat besar (seperti kardus) → ditolak

    Batas ukuran per kelas dikonfigurasi di BOX_SIZE_PER_CLASS.

    Args:
        detections  : list of (box [x1,y1,x2,y2], cls_name str, conf float)
        frame_shape : tuple (height, width, ...)

    Returns:
        Filtered list of detections.
    """
    fh, fw = frame_shape[:2]
    frame_area = max(fh * fw, 1)
    valid = []

    for det in detections:
        (x1, y1, x2, y2), cls, conf = det
        if cls not in BOX_SIZE_PER_CLASS:
            valid.append(det)
            continue

        box_area  = (x2 - x1) * (y2 - y1)
        box_ratio = box_area / frame_area
        min_r, max_r = BOX_SIZE_PER_CLASS[cls]

        if box_ratio < min_r:
            # Terlalu kecil untuk kelas ini — kemungkinan salah tebak
            continue
        if box_ratio > max_r:
            # Terlalu besar untuk kelas ini — kemungkinan salah tebak
            continue

        valid.append(det)
    return valid


def suppress_conflicting_detections(detections, iou_threshold=0.35):
    """
    Menghilangkan deteksi ganda / konflik kelas:
    1. Objek Spesifik Memiliki Prioritas Atas Kontainer pada Lokasi yang Sama:
       - Jika 'lakban' dan 'kardus' tumpang tindih pada lokasi yang sama (misal roll lakban
         ditebak juga sebagai kardus karena warnanya cokelat/ada karton core):
         -> 'lakban' SELALU MENANG dan 'kardus' disupresi, kecuali jika ukuran kardus
            jauh lebih besar (>= 2.8x ukuran lakban, yaitu kardus nyata yang sedang dilakban).
       - Jika 'resi' dan 'kardus' / 'lakban' tumpang tindih pada ukuran sebanding:
         -> 'resi' SELALU MENANG dan deteksi lain disupresi.
    2. NMS Lintas Kelas:
       - Jika 2 deteksi dari kelas berbeda memiliki IoU > iou_threshold (0.35),
         pertahankan hanya yang memiliki prioritas kontekstual atau confidence tertinggi.
    """
    if len(detections) <= 1:
        return detections

    CLASS_PRIORITY = {"resi": 3, "lakban": 2, "kardus": 1}

    # Tahap 1: Eliminasi kardus palsu yang menempel di atas roll lakban atau resi
    lakban_boxes = [d for d in detections if d[1] == "lakban"]
    resi_boxes   = [d for d in detections if d[1] == "resi"]

    survivors = []
    for d in detections:
        box, cls_name, conf = d
        x1, y1, x2, y2 = box
        area = max((x2 - x1) * (y2 - y1), 1)

        # Cek jika d adalah kardus, apakah menempel pada lakban
        if cls_name == "kardus":
            is_stuck_on_lakban = False
            for l_box, l_cls, l_conf in lakban_boxes:
                lx1, ly1, lx2, ly2 = l_box
                l_area = max((lx2 - lx1) * (ly2 - ly1), 1)
                ix1, iy1 = max(x1, lx1), max(y1, ly1)
                ix2, iy2 = min(x2, lx2), min(y2, ly2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    # Jika kardus menutupi lakban atau sebaliknya, dan ukuran kardus tidak jauh lebih besar (< 2.8x)
                    if (inter / l_area > 0.15 or inter / area > 0.20) and (area < 2.8 * l_area):
                        is_stuck_on_lakban = True
                        break
            if is_stuck_on_lakban:
                continue

            # Cek jika d adalah kardus, apakah menempel pada resi
            is_stuck_on_resi = False
            for r_box, r_cls, r_conf in resi_boxes:
                rx1, ry1, rx2, ry2 = r_box
                r_area = max((rx2 - rx1) * (ry2 - ry1), 1)
                ix1, iy1 = max(x1, rx1), max(y1, ry1)
                ix2, iy2 = min(x2, rx2), min(y2, ry2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if (inter / r_area > 0.15 or inter / area > 0.20) and (area < 2.8 * r_area):
                        is_stuck_on_resi = True
                        break
            if is_stuck_on_resi:
                continue

        # Cek jika d adalah lakban, apakah menempel pada resi
        if cls_name == "lakban":
            is_stuck_on_resi = False
            for r_box, r_cls, r_conf in resi_boxes:
                rx1, ry1, rx2, ry2 = r_box
                r_area = max((rx2 - rx1) * (ry2 - ry1), 1)
                ix1, iy1 = max(x1, rx1), max(y1, ry1)
                ix2, iy2 = min(x2, rx2), min(y2, ry2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if (inter / r_area > 0.15 or inter / area > 0.20) and (area < 2.8 * r_area):
                        is_stuck_on_resi = True
                        break
            if is_stuck_on_resi:
                continue

        survivors.append(d)

    # Tahap 2: NMS lintas kelas untuk box dengan IoU tinggi
    sorted_dets = sorted(
        survivors,
        key=lambda d: (CLASS_PRIORITY.get(d[1], 0) * 10.0 + d[2]),
        reverse=True
    )
    kept = []

    for det in sorted_dets:
        box_a, cls_a, conf_a = det
        ax1, ay1, ax2, ay2 = box_a
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1)

        conflict = False
        for k_det in kept:
            box_b, cls_b, conf_b = k_det
            bx1, by1, bx2, by2 = box_b
            area_b = max((bx2 - bx1) * (by2 - by1), 1)

            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                union_area = area_a + area_b - inter_area
                iou = inter_area / max(union_area, 1)

                if iou > iou_threshold:
                    conflict = True
                    break

        if not conflict:
            kept.append(det)

    return kept


def suppress_overlapping_classes(detections, iou_threshold=0.35):
    """Alias kompatibilitas untuk suppress_conflicting_detections."""
    return suppress_conflicting_detections(detections, iou_threshold=iou_threshold)


def apply_spatial_context(detections, step1_passed=False, step2_passed=False):
    """
    Validasi spasial & konteks sekuensial SOP Packing:
    - Sebelum Step 1 PASSED: hanya terima 'kardus'. Tolak 'lakban' dan 'resi' (belum waktunya).
    - Sebelum Step 2 PASSED: hanya terima 'kardus' dan 'lakban'. Tolak 'resi' (belum waktunya).
    - Setelah Step 2 PASSED: terima semua ('kardus', 'lakban', 'resi').
    - Filter dominansi latar depan: jika item foreground (lakban/resi) terdeteksi kuat (conf >= 0.35),
      buang deteksi sekunder yang sangat lemah (conf < 0.08) untuk mencegah false-positive kardus di background.
    - Supresi deteksi yang berkonflik (IoU tinggi dari 2 kelas berbeda).
    """
    if not detections:
        return []

    # 1. Konteks Sekuensial SOP Packing
    contextual_dets = []
    for det in detections:
        cls_name = det[1]
        if not step1_passed:
            # Tahap 1: Belum ada kardus terverifikasi -> abaikan lakban & resi
            if cls_name != "kardus":
                continue
        elif not step2_passed:
            # Tahap 2: Sedang penyegelan lakban -> abaikan resi (belum ditempel)
            if cls_name == "resi":
                continue
        contextual_dets.append(det)

    if not contextual_dets:
        return []

    # 2. Foreground Dominance Filter:
    # Saat pengguna sedang memegang/memperlihatkan lakban atau resi dengan jelas (conf >= 0.35),
    # eliminasi deteksi objek lain yang sangat lemah / noise (conf < 0.08)
    has_strong_foreground = any(
        d[1] in ("lakban", "resi") and d[2] >= 0.35 for d in contextual_dets
    )
    if has_strong_foreground:
        contextual_dets = [
            d for d in contextual_dets
            if d[2] >= 0.08 or (d[1] in ("lakban", "resi") and d[2] >= 0.30)
        ]

    # 3. Supresi konflik tumpang tindih
    return suppress_conflicting_detections(contextual_dets)

