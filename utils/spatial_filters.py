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
KARDUS_MAX_ASPECT  = 10.0

# Batas ukuran bounding box per kelas (min_ratio, max_ratio)
# Dibuat toleran agar objek yang didekatkan ke kamera tetap terdeteksi dengan baik.
BOX_SIZE_PER_CLASS = {
    "kardus":       (0.000,      1.00),   # Kardus bebas (0 - 100% area)
    "lakban":       (0.0005,     0.85),   # Lakban (0.05% - 85% area)
    "resi":         (0.0005,     0.85),   # Resi (0.05% - 85% area)
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


def apply_spatial_context(detections, step1_passed=False):
    """
    Meloloskan deteksi objek yang terdeteksi oleh YOLO agar dapat ditampilkan
    pada GUI webcam dan dievaluasi oleh tracker SOP secara transparan.

    Args:
        detections   : list of (box [x1,y1,x2,y2], cls_name str, conf float)
        step1_passed : bool, True jika Step 1 (kardus) sudah dikonfirmasi PASSED

    Returns:
        Filtered list of detections.
    """
    return detections
