"""
Modul Visual Frame Annotator & Panel HUD Overlay (OpenCV)
Sesuai dengan Spesifikasi StyleGuide.md (Tajam, Jelas, High-Contrast & Kompak)
"""

import cv2
import numpy as np


class SOPAnnotator:
    # Skema Warna BGR ber-kontras tinggi berdasarkan StyleGuide.md
    COLOR_PASSED      = (80, 235, 50)       # Bright Emerald Green
    COLOR_FAILED      = (50, 50, 255)       # Bright Crimson Red
    COLOR_IN_PROGRESS = (10, 215, 255)      # Bright Yellow
    COLOR_PENDING     = (190, 190, 190)     # Light Gray
    COLOR_CYAN        = (255, 200, 0)       # Deep Cyan (default)
    COLOR_WHITE       = (255, 255, 255)
    COLOR_PANEL_BG    = (15, 15, 15)        # Dark Panel High Contrast

    # Warna bounding box per kelas objek (BGR)
    CLASS_COLORS = {
        "kardus": (235, 100, 20),   # Biru  #1464EB
        "lakban": (0,   210, 255),  # Kuning #FFD200
        "resi":   (50,  50,  255),  # Merah  #FF3232
    }

    def __init__(self):
        pass

    def draw_text_sharp(self, frame, text, pos, font_scale=0.40, color=(255, 255, 255), thickness=1):
        """
        Melukis teks dengan outline hitam 2px di belakangnya agar teks tajam,
        jelas, dan 100% terbaca tanpa efek blur pada video.
        """
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Stroke Hitam Tegas di Belakang
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # 2. Teks Warna Utama di Depan
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_bounding_boxes(self, frame, detections):
        """
        Melukis bounding box objek dan tag label pada frame secara tajam dan presisi.
        Warna berbeda per kelas: kardus=biru, lakban=kuning, resi=merah.
        :param detections: List tuple/dict [(bbox, label, confidence), ...]
                           di mana bbox = [x1, y1, x2, y2]
        """
        for box, label, conf in detections:
            x1, y1, x2, y2 = map(int, box)

            # Pilih warna sesuai kelas (default cyan jika kelas tidak dikenal)
            color = self.CLASS_COLORS.get(label, self.COLOR_CYAN)

            # Gambar Bounding Box 2px
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Tag Label Presisi
            caption = f"{label.replace('_', ' ').title()} ({conf:.0%})"
            font_scale = 0.45
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            tag_h = th + baseline + 6
            tag_w = tw + 8

            if y1 - tag_h >= 0:
                tag_y1 = y1 - tag_h
                tag_y2 = y1
                text_y = y1 - baseline - 2
            else:
                tag_y1 = y1
                tag_y2 = y1 + tag_h
                text_y = y1 + th + 2

            cv2.rectangle(frame, (x1, tag_y1), (x1 + tag_w, tag_y2), color, -1)
            cv2.putText(frame, caption, (x1 + 4, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    def draw_hud_panel(self, frame, tracker, current_fps=0.0):
        """
        Melukis Panel HUD (Heads-Up Display) Tajam, Jelas, & Kompak di pojok kiri atas.
        Menampilkan status 3 langkah SOP, badge kepatuhan, dan FPS runtime aktual.
        """
        panel_x1, panel_y1 = 12, 12
        panel_w, panel_h = 270, 125  # Sedikit lebih tinggi untuk baris FPS
        panel_x2, panel_y2 = panel_x1 + panel_w, panel_y1 + panel_h

        # Masking Transparansi High Contrast (88% Opacity)
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), self.COLOR_PANEL_BG, -1)

        alpha = 0.88
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Border luar HUD panel 1px
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), self.COLOR_CYAN, 1)

        # Header Panel Tajam
        self.draw_text_sharp(frame, "SOP PACKING AUDIT (3 LANGKAH)", (panel_x1 + 10, panel_y1 + 18),
                            font_scale=0.40, color=self.COLOR_WHITE, thickness=1)
        cv2.line(frame, (panel_x1 + 6, panel_y1 + 24), (panel_x2 - 6, panel_y1 + 24), (120, 120, 120), 1)

        # 3 Step SOP — Model Nyata 3 Class (kardus, lakban, resi)
        short_names = {
            1: "1. Kardus Terdeteksi",
            2: "2. Lakban & Segel",
            3: "3. Resi Pengiriman",
        }

        # Status tiap Step (Font Tajam scale 0.38, spasi 17px)
        y_offset = panel_y1 + 41
        for step in tracker.steps:
            st = step["status"]
            step_id = step["id"]

            if st == "PASSED":
                icon_tag = "[OK] "
                color = self.COLOR_PASSED
            elif st == "SKIPPED":
                icon_tag = "[FAIL]"
                color = self.COLOR_FAILED
            elif st == "IN_PROGRESS":
                icon_tag = "[RUN]"
                color = self.COLOR_IN_PROGRESS
            else:
                icon_tag = "[WAIT]"
                color = self.COLOR_PENDING

            display_name = short_names.get(step_id, step["name"])
            step_text = f"{icon_tag} {display_name}"
            self.draw_text_sharp(frame, step_text, (panel_x1 + 10, y_offset),
                                font_scale=0.38, color=color, thickness=1)
            y_offset += 17

        cv2.line(frame, (panel_x1 + 6, y_offset - 4), (panel_x2 - 6, y_offset - 4), (120, 120, 120), 1)

        # Badge Status Akhir Tajam
        if tracker.overall_status == "COMPLIANT":
            badge_text = "STATUS: COMPLIANT"
            badge_color = self.COLOR_PASSED
        elif tracker.overall_status == "NON_COMPLIANT":
            badge_text = "STATUS: MELANGGAR"
            badge_color = self.COLOR_FAILED
        else:
            badge_text = "STATUS: AUDIT BERJALAN..."
            badge_color = self.COLOR_IN_PROGRESS

        self.draw_text_sharp(frame, badge_text, (panel_x1 + 10, y_offset + 10),
                            font_scale=0.40, color=badge_color, thickness=1)

        # ── FPS Runtime Aktual ──
        fps_text = f"FPS: {current_fps:.1f}"
        fps_color = (
            (80, 235, 50)   if current_fps >= 20 else   # Hijau   — FPS bagus
            (10, 215, 255)  if current_fps >= 10 else   # Kuning  — FPS sedang
            (50, 50, 255)                               # Merah   — FPS rendah
        )
        self.draw_text_sharp(frame, fps_text,
                            (panel_x2 - 70, panel_y2 - 8),
                            font_scale=0.38, color=fps_color, thickness=1)

    def draw_guide_hint(self, frame, tracker):
        """
        Menampilkan panel petunjuk di pojok kanan atas:
        Memberitahu user apa yang perlu ditunjukkan ke kamera selanjutnya.
        """
        frame_h, frame_w = frame.shape[:2]

        # Tentukan hint berdasarkan step aktif saat ini
        hints = []
        for step in tracker.steps:
            st = step["status"]
            sid = step["id"]
            if st in ("PENDING", "IN_PROGRESS"):
                if sid == 1:
                    hints = [
                        "STEP 1 — TUNJUKKAN:",
                        "> Kardus / kotak pengiriman",
                        "  ke kamera dengan jelas",
                    ]
                elif sid == 2:
                    hints = [
                        "STEP 2 — TUNJUKKAN:",
                        "> Lakban / selotip ke kamera",
                        "  (kardus tidak harus ada)",
                    ]
                elif sid == 3:
                    hints = [
                        "STEP 3 — TUNJUKKAN:",
                        "> Resi / label ke kamera",
                        "  (kardus tidak harus ada)",
                    ]
                break
            elif st == "PASSED" and sid == 3:
                hints = ["Semua langkah SELESAI!", "> Tekan Q untuk laporan"]
                break

        if not hints:
            return

        # Ukur lebar teks terpanjang
        max_w = 0
        for h in hints:
            (tw, _), _ = cv2.getTextSize(h, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 20
        panel_h = len(hints) * 18 + 16
        panel_x2 = frame_w - 10
        panel_x1 = panel_x2 - panel_w
        panel_y1 = 12
        panel_y2 = panel_y1 + panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (10, 10, 30), -1)
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 200, 0), 1)

        y = panel_y1 + 14
        for i, line in enumerate(hints):
            color = (255, 200, 0) if i == 0 else (200, 255, 200)
            self.draw_text_sharp(frame, line, (panel_x1 + 8, y),
                                 font_scale=0.38, color=color, thickness=1)
            y += 18

    def annotate_frame(self, frame, detections, tracker, current_fps=0.0):
        """
        Metode utama: melukis bounding box, HUD panel, dan guide hint pada frame video.

        Args:
            frame        : frame OpenCV (BGR numpy array)
            detections   : list of (box, cls_name, conf)
            tracker      : SOPSequenceTracker instance
            current_fps  : FPS runtime aktual untuk ditampilkan di HUD

        Returns:
            Annotated frame (copy dari frame asli).
        """
        annotated_frame = frame.copy()
        self.draw_bounding_boxes(annotated_frame, detections)
        self.draw_hud_panel(annotated_frame, tracker, current_fps=current_fps)
        self.draw_guide_hint(annotated_frame, tracker)
        return annotated_frame
