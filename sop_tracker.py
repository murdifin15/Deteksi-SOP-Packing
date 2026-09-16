"""
State Machine Evaluator untuk Sekuensial SOP Packing Gudang
SOP 3 Langkah — Model Nyata 3 Class (kardus, lakban, resi)

Class dari Model Nyata:
  0: kardus          (Kardus / Box pengiriman)
  1: lakban          (Lakban penyegel dus)
  2: resi            (Resi / Label pengiriman)

SOP 3 Langkah:
  Step 1: Kardus Terdeteksi          → class: kardus
  Step 2: Lakban & Penyegelan Dus    → class: lakban (kardus harus ada atau step1 PASSED)
  Step 3: Tempel Resi Pengiriman     → class: resi   (kardus harus ada atau step1 PASSED)

Anti-Hallucination v4:
- Debounce: 4 frame (live) / 12 frame (video) — dikonfigurasi dari sop_detector.py
- Dual condition PASSED: debounce + min_duration_seconds
- Decay: frames_consecutive direset ke 0 saat objek hilang (total_active_duration tetap stabil)
- Konteks Spasial: lakban/resi hanya aktif jika ada kardus di frame atau step1 sudah PASSED
  (filter spasial dilakukan di sop_detector.py sebelum data masuk ke tracker ini)
"""


class SOPSequenceTracker:
    def __init__(self, debounce_threshold=35, min_duration_seconds=1.5):
        """
        Inisialisasi tracker 3 tahapan SOP Packing.

        :param debounce_threshold: Frame berturut-turut untuk konfirmasi step.
                                   35 frame = ~1.2 detik di 30 FPS.
        :param min_duration_seconds: Durasi minimum total aktif (detik) sebelum PASSED.
        """
        self.debounce_threshold  = debounce_threshold
        self.min_duration_seconds = min_duration_seconds

        self.steps = [
            {
                "id": 1,
                "name": "Step 1: Kardus Terdeteksi",
                "status": "PENDING",
                "start_time": None,
                "end_time": None,
                "frames_active": 0,
                "frames_consecutive": 0,
                "total_active_duration": 0.0,
                "last_active_time": None,
            },
            {
                "id": 2,
                "name": "Step 2: Lakban & Penyegelan Dus",
                "status": "PENDING",
                "start_time": None,
                "end_time": None,
                "frames_active": 0,
                "frames_consecutive": 0,
                "total_active_duration": 0.0,
                "last_active_time": None,
            },
            {
                "id": 3,
                "name": "Step 3: Tempel Resi Pengiriman",
                "status": "PENDING",
                "start_time": None,
                "end_time": None,
                "frames_active": 0,
                "frames_consecutive": 0,
                "total_active_duration": 0.0,
                "last_active_time": None,
            },
        ]

        self.overall_status = "IN_PROGRESS"  # IN_PROGRESS | COMPLIANT | NON_COMPLIANT
        self.violations = []
        self._prev_timestamp = None

    def format_timestamp(self, seconds):
        """Mengubah detik float menjadi string format MM:SS.S"""
        if seconds is None:
            return "--:--"
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:04.1f}"

    def update(self, detected_classes, current_timestamp_sec):
        """
        Memproses frame saat ini dengan kelas objek yang terdeteksi.

        :param detected_classes: list/set string kelas terdeteksi pada frame ini.
        :param current_timestamp_sec: timestamp video dalam detik.
        """
        detected_set = set(detected_classes)

        # Hitung delta waktu sejak frame sebelumnya
        delta_t = 0.0
        if self._prev_timestamp is not None:
            delta_t = max(0.0, current_timestamp_sec - self._prev_timestamp)
        self._prev_timestamp = current_timestamp_sec

        # Pemetaan kelas terdeteksi ke indikator step
        has_kardus = "kardus" in detected_set
        has_lakban = "lakban" in detected_set or "tape" in detected_set
        has_resi   = ("resi" in detected_set or "resi_pengiriman" in detected_set
                      or "shipping_label" in detected_set or "label" in detected_set)

        # Status step sebelumnya untuk validasi konteks urutan SOP
        step1_passed = self.steps[0]["status"] == "PASSED"

        # Kondisi aktif per step: objek terdeteksi langsung mengaktifkan step-nya masing-masing
        step_active_indicators = {
            1: has_kardus,
            2: has_lakban,
            3: has_resi,
        }

        for step in self.steps:
            step_id   = step["id"]
            is_active = step_active_indicators.get(step_id, False)

            if is_active:
                step["frames_consecutive"] += 1
                step["frames_active"]      += 1

                # Akumulasi durasi aktif (detik nyata)
                if delta_t > 0:
                    step["total_active_duration"] += delta_t

                if step["start_time"] is None:
                    step["start_time"] = current_timestamp_sec
                step["end_time"]         = current_timestamp_sec
                step["last_active_time"] = current_timestamp_sec

                # Update status visual menjadi IN_PROGRESS saat sudah cukup aktif
                if (step["status"] == "PENDING" and
                        step["frames_consecutive"] >= max(5, self.debounce_threshold // 3)):
                    step["status"] = "IN_PROGRESS"

                # PASSED: KEDUA syarat harus terpenuhi
                #   1. frames_consecutive >= debounce_threshold
                #   2. total_active_duration >= min_duration_seconds
                if (step["status"] in ("PENDING", "IN_PROGRESS") and
                        step["frames_consecutive"] >= self.debounce_threshold and
                        step["total_active_duration"] >= self.min_duration_seconds):
                    step["status"] = "PASSED"

                    # Evaluasi urutan (sequence compliance check)
                    for prev_idx in range(step_id - 1):
                        prev_step = self.steps[prev_idx]
                        if prev_step["status"] in ("PENDING", "IN_PROGRESS"):
                            prev_step["status"] = "SKIPPED"
                            msg = (f"Pelanggaran: {prev_step['name']} dilewati "
                                   f"(langsung ke {step['name']}).")
                            if msg not in self.violations:
                                self.violations.append(msg)

            else:
                # Soft-decay: kurangi frames_consecutive sebesar 1 (bukan hard-reset ke 0)
                # agar flicker 1-2 frame tidak menghancurkan akumulasi progres kardus.
                step["frames_consecutive"] = max(0, step["frames_consecutive"] - 1)

    def finalize(self):
        """Evaluasi kesimpulan akhir setelah seluruh video selesai diproses."""
        passed_count = sum(1 for s in self.steps if s["status"] == "PASSED")

        if passed_count == 3 and len(self.violations) == 0:  # 3 step
            self.overall_status = "COMPLIANT"
        else:
            self.overall_status = "NON_COMPLIANT"
            for step in self.steps:
                if step["status"] in ("PENDING", "IN_PROGRESS", "SKIPPED"):
                    step["status"] = "SKIPPED"
                    v_msg = f"{step['name']} tidak dilaksanakan atau terlewati."
                    if v_msg not in self.violations:
                        self.violations.append(v_msg)

    def get_summary_report(self, input_filename, duration_sec, total_frames, fps):
        """Menghasilkan teks string laporan audit SOP berformat tabel ASCII."""
        lines = []
        lines.append("=" * 83)
        lines.append("               HASIL AUDIT KEPATUHAN SOP PACKING GUDANG (3 LANGKAH)              ")
        lines.append("=" * 83)
        lines.append(f" File Input  : {input_filename}")
        lines.append(f" Durasi Video: {duration_sec:.1f} Detik")
        lines.append(f" Total Frame : {total_frames} Frame")
        lines.append(f" Frame Rate  : {fps:.1f} FPS")
        lines.append("-" * 83)
        lines.append(" DETAIL TAHAPAN SOP (Model Nyata 3 Class - Kardus, Lakban, Resi):")
        lines.append("-" * 83)

        for step in self.steps:
            st = step["status"]
            if st == "PASSED":
                status_tag = "[PASS]"
            elif st == "SKIPPED":
                status_tag = "[FAIL]"
            elif st == "IN_PROGRESS":
                status_tag = "[RUN] "
            else:
                status_tag = "[WAIT]"

            start_str = self.format_timestamp(step["start_time"])
            end_str   = self.format_timestamp(step["end_time"])
            dur_str   = f"{step['total_active_duration']:.1f}s aktif"
            time_str  = (f"Timestamp: {start_str} - {end_str} | {dur_str}"
                         if step["start_time"] is not None else "Timestamp: Terlewati")
            lines.append(f" {status_tag} {step['name']:<42} | {time_str}")

        lines.append("-" * 83)
        status_text = ("COMPLIANT - SOP SESUAI" if self.overall_status == "COMPLIANT"
                       else "NON-COMPLIANT - MELANGGAR SOP")
        lines.append(f" KESIMPULAN AKHIR: [ {status_text} ]")

        if self.violations:
            lines.append(" Catatan Audit   : " + "; ".join(self.violations))
        else:
            lines.append(" Catatan Audit   : Seluruh 3 tahapan SOP berhasil dilakukan sesuai urutan.")

        lines.append("=" * 83)
        return "\n".join(lines)
