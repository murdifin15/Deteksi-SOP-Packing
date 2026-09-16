"""
CLI Entry Point & Terminal Audit Reporter
SOP Packing Detection System (YOLO11 CLI & Live Real-Time Webcam)
"""

import argparse
import os
import sys
import warnings
from tqdm import tqdm

# Suppress urllib3/chardet version mismatch warning dari library requests
# agar laporan audit terminal tidak terganggu noise library eksternal
warnings.filterwarnings("ignore", category=Warning, module="requests")

from sop_detector import SOPDetector


def parse_args():
    parser = argparse.ArgumentParser(description="AI SOP Packing Compliance Detection System (YOLO11 CLI & Real-Time Webcam)")
    parser.add_argument("--webcam", "-w", action="store_true", default=True,
                        help="Jalankan mode Kamera Real-Time / Live Webcam secara langsung (Default: True)")
    parser.add_argument("--cam-id", type=int, default=0,
                        help="Indeks perangkat kamera/webcam (default: 0)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path file video input (.avi / .mp4). Jika diisi, mode akan beralih ke analisis file video.")
    parser.add_argument("--output", "-o", type=str, default="hasil_rekaman.avi",
                        help="Path file video output untuk menyimpan hasil (.avi / .mp4)")

    parser.add_argument("--model", "-m", type=str, default="models/best_sop_packing.pt",
                        help="Path file model weight YOLO11 (default: models/best_sop_packing.pt)")
    return parser.parse_args()


def main():
    args = parse_args()


    print("===================================================================================")
    print("        SISTEM DETEKSI KEPATUHAN SOP PACKING GUDANG (YOLO11 REAL-TIME)              ")
    print("===================================================================================")

    detector = SOPDetector(model_path=args.model)

    # 1. Mode Analisis File Video (jika parameter --input diisi)
    if args.input:
        if not os.path.exists(args.input):
            print(f"[ERROR] File video input '{args.input}' tidak ditemukan.")
            print("[INFO] Gunakan: python main.py --input <path_video.avi>")
            sys.exit(1)

        print(f" Mode        : Analisis File Video")
        print(f" File Input  : {args.input}")
        print(f" File Output : {args.output}")
        print(f" Model Weight: {detector.model_path}")
        print("-----------------------------------------------------------------------------------")

        pbar = None
        def progress_update(current_frame, total_frames):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total_frames, desc="[PROSES DETEKSI]", unit="frame", ncols=85)
            pbar.update(1)

        print("\nMemulai proses analisis per frame...")
        summary_report, overall_status = detector.process_video(
            input_video_path=args.input,
            output_video_path=args.output,
            progress_callback=progress_update
        )
        if pbar is not None:
            pbar.close()

    # 2. Mode Kamera Real-Time / Live Webcam (Default)
    else:
        print(f" Mode        : Live Camera / Real-Time Webcam (Indeks #{args.cam_id})")
        print(f" Rekam Sesi  : {args.output}")
        print(f" Model Weight: {detector.model_path}")
        print("-----------------------------------------------------------------------------------")

        summary_report, overall_status = detector.process_webcam(
            camera_id=args.cam_id,
            save_output_path=args.output
        )

    # Cetak Hasil Audit di Terminal
    print("\n")
    print(summary_report)
    if args.output:
        print(f" Video Output Tersimpan di: {args.output}")
    print("===================================================================================\n")


if __name__ == "__main__":
    main()
