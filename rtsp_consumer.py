#!/usr/bin/env python3
"""
Simple RTSP/HTTP MJPEG consumer for the MVp pipeline.

Usage:
  python3 rtsp_consumer.py rtsp://<WINDOWS-IP>:8554/mystream

What it does:
 - Opens the provided stream with OpenCV (preferred) or ffmpeg fallback.
 - Saves periodic frames to `stream_frames/` and logs to `rtsp_consumer.log`.
 - Exits cleanly on Ctrl+C.

Notes:
 - Start an RTSP server on Windows (e.g., rtsp-simple-server) and push the camera with ffmpeg.
 - Example Windows push command (PowerShell):
     ffmpeg -f dshow -i video="YOUR CAMERA NAME" -vcodec libx264 -preset veryfast -tune zerolatency -f rtsp rtsp://0.0.0.0:8554/mystream
 - Replace the URL below with your stream URL.
"""
import sys
import os
import time
from datetime import datetime
import signal

try:
    import cv2
except Exception:
    print('OpenCV is required (cv2). Activate your virtualenv and install opencv-python.')
    raise

OUT_DIR = os.path.join(os.path.dirname(__file__), 'stream_frames')
LOG = os.path.join(os.path.dirname(__file__), 'rtsp_consumer.log')
STOP = False


def signal_handler(sig, frame):
    global STOP
    STOP = True


def log(msg):
    line = f"[{datetime.now().isoformat(sep=' ')}] {msg}\n"
    print(line, end='')
    with open(LOG, 'a') as f:
        f.write(line)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def consume(url, frame_interval=5):
    ensure_dirs()
    log(f"Opening stream: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        log('cv2.VideoCapture failed to open the stream')
        return

    count = 0
    while not STOP:
        ret, frame = cap.read()
        if not ret or frame is None:
            log('Failed to read frame from stream; retrying in 1s...')
            time.sleep(1)
            continue

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(OUT_DIR, f'frame_{ts}.jpg')
        cv2.imwrite(out_path, frame)
        log(f'Saved frame {out_path} shape={frame.shape}')
        count += 1
        for _ in range(frame_interval):
            if STOP:
                break
            time.sleep(1)

    cap.release()
    log('Stream consumer stopped')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 rtsp_consumer.py <stream_url>')
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    consume(sys.argv[1])
