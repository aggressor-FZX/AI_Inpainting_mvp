#!/usr/bin/env python3
"""
listen_raw_server.py - Standalone script for real-time camera processing
Receives raw video stream from Windows ffmpeg, processes with YOLO + LaMa, displays live feed.
"""

import socket
import numpy as np
import cv2
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama
import time
import select
import argparse

def main(save_videos=True, max_frames=200):
    HOST, PORT = "0.0.0.0", 5000
    W, H = 1280, 720  # Resolution, must match client
    BYTES = W * H * 3

    # Initialize frame counter
    frame_count = 0

    # For live-only, process indefinitely
    if not save_videos:
        max_frames = 10**6  # Large number for continuous processing

    print("Loading models...")
    # Load YOLO model
    yolo_model = YOLO('yolov8n-seg.pt')

    # LaMa model
    lama_model = SimpleLama()

    # Initialize video writers if saving
    if save_videos:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_original = cv2.VideoWriter('./Original_Webcam.avi', fourcc, 30, (W, H))
        out_inpainted = cv2.VideoWriter('./SegYolov8_InptSimLama.avi', fourcc, 30, (W, H))
        print("Video saving enabled.")
    else:
        print("Live-only mode (no video saving).")

    # Create server socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    print("Waiting for Windows connection (timeout in 30 seconds)...")

    ready = select.select([srv], [], [], 30)

    if ready[0]:
        conn, addr = srv.accept()
        print(f"Connected to {addr}")
        
        buf = bytearray(BYTES)
        mv = memoryview(buf)
        start_time = time.time()
        
        while frame_count < max_frames:
            got = 0
            while got < BYTES:
                n = conn.recv_into(mv[got:], BYTES - got)
                if n == 0:
                    print("Connection closed by client")
                    break
                got += n
            if got < BYTES:
                break
            
            # Convert buffer to image
            frame = np.frombuffer(buf, np.uint8).reshape((H, W, 3))  # BGR format
            
            if save_videos:
                # Write original frame
                out_original.write(frame)
            
            # YOLO inference for people (class 0)
            results = yolo_model(frame, classes=[0], verbose=False)
            
            num_people = len(results[0].masks) if results[0].masks is not None else 0
            print(f"Frame {frame_count}: YOLO detected {num_people} people")
            
            # Create mask
            if results[0].masks is not None and len(results[0].masks) > 0:
                mask = results[0].masks.data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                # No people, use empty mask
                mask = np.zeros((H, W), dtype=np.uint8)
            
            # Inpaint using LaMa
            inpainted_result = lama_model(frame, mask)
            if isinstance(inpainted_result, np.ndarray):
                inpainted_frame = inpainted_result
            else:
                inpainted_frame = np.array(inpainted_result)
                if len(inpainted_frame.shape) == 3 and inpainted_frame.shape[2] == 3:
                    inpainted_frame = cv2.cvtColor(inpainted_frame, cv2.COLOR_RGB2BGR)
            
            if save_videos:
                # Write inpainted frame
                out_inpainted.write(inpainted_frame)
            
            # Display real-time side-by-side feed
            combined_frame = np.concatenate((frame, inpainted_frame), axis=1)
            
            # Add real-time status text
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            status_text = f"FPS: {fps:.1f} | Frame: {frame_count} | Recording: {'Yes' if save_videos else 'Live Only'}"
            cv2.putText(combined_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Real-time Side-by-Side Feed', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q' to quit real-time display")
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames... FPS: {fps:.2f}")
        
        conn.close()
    else:
        print("Timeout: No connection from Windows")

    srv.close()
    if save_videos:
        out_original.release()
        out_inpainted.release()
    cv2.destroyAllWindows()

    print(f"Processing complete! Processed {frame_count} frames.")
    if save_videos:
        print("Videos saved: ./Original_Webcam.avi and ./SegYolov8_InptSimLama.avi")
    else:
        print("Videos not saved (live-only mode)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time camera processing server")
    parser.add_argument("--no-save", action="store_true", help="Run in live-only mode without saving videos")
    parser.add_argument("--frames", type=int, default=200, help="Maximum frames to process when saving (default: 200)")
    args = parser.parse_args()

    save_videos = not args.no_save
    main(save_videos=save_videos, max_frames=args.frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time camera processing server")
    parser.add_argument("--no-save", action="store_true", help="Run in live-only mode without saving videos")
    parser.add_argument("--frames", type=int, default=200, help="Maximum frames to process when saving (default: 200)")
    args = parser.parse_args()

    save_videos = not args.no_save
    main(save_videos=save_videos, max_frames=args.frames)