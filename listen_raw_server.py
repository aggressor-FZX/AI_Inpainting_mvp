#!/usr/bin/env python3
import socket, select, time, argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

def largest_bbox_from_mask(mask, margin=32):
    # mask: HxW uint8 {0,255}
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return (max(0, x1 - margin), max(0, y1 - margin),
            min(mask.shape[1], x2 + margin), min(mask.shape[0], y2 + margin))

def main(host="0.0.0.0", port=5000, W=1280, H=720,
         save_videos=False, show_side=False, max_frames=10**9,
         yolo_imgsz=640, roi_scale=0.5):
    BYTES = W * H * 3

    # ---- Models ----
    print("Loading models...")
    torch.backends.cudnn.benchmark = True  # speed autotune
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yolo = YOLO('yolov8n-seg.pt')
    try:
        yolo.fuse()
    except Exception:
        pass

    lama = SimpleLama()  # expects RGB uint8, returns RGB

    # ---- Writers (optional) ----
    if save_videos:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_orig = cv2.VideoWriter('./Original_Webcam.avi', fourcc, 30, (W, H))
        out_inpt = cv2.VideoWriter('./SegYolo_Inpaint_LaMa.avi', fourcc, 30, (W, H))
        print("Saving videos to disk.")
    else:
        out_orig = out_inpt = None
        print("Live-only (no video saving).")

    # ---- Server socket ----
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"Listening on {host}:{port} ... (30s timeout)")
    rdy = select.select([srv], [], [], 30)

    if not rdy[0]:
        print("Timeout: no connection")
        return

    conn, addr = srv.accept()
    print(f"Connected from {addr}")

    buf = bytearray(BYTES)
    mv  = memoryview(buf)
    frame_count = 0
    t0 = time.time()

    try:
        while frame_count < max_frames:
            # --- read one frame exactly ---
            got = 0
            while got < BYTES:
                n = conn.recv_into(mv[got:], BYTES - got)
                if n == 0:
                    print("Client closed")
                    return
                got += n

            frame_bgr = np.frombuffer(buf, np.uint8).reshape((H, W, 3))

            if save_videos:
                out_orig.write(frame_bgr)

            # ---- YOLO people mask (class 0) ----
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device=='cuda'), dtype=torch.float16 if device=='cuda' else torch.float32):
                    res = yolo.predict(frame_bgr, imgsz=yolo_imgsz, classes=[0], verbose=False, device=device)[0]

            if res.masks is not None and len(res.masks) > 0:
                # Combine all person masks
                m = res.masks.data  # Tensor [N, Hm, Wm] on device
                # Bring to CPU once; binarize
                mask_small = (m.any(dim=0).float().cpu().numpy() * 255).astype(np.uint8)
                # Resize to full frame with NN to keep hard edges
                mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((H, W), dtype=np.uint8)

            # ---- ROI LaMa (fast) ----
            bbox = largest_bbox_from_mask(mask, margin=48)
            if bbox is None:
                inpaint_bgr = frame_bgr
            else:
                x1, y1, x2, y2 = bbox
                roi_bgr = frame_bgr[y1:y2, x1:x2]
                roi_msk = mask[y1:y2, x1:x2]

                # Optional downscale for speed
                if roi_scale != 1.0:
                    rw = max(2, int((x2 - x1) * roi_scale))
                    rh = max(2, int((y2 - y1) * roi_scale))
                    roi_bgr_small = cv2.resize(roi_bgr, (rw, rh), interpolation=cv2.INTER_AREA)
                    roi_msk_small = cv2.resize(roi_msk, (rw, rh), interpolation=cv2.INTER_NEAREST)
                else:
                    roi_bgr_small = roi_bgr
                    roi_msk_small = roi_msk

                # LaMa expects RGB; ensure HxW uint8 {0,255} mask
                roi_rgb_small = cv2.cvtColor(roi_bgr_small, cv2.COLOR_BGR2RGB)
                inpaint_rgb_small = lama(roi_rgb_small, roi_msk_small)  # returns RGB
                if not isinstance(inpaint_rgb_small, np.ndarray):
                    inpaint_rgb_small = np.array(inpaint_rgb_small)

                # Upscale back if needed, convert to BGR
                if roi_scale != 1.0:
                    inpaint_rgb = cv2.resize(inpaint_rgb_small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                else:
                    inpaint_rgb = inpaint_rgb_small
                inpaint_bgr = frame_bgr.copy()
                inpaint_bgr[y1:y2, x1:x2] = cv2.cvtColor(inpaint_rgb, cv2.COLOR_RGB2BGR)

            # ---- Show / Save ----
            if save_videos:
                out_inpt.write(inpaint_bgr)

            show_frame = inpaint_bgr
            if show_side:
                show_frame = np.concatenate((frame_bgr, inpaint_bgr), axis=1)

            # HUD
            dt = time.time() - t0
            fps = frame_count / dt if dt > 0 else 0.0
            cv2.putText(show_frame, f"FPS:{fps:.1f}  frame:{frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Inpaint (fast)", show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    finally:
        conn.close()
        srv.close()
        if out_orig: out_orig.release()
        if out_inpt: out_inpt.release()
        cv2.destroyAllWindows()
        print(f"Done. Frames: {frame_count}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-save", action="store_true", help="no video writing")
    p.add_argument("--show-side", action="store_true", help="show side-by-side")
    p.add_argument("--frames", type=int, default=10**9)
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--imgsz", type=int, default=640, help="YOLO imgsz")
    p.add_argument("--roi-scale", type=float, default=0.5, help="downscale factor for inpaint ROI")
    args = p.parse_args()
    main(W=args.w, H=args.h,
         save_videos=not args.no_save,
         show_side=args.show_side,
         max_frames=args.frames,
         yolo_imgsz=args.imgsz,
         roi_scale=args.roi_scale)