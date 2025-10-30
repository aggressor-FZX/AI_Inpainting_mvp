# Camera capture issue report

Date: 2025-10-28

This document summarizes the webcam capture corruption encountered when running the pipeline in WSL2, what we tried, reproduction steps, diagnostics and evidence, the current mitigations in the codebase, and recommended next steps.

---

## Executive summary

- Symptom: single-frames captured from `/dev/video0` inside WSL2 frequently contain a solid green band across the bottom region of the image (a run of rows with identical pure green pixels). The corruption appears before any ML processing (YOLO/LaMa) and is present in raw ffmpeg captures.
- Root cause: most likely a truncated MJPEG packet / decoder mismatch introduced by the USB→usbipd→v4l2 pipeline or by format/stride packing (camera advertises an unusual `1280x480` MJPG packing among other formats). ffmpeg logs show MJPEG decode warnings such as `EOI missing, emulating` and `overread` which are consistent with truncated JPEG streams.
- Impact: the downstream pipeline (YOLO segmentation + LaMa inpainting) receives corrupted frames, causing incorrect masks/outputs and visible artifacts in saved videos. This prevents reliable E2E testing on live webcam input in WSL.

---

## Environment & architecture (what we have)

- Host: Windows (camera physically attached to Windows). WSL2 is used for the development environment and for running the ML pipeline.
- Camera bridging: `usbipd` on Windows attaching the camera into WSL2 as `/dev/video0` (v4l2 device). This is the key path we suspect.
- Virtual environment: project `.venv` inside the workspace (Python 3.12). Packages used include PyTorch (CUDA-enabled), Ultralytics YOLOv8 (yolov8n-seg), SimpleLaMa for inpainting, OpenCV (`cv2`), numpy and ffmpeg.
- GPU: RTX 4080 accessible to WSL2 (CUDA-enabled PyTorch runs successfully in tests).
- Video capture stack used in experiments:
  - ffmpeg v4l2 capture: `ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -i /dev/video0 -frames:v 1 ...`
  - alternative attempted input formats: `yuyv422`, `mjpeg`, and `1280x480` packing variations
  - libv4l wrapper attempted via `LD_PRELOAD=/usr/lib/.../v4l2convert.so` to force conversion
  - PyAV attempted as alternative capture library (not installed by default in `.venv` during early tests)

---

## Observed evidence

- ffmpeg capture logs repeatedly showed messages indicating truncated MJPEG data, for example:
  - `[mjpeg] EOI missing, emulating` and `[mjpeg] overread` warnings while capturing frames.
- Per-row pixel inspection on captured frames (using numpy + cv2) shows the corrupted rows are uniform pure green (B=0, G≈134, R=0) with `unique_colors == 1` for those rows.
- The green band typically spans a contiguous block of rows near the bottom (examples observed: rows 352–479 or ~376–479; a common count is 104 or 128 rows depending on capture).
- ffprobe/ffmpeg inspection of saved videos indicates codec `mjpeg` and pixel formats such as `yuvj420p`, and streams captured at sizes `640x480` and `1280x480` (note: 1280x480 is unusual packing).

---

## What we tried (actions & tests)

1. Baseline ffmpeg single-frame captures
   - Commands used (examples):
     ```bash
     ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -i /dev/video0 -frames:v 1 -y test_capture.png
     ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x480 -i /dev/video0 -frames:v 1 -y test_capture_1280x480.png
     ffmpeg -f v4l2 -input_format yuyv422 -video_size 640x480 -i /dev/video0 -vframes 1 -f image2 -vcodec png -y test_capture_yuyv.png
     ```
   - Result: captures saved but often exhibited green-band corruption. ffmpeg output contained MJPEG decode warnings.

2. libv4l wrapper
   - Attempted `LD_PRELOAD=/usr/lib/.../v4l2convert.so ffmpeg ...` to force conversion by libv4l.
   - Result: inconsistent; sometimes produced frames without green band but was unreliable and produced other errors (buffer/unavailable dequeue in some runs).

3. Per-row pixel diagnostics
   - Wrote Python diagnostics to compute per-row means and unique color counts; consistently found rows near bottom that are pure green.
   - Confirmed corruption exists in the raw capture image (before YOLO or LaMa are run).

4. Implemented pragmatic repair
   - `capture_utils.py` added with functions: `run_ffmpeg_capture`, `detect_green_band`, `repair_green_band`.
   - Repair strategy: detect first corrupted row `start_row` and replace corrupted rows by copying the last good row above the band. This is a pragmatic hack to allow the downstream pipeline to run.
   - `test_capture_pipeline.py` runs 10 frames and saves repaired frames to `pipeline_test_out/`. All frames showed detection of the band and were repaired.

5. PyAV-based capture module (fallback)
   - Added `capture_av.py` which attempts to use PyAV to decode frames from `/dev/video0` and falls back to the ffmpeg multi-strategy repair (`get_frame_trial`) if PyAV isn't available or fails.
   - Ran `test_av_capture.py`. In our environment PyAV was not installed, so the code used the ffmpeg fallback and saved a repaired frame to `av_test_out/` (meta: `has_band=True, start_row=376, bad_count=104`).

6. Background monitor and RTSP consumer
   - `background_capture.py` added to capture periodic frames and log when corruption occurs; logs saved to `background_capture.log` and frames saved to `bg_frames/` for long-term monitoring.
   - `rtsp_consumer.py` added as a helper to consume camera streams if the camera is streamed from the Windows host (recommended fallback).

7. Notebook integration and tests
   - `MVp_clean.ipynb` ported to WSL and modified to:
     - Use ffmpeg single-frame capture function `capture_frame_with_ffmpeg()` (initially) which produced the same corrupted frames.
     - YOLO set to detect class 0 (people) and LaMa integrated for inpainting.
     - VideoWriter outputs saved: `Original_Webcam.avi`, `SegYolov8_InptSimLama.avi`, `Side_by_Side_Comparison.avi`.

---

## Files added / modified during diagnosis (repo)

- `MVp_clean.ipynb` — notebook with ffmpeg capture + YOLO + SimpleLaMa pipeline.
- `capture_utils.py` — multi-strategy ffmpeg capture utilities, detect/repair functions and `get_frame_trial()`.
- `test_capture_pipeline.py` — test runner that captures and repairs a small number of frames and saves results to `pipeline_test_out/`.
- `capture_av.py` — PyAV-based capture with fallback to capture_utils.
- `test_av_capture.py` — small runner that tries PyAV then fallback and writes a PNG to `av_test_out/`.
- `background_capture.py` — background monitor that periodically captures frames and logs corruption events to `background_capture.log`.
- `rtsp_consumer.py` — helper to consume RTSP/HTTP streams if the camera is streamed from the Windows host.
- `CAMERA_MJPEG_ISSUE.md` (this file) — detailed issue write-up and plan.

---

## Why this likely happens (technical explanation)

- The camera is physically connected to Windows. `usbipd` attaches the USB device to WSL2 and exposes it as `/dev/video0`.
- The MJPEG stream from certain cameras is packetized and requires complete JPEG EOI (End Of Image) markers. If USB transfer is truncated or reordered over usbip/driver boundary, ffmpeg's MJPEG decoder may see incomplete JPEG frames and attempt to emulate EOI or overread data, causing corrupted output.
- The camera advertises an unusual packed mode `1280x480` for MJPEG; this packing may be a hardware-specific stride/packing arrangement that interacts poorly with decoders expecting discrete 640x480 slices. A decoder or v4l2 driver expecting different strides can mis-decode rows at the end of the frame manifesting as uniform lines or color bands.
- Libv4l (`v4l2convert.so`) can sometimes fix pixel format/stride mismatches by reformatting frames, but its behavior depends on kernel, driver, and library versions. It worked intermittently for us but is not reliable.
- Because the corruption occurs at capture/decoder time, any downstream processing (YOLO/LaMa) receives corrupted inputs and cannot fully correct them.

---

## Reproduction steps (quick)

1. Ensure the camera is attached to the Windows host and shared into WSL with `usbipd attach --wsl --busid <BUSID>`.
2. In WSL run (example):

```bash
# single-frame ffmpeg capture
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -i /dev/video0 -frames:v 1 -y test_capture.png

# inspect with ffprobe
ffprobe -v error -show_entries stream=codec_name,width,height,pix_fmt -of default=noprint_wrappers=1 test_capture.png
```

3. Use `test_capture_pipeline.py` to run 10 trial captures and inspect files in `pipeline_test_out/`.

4. Run the notebook `MVp_clean.ipynb` cell that uses `capture_frame_with_ffmpeg()` to see green-band frames saved to `Original_Webcam.avi`.

---

## Short-term mitigations implemented

- Multi-strategy capture: try multiple ffmpeg input formats and sizes (`640x480 mjpeg`, `1280x480 mjpeg`, `yuyv422`) to find the least-broken capture for a particular run.
- Pragma repair: detect contiguous corrupted rows and copy the last good row into the corrupted region so the ML pipeline can continue. This produces visually plausible but synthetic content in the corrupted region and allows automated tests to proceed.
- PyAV fallback: try PyAV first (if installed) because in some environments its decoder/containers can handle marginal JPEGs differently.
- Windows-streaming alternate: code and instructions prepared to stream the camera from Windows (OBS/ffmpeg) to WSL via RTSP/HTTP and consume the stream in WSL — this bypasses usbipd and is the recommended robust fallback.

---

## Recommended next steps (ordered)

1. Short-term: stream the camera from Windows to WSL and consume the RTSP/HTTP stream in WSL.
   - Rationale: removes `usbipd` and driver bridging as a failure surface and is the fastest route to stable frames.
   - Action: Run an ffmpeg command (or OBS) on Windows to stream; consume with OpenCV/PyAV/ffmpeg in WSL. I can add the exact Windows ffmpeg commands and WSL consumer code if desired.
      - Action: Run an ffmpeg command (or OBS) on Windows to stream; consume with OpenCV/PyAV/ffmpeg in WSL. See the new "Windows streaming" section below for copy-paste commands and a ready-to-run WSL consumer script.

2. Try installing PyAV in `.venv` and re-run `test_av_capture.py`.
   - Rationale: PyAV/libav may decode truncated JPEG frames more gracefully than the current ffmpeg invocation in WSL.
   - Note: PyAV requires system development libs (apt install libavformat-dev etc.) — I can run and test this if you permit sudo.

3. Try `imageio-ffmpeg` pipe-based capture or `opencv` with GStreamer backend in WSL.
   - Rationale: different capture paths reassemble frames differently and sometimes avoid v4l2 MJPEG truncation issues.

4. If you must run on WSL and none of the above help: continue to harden the repair (for example, inpaint the corrupted region with a learned method rather than copy-last-row) and monitor stability with `background_capture.py`.

5. Avoid VM GPU passthrough unless you already have a hypervisor + hardware suited to PCIe passthrough. It is possible but complex; migrating the notebook to native Windows is a simpler and reliable approach if you prefer Windows.

---

## Commands & snippets used during diagnosis (copyable)

FFmpeg single-frame capture (mjpeg):

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -i /dev/video0 -frames:v 1 -y test_capture.png
```

FFmpeg single-frame capture (try yuyv422 instead):

```bash
ffmpeg -f v4l2 -input_format yuyv422 -video_size 640x480 -i /dev/video0 -vframes 1 -f image2 -vcodec png -y test_capture_yuyv.png
```

Libv4l wrapper (example, path may differ):

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libv4l/v4l2convert.so ffmpeg -f v4l2 -input_format mjpeg -video_size 640x480 -i /dev/video0 -frames:v 1 -y test_capture_v4l.png
```

Run the pipeline test harness:

```bash
. .venv/bin/activate
python3 test_capture_pipeline.py
```

Run PyAV fallback test (after installing PyAV or with fallback already present):

```bash
python3 test_av_capture.py
```

---

## Windows streaming (exact commands and WSL consumer)

Below are copyable examples to run on the Windows host (PowerShell) to stream the physical camera into the network so WSL can consume it reliably. Replace YOUR_CAMERA_NAME with the name reported by `ffmpeg -list_devices true -f dshow -i dummy` or the device name in Device Manager.

RTSP (recommended for robust consumption):

PowerShell / Windows (ffmpeg):

```powershell
# Stream camera to rtsp on port 8554
ffmpeg -f dshow -i video="YOUR_CAMERA_NAME" -vcodec libx264 -preset veryfast -tune zerolatency -f rtsp rtsp://0.0.0.0:8554/live.sdp
```

HTTP MJPEG (simpler, lower-latency for single clients):

```powershell
ffmpeg -f dshow -i video="YOUR_CAMERA_NAME" -vcodec mjpeg -q:v 5 -f mpegts http://0.0.0.0:8090
```

OBS Studio (GUI):
- Add the camera as a video source, then use "Start Virtual Camera" or configure a custom streaming output (use ffmpeg/RTSP if you want to expose an RTSP URL).

Notes:
- Using `0.0.0.0` binds to all interfaces on the Windows host; from WSL use the Windows host IP (usually the host gateway for WSL) or `localhost` depending on how your network is configured. If using `localhost`/127.0.0.1 from WSL doesn't connect, use the Windows host IP (find with `ipconfig` on Windows) or try `hostname.local`.

WSL consumer (ready-to-run)

I added `wsl_rtsp_consumer.py` to the repository. It tries OpenCV first, then PyAV, and saves incoming frames to `rtsp_frames/` for inspection. It also runs the green-band detector from `capture_utils.py` (if present) and logs detections to `rtsp_consumer.log`.

Usage in WSL (example):

```bash
. .venv/bin/activate
python3 wsl_rtsp_consumer.py --url rtsp://<WINDOWS_HOST_IP>:8554/live.sdp --max-frames 100
```

Example HTTP MJPEG usage:

```bash
python3 wsl_rtsp_consumer.py --url http://<WINDOWS_HOST_IP>:8090 --max-frames 100
```

If you start the Windows ffmpeg command first and then run the WSL consumer above, `wsl_rtsp_consumer.py` will save frames and print whether a green band was detected.


---

## Where to look next / artifacts produced

- `pipeline_test_out/` — repaired frames from `test_capture_pipeline.py`.
- `av_test_out/` — frame saved by `test_av_capture.py`.
- `bg_frames/` and `background_capture.log` — logs and captures from the background monitor.
- `Original_Webcam.avi`, `SegYolov8_InptSimLama.avi`, `Side_by_Side_Comparison.avi` — videos produced by the notebook when it runs.
- `capture_utils.py`, `capture_av.py`, `test_capture_pipeline.py`, `test_av_capture.py`, `background_capture.py`, `rtsp_consumer.py` — helper scripts added during diagnosis.

---

## Final notes

- The corruption is reproducible and consistent with MJPEG truncation or stride/packing mismatches. The most reliable workaround is to avoid the usbipd+v4l2 capture path and either run the pipeline natively on Windows or stream the camera from Windows into WSL using RTSP/HTTP and consume that clean stream inside WSL.
- I'm available to (pick one):
  - A) prepare the Windows ffmpeg/OBS commands and the WSL consumer code and verify the stream in WSL; or
  - B) install PyAV and/or imageio-ffmpeg into `.venv` and re-run tests here (I will need sudo for system libs to build PyAV); or
  - C) migrate the notebook to run natively on Windows (I can produce a `requirements.txt`/conda env and a short migration checklist).

Please tell me which option you prefer and I will implement and test it next.
