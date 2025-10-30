# AI Inpainting MVP: How to Use

This guide explains how to set up and run the AI Inpainting MVP, a real-time video processing pipeline that removes people from camera feeds using AI models. It streams video from Windows FFmpeg to a WSL Python server for processing.

## Tech Stack

- **YOLOv8 (ultralytics)**: For real-time person segmentation and detection.
- **Simple LaMa Inpainting**: AI-based background restoration to fill in removed people.
- **OpenCV**: Video capture, processing, display, and saving.
- **FFmpeg**: Command-line tool for camera streaming on Windows.
- **NumPy**: Image array manipulations.
- **Python Socket Library**: TCP streaming between Windows and WSL.
- **PyTorch/TorchVision**: Backend for AI models (CUDA-enabled for GPU acceleration).

## Prerequisites

- **Windows**: FFmpeg installed (download from ffmpeg.org).
- **WSL2 (Ubuntu)**: Python 3.8+, virtual environment.
- **Hardware**: NVIDIA GPU with CUDA support (for optimal performance).
- **Network**: WSL Ethernet bridge for TCP connectivity.

## Step-by-Step Setup

1. **Clone or Download the Repo**:
   ```bash
   git clone https://github.com/aggressor-FZX/AI_Inpainting_mvp.git
   cd AI_Inpainting_mvp
   ```

2. **Install Dependencies in WSL**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Find Your WSL IP**:
   In WSL terminal:
   ```bash
   ip addr show eth0 | grep inet
   ```
   Note the IP (e.g., 172.24.146.232).

## How to Run

### Option 1: Terminal Script (Recommended for Production)

1. In WSL, run the server:
   ```bash
   source .venv/bin/activate
   python3 listen_raw_server.py --no-save  # Live-only, no saving
   # or python3 listen_raw_server.py        # Saves 200 frames
   ```

2. On Windows, open PowerShell and run FFmpeg (replace paths and IP):
   ```powershell
   & "C:\path\to\ffmpeg.exe" -f dshow -video_size 1280x720 -framerate 30 -i video="Your Camera Name" -pix_fmt bgr24 -f rawvideo tcp://172.24.146.232:5000
   ```
   - Find camera name with `ffmpeg -list_devices -f dshow -i dummy`.
   - Start FFmpeg within 30 seconds of the Python script.

The script will display a live video window with FPS overlay and process frames in real-time.

### Option 2: Jupyter Notebook (For Development/Testing)

1. In WSL, start Jupyter:
   ```bash
   source .venv/bin/activate
   jupyter notebook
   ```

2. Open `MVp_clean.ipynb`.

3. **Run Cell 1 (Server Setup)**: This starts the TCP server in WSL to receive raw video from Windows.

4. **On Windows, Run FFmpeg**: As above, stream to the WSL IP:5000. Do this within 30 seconds of running the server cell.

5. **Run Cell 2 (Processing Loop)**: The notebook will receive frames, apply YOLO segmentation to detect people, use LaMa to inpaint (remove) them, and display a side-by-side comparison (original left, inpainted right) with FPS and frame count overlay.

6. **Optional Cells**: Run additional cells to save videos or create comparison clips.

The notebook allows interactive debugging and visualization, while the script is for headless operation.

## TCP data format & Windows → WSL flow

This project uses a very small custom pipeline: FFmpeg on Windows captures raw frames from a camera and streams them as raw BGR24 frames over a TCP socket to a Python server running in WSL. The server reads fixed-size frames (W * H * 3 bytes) and processes each frame with YOLO + LaMa.

- Data format: raw BGR24 (8-bit per channel), row-major (height, width, channels). Each frame is exactly W * H * 3 bytes. In the notebook and script W=1280, H=720 by default, so BYTES = 1280 * 720 * 3 = 2,764,800 bytes per frame.
- Frame layout: each pixel is three bytes in BGR order (not RGB). The code reconstructs the frame with:

   frame = np.frombuffer(buf, np.uint8).reshape((H, W, 3))

- Socket behavior: the server uses blocking reads to collect BYTES bytes per frame. The client (FFmpeg) streams rawvideo to tcp://WSL_IP:5000. The server should be started before FFmpeg.

Connection sequence (recommended):

1. Start the WSL server (`listen_raw_server.py` or run the server cell in `MVp_clean.ipynb`). The server listens on 0.0.0.0:5000 and waits (30s timeout in the notebook example).
2. Start FFmpeg on Windows and stream rawvideo to tcp://<WSL_IP>:5000 using `-pix_fmt bgr24 -f rawvideo`.
3. The server accepts the connection, reads BYTES per frame in a loop, and processes each frame.

Important implementation notes:

- The server uses a loop that calls `recv_into` until the full BYTES are read. Partial reads are expected with TCP; code must accumulate into a preallocated buffer (bytearray/memoryview) until the whole frame arrives.
- If `recv` returns 0 bytes, the connection was closed by the client — stop the loop and/or reconnect.
- The client must send frames at the expected resolution and pixel format. Mismatched shape will raise reshape errors.

Visual illustration (Windows ↔ WSL, same physical machine):

      [Windows: Camera] --(FFmpeg rawvideo bgr24)--> [Windows FFmpeg] --TCP--> [WSL IP:5000] -> [Python server in WSL]

ASCII flow (more detailed):

      +----------------------+       TCP(rawvideo bgr24)       +--------------------------+
      | Windows (host)      |  ----------------------------->  | WSL2 (Ubuntu)           |
      |                      |                                   |                          |
      | Camera device        | --capture--> FFmpeg --socket-->  | listen_raw_server.py     |
      | (dshow / v4l2)       |                                   | - recv_into() loop       |
      |                      |                                   | - reconstruct frame      |
      +----------------------+                                   | - YOLO segmentation      |
                                                                                              | - LaMa inpainting        |
                                                                                              | - display / save video   |
                                                                                              +--------------------------+

Expected speed and latency (practical notes):

- The data transport itself (TCP over the loopback/Ethernet bridge between Windows and WSL) is typically not the bottleneck on the same machine: the transfer of ~2.76 MB per 1280x720 frame is easily handled by modern NICs and WSL bridge. The main limits are serialization (FFmpeg encoding to rawvideo), kernel buffering, and the Python processing pipeline.
- In practice with the full pipeline (YOLOv8 segmentation + LaMa inpainting) we measured roughly ~5 FPS at 1280x720 on an RTX 40-series GPU running in WSL2. That implies ~200 ms per frame of processing (inference + inpainting + display). TCP transport adds only a few to tens of milliseconds on the same host.
- To improve effective throughput consider:
   - Lowering resolution to 640x480 (reduces frame size by ~4x) and will likely increase FPS significantly.
   - Reducing model size (e.g., a smaller YOLO variant) or offloading post-processing to a faster implementation.
   - Running the LaMa model on a separate worker or batching inpainting calls where possible.

When to expect transport limits:

- If you use high resolutions (4K) the raw frame size becomes large and TCP buffer/latency and FFmpeg throughput can become relevant.
- If you stream across a slow network (not local machine), transport will add latency and potentially packetization stalls.

Safety tips / debugging checklist:

- If you see reshape errors, confirm W and H match the FFmpeg command and the server code.
- If the server times out waiting for a connection, check the WSL IP and firewall; start the server first.
- Use a small test client (or `nc`) to confirm the server accepts connections and receives bytes before running FFmpeg.

## Hardware Details and Performance

Hardware specs are provided for comparing model speeds across different GPUs/CPUs. The pipeline is GPU-intensive due to YOLO and LaMa inference.

- **NVIDIA GPU**: RTX 40-series with CUDA 13.0 (e.g., RTX 4070/4080).
- **CPU**: Intel/AMD with 16GB+ RAM.
- **Network**: WSL2 Ethernet for low-latency TCP (avoids WSL localhost issues).

Note: the quoted performance numbers below are measured on a live feed at 1280x720 (720p). Recorded files, different resolutions, or different model choices will produce different results.

**Performance Benchmarks** (live feed at 1280x720):
- **FPS**: ~5 on RTX 40-series (tested); 3-5 on RTX 30-series; <3 on CPU-only.
- **Latency**: 200-300ms end-to-end (capture → processing → display).
- **VRAM Usage**: 2-4GB.
- **CPU Load**: Minimal (GPU-bound).

Use these to estimate performance on your hardware. Lower resolutions (e.g., 640x480) can achieve higher FPS; higher resolutions will reduce FPS.

Pixel format and color order (BGR vs RGB)

- The pipeline streams raw BGR24 frames (`-pix_fmt bgr24`) from FFmpeg. BGR byte order is used by OpenCV by default, so the notebook and script reconstruct incoming frames as BGR with:

   frame = np.frombuffer(buf, np.uint8).reshape((H, W, 3))

- Some libraries or model outputs may return images in RGB order. The notebook contains a conversion step after LaMa inpainting to ensure the image is converted to BGR before display or writing (example):

   if len(inpainted_frame.shape) == 3 and inpainted_frame.shape[2] == 3:
         inpainted_frame = cv2.cvtColor(inpainted_frame, cv2.COLOR_RGB2BGR)

- Always ensure the client FFmpeg command (`-pix_fmt`) and server reshape/order agree — otherwise colors will appear swapped (blue↔red).

## Troubleshooting

- **No Video in WSL**: Ensure FFmpeg is streaming to the correct WSL IP/port. Check firewall.
- **Low FPS**: Use a better GPU or lower resolution.
- **Connection Refused**: Start Python server before FFmpeg.
- **Model Errors**: Ensure CUDA is installed and GPU is detected (`nvidia-smi`).

## Files Overview

- `listen_raw_server.py`: Standalone script.
- `MVp_clean.ipynb`: Notebook with interactive cells.
- `yolov8n-seg.pt`: YOLO model file.
- `requirements.txt`: Dependencies.