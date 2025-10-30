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

## Hardware Details and Performance

Hardware specs are provided for comparing model speeds across different GPUs/CPUs. The pipeline is GPU-intensive due to YOLO and LaMa inference.

- **NVIDIA GPU**: RTX 40-series with CUDA 13.0 (e.g., RTX 4070/4080).
- **CPU**: Intel/AMD with 16GB+ RAM.
- **Network**: WSL2 Ethernet for low-latency TCP (avoids WSL localhost issues).

**Performance Benchmarks** (at 1280x720):
- **FPS**: ~5 on RTX 40-series (tested); 3-5 on RTX 30-series; <3 on CPU-only.
- **Latency**: 200-300ms end-to-end.
- **VRAM Usage**: 2-4GB.
- **CPU Load**: Minimal (GPU-bound).

Use these to estimate performance on your hardware. Lower resolutions (e.g., 640x480) can achieve higher FPS.

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