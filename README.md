# AI Inpainting MVP

Real-time camera processing with YOLO person detection and LaMa inpainting.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download YOLO model (included in repo):
   - `yolov8n-seg.pt` is already included.

## Usage

### Terminal Script (Recommended)
Run the standalone server:
```bash
python3 listen_raw_server.py --no-save  # Live-only mode
# or
python3 listen_raw_server.py            # Save 200 frames
```

### Notebook
Open `MVp_clean.ipynb` in Jupyter and run the cells.

## Windows FFmpeg Setup
Install FFmpeg on Windows and run:
```powershell
& "C:\path\to\ffmpeg.exe" -f dshow -video_size 1280x720 -framerate 30 -i video="Your Camera" -pix_fmt bgr24 -f rawvideo tcp://YOUR_WSL_IP:5000
```

Replace `YOUR_WSL_IP` with your WSL IP (e.g., 172.24.146.232).

## Operation

### Notebook Operation
The notebook (`MVp_clean.ipynb`) contains cells for:
- Setting up the raw video server in WSL
- Receiving frames from Windows FFmpeg over TCP
- Processing each frame with YOLOv8 for person segmentation
- Applying LaMa inpainting to remove detected people
- Displaying a real-time side-by-side comparison (original vs. inpainted)
- Optionally saving videos and creating comparison clips
- Monitoring FPS and frame count in real-time

Run the server cell first, then start FFmpeg on Windows within 30 seconds.

### Script Operation
The script (`listen_raw_server.py`) is a standalone version of the notebook's server logic.
- Listens on port 5000 for raw BGR video stream
- Processes frames continuously or up to 200 frames if saving
- Outputs FPS and status to terminal
- Shows live video window with overlay text
- Saves videos if enabled

Use `--no-save` for continuous live processing without disk I/O.

## Hardware Details
- **NVIDIA GPU**: RTX 40-series with CUDA 13.0 support
- **CPU**: Intel/AMD with sufficient RAM for processing
- **Network**: WSL2 with Ethernet bridge for low-latency TCP streaming

## Performance Results
- **Resolution**: 1280x720 (720p)
- **FPS**: 10-15 FPS on NVIDIA RTX 40-series GPU
- **Latency**: ~100-200ms end-to-end (capture to display)
- **CPU Usage**: Low (mostly GPU-bound)
- **Memory**: ~2-4GB GPU VRAM usage

Performance scales with GPU power; lower-end GPUs may achieve 5-10 FPS.

## Features
- Real-time YOLO person segmentation
- LaMa AI inpainting to remove people
- Side-by-side live display
- Optional video saving
- FPS monitoring

## Requirements
See `requirements.txt` for full list. Key packages:
- ultralytics
- simple-lama-inpainting
- opencv-python
- numpy