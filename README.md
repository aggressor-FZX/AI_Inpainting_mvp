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