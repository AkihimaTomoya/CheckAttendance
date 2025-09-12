# Face Recognition Realtime Server

This project provides a FastAPI-based server for **real-time face detection and recognition** with webcam streaming and a modern web UI.

---

## Environment Setup

- Python version: **3.10.18**

Create and activate the environment with **conda**:

```bash
conda create -n face_recog python=3.10.18 -y
conda activate face_recog

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Prepare Facebank

Before running inference, prepare your **facebank** (the database of known identities).

1. Create a folder named `facebank/`.
2. Inside it, create subfolders for each person you want to recognize.
3. Place several face images of each person inside their folder.

**Example structure:**
```
facebank/
├── _nonffm/          
│   ├── Alice/        # one folder per person
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── Bob/
│       ├── 1.jpg
│       └── 2.jpg
├── _ffm/          
│   ├── Alice/        # one folder per person
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── Bob/
│       ├── 1.jpg
│       └── 2.jpg
└── ...
```

- Each subfolder name is treated as the person’s identity.  
- The system will automatically build embeddings (face features) from these images to use during recognition.

---

## Run Realtime Inference

To start the FastAPI server with your trained model, run:

```bash
python fastapi-app.py -d work_dirs/res50_ffm_onegpu
```

- The server starts on: `http://0.0.0.0:5050` (default).  
- Open your browser at: `http://localhost:5050` to access the UI.  
- The UI will automatically:
  - Start your webcam
  - Draw bounding boxes around detected faces
  - Display recognition results (identity, distance, threshold, etc.) in real time

---

## Notes

- Ensure your GPU (if available) is accessible via PyTorch for better performance.
- Thresholds and runtime options (e.g., **TTA**, **Show Top-1**) can be adjusted live from the web UI.
- To rebuild or refresh the facebank at runtime, use the **Reload Facebank** button in the interface.
- Debug information (server state, facebank details) can also be viewed directly in the UI.
