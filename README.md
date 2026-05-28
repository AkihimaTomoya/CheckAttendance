# Face Recognition Realtime Server

This project provides a FastAPI-based server for **real-time face detection and recognition** with webcam streaming and a modern web UI.

---

## Environment Setup

- Python version: **3.8.13**

Create and activate the environment with **conda**:

```bash
conda create -n face_recog python=3.8.13 -y
conda activate face_recog

python.exe -m pip install --upgrade pip 
pip install -r requirements.txt
```

---

## Prepare Facebank

Before running inference, prepare your **facebank** (the database of known identities). You need at least **12 face images** for each identity to cover different angles and expressions.

### Option 1: Using the Capture Script (Recommended)
You can use the built-in `capture_facebank.py` script to capture faces directly from your webcam. It will guide you through capturing **6 different poses** (2 images per pose, making a total of 12 images):
- Frontal face (Natural)
- Frontal face (Smiling)
- Turn face LEFT
- Turn face RIGHT
- Look UP
- Look DOWN

To run the capture script:
```bash
python capture_facebank.py -n Alice
```

If you already have a folder of raw images for a person, you can also use the script to detect, crop, and align them:
```bash
python capture_facebank.py -n Alice --input /path/to/raw/images
```

### Option 2: Manual Preparation
1. Create a folder named `facebank/`.
2. Inside it, create subfolders named after each person you want to recognize.
3. Place at least **12 aligned/cropped face images** of each person inside their folder. The images should have a size of `112x112` pixels.

**Example structure:**
```
facebank/
├── Alice/                      # One folder per person
│   ├── Alice_0001.jpg
│   ├── Alice_0002.jpg
│   ├── ...
│   └── Alice_0012.jpg          # Minimum 12 images per person
└── Bob/
    ├── Bob_0001.jpg
    ├── Bob_0002.jpg
    ├── ...
    └── Bob_0012.jpg
```

- Each subfolder name is treated as the person’s identity.  
- The system will automatically build embeddings (face features) from these images to use during recognition.

---

## Train a Model (Optional)
You can train a face recognition model using the provided training scripts. If you already have a trained model, you can skip this step.
To train a model, run:

```bash
python train_v2.py -c configs/res50_custom_onegpu.py
```

This will save the trained model in the `work_dirs/res50_custom_onegpu/` directory.

---

## Download Pretrained Model (Optional)
If you don't want to train a model from scratch, you can download pretrained models.
```bash
pip install huggingface_hub
```
Then, use the following Python script to download the model:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Akihima/EnchancedArcFace",
    local_dir="work_dirs",
    allow_patterns=["**/*"],
    local_dir_use_symlinks=False
)
print("Download complete. Check the 'work_dirs' directory.")
```
**Struceture:**
```
work_dirs/
├── ms1mv3_r50_onegpu/          
│   └── model.pt
├── res50_custom2_onegpu/          
│   └── model.pt
├── res50_custom_onegpu/          
│   └── model.pt
├── res50_ffm_onegpu/          
│   └── model.pt 
└── ...
```
---

## Run Realtime Inference

To start the FastAPI server with your trained model, run:

```bash
python fastapi-app-demo.py -d work_dirs/res50_ffm_onegpu
```

- The server starts on: `http://0.0.0.0:5050` (default).  
- Open your browser at: `http://localhost:5050` to access the login page.

### 🔐 Authentication (`cfg/users.json`)
The application uses local authentication. If `cfg/users.json` does not exist, it will be automatically created with the following default accounts:
- **Admin**:
  - **Username**: `admin`
  - **Password**: `admin123`
- **Student**:
  - **Username**: `SV001`
  - **Password**: `sv001` (corresponding to face identity list in `users.json` file)

---

## Web Interface UI Description

### 1. Login Page (`/`)
- A modern login card where users enter their username and password.
- Based on the role specified in `cfg/users.json`, the user is redirected to either the **Student** page or the **Admin** dashboard.

### 2. Student Interface (`/student`)
- **Top Navigation Bar**: Displays the logged-in student's display name and student ID.
- **Attendance Bar**: Shows the current status of the student:
  - `⏸ Camera đang tắt` (Camera is off)
  - `🟢 Đang trên hình` (Present - student's face is detected)
  - `🔴 Vắng mặt` (Absent - student's face is not detected)
  - Real-time study duration and accumulated absence duration.
- **Webcam & Overlay Canvas**:
  - Click **"Bật Camera"** (Turn on Camera) to start streaming frames to the server via WebSockets (sends frames every 150ms).
  - Draws a bounding box over detected faces on a canvas overlay:
    - **Green Box (`#10b981`)**: Face recognized as the student's registered `face_name` (matching database).
    - **Orange/Yellow Box (`#f59e0b`)**: Another person detected (`Sai người`).
    - **Red Box (`#ef4444`)**: Unknown face (`Unknown`).
- **Live Logging**:
  - If a face disappears for more than 5 seconds, it triggers a `face_lost` event.
  - All check-in, check-out, camera toggle, and presence state changes are logged as a CSV file in `logs/attendance_YYYY-MM-DD.csv` via backend API requests.

### 3. Admin Dashboard (`/admin`)
- **Real-time Attendance Table**:
  - Lists all registered student identities.
  - Shows live status badges (`Có mặt`, `Vắng mặt`, `Đã rời lớp`, `Chưa vào`), check-in time, and total absence time.
- **Add Student Form ("Thêm SV Mới")**:
  - Admins can register new student accounts directly by typing a Student ID, Full Name, Facebank ID, and Password. These details are written instantly into `cfg/users.json`.
- **Facebank Management ("Tải lại Facebank")**:
  - After preparing new images in the `facebank/` folder, the Admin can click **"Tải lại Facebank"** to rebuild face embeddings at runtime without restarting the server.

---

## Notes

- Ensure your GPU (if available) is accessible via PyTorch for better performance.
- When running a new model or updating images in `facebank/`, click **"Tải lại Facebank"** in the Admin dashboard to recalculate the face embeddings.
