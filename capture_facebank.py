"""
capture_facebank.py (Updated: Minimalist UI & Centered Poses)

Capture faces using the project's `FaceDetector` (from `face_detector.py`) and save
aligned 112x112 RGB JPEGs into `facebank/<PersonName>/`.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import unicodedata
import sys
import numpy as np

# Cấu hình encoding utf-8 cho stdout/stderr để tránh lỗi UnicodeEncodeError trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Lazy imports
try:
    import cv2
except Exception:
    cv2 = None

try:
    from face_detector import FaceDetector
except Exception as e:
    FaceDetector = None
    _FD_IMPORT_ERROR = e
else:
    _FD_IMPORT_ERROR = None

FACEBANK_ROOT = Path("facebank")
CANON_SIZE = (112, 112)
IMG_EXT = ".jpg"

# Cấu hình số ảnh cho mỗi góc độ
PHOTOS_PER_POSE = 2

POSES = [
    "Nhìn thẳng (Tự nhiên)",
    "Nhìn thẳng (Mỉm cười)",
    "Nghiêng mặt sang TRÁI",
    "Nghiêng mặt sang PHẢI",
    "Ngước mặt LÊN",
    "Cúi mặt XUỐNG"
]

def _ensure_person_dir(name: str) -> Path:
    person_dir = FACEBANK_ROOT / name
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir

def _next_index_for_name(person_dir: Path, name: str) -> int:
    max_idx = 0
    for p in person_dir.iterdir():
        if not p.is_file() or not p.stem.startswith(f"{name}_"):
            continue
        suffix = p.stem[len(name) + 1 :]
        if suffix.isdigit():
            try:
                idx = int(suffix)
                if idx > max_idx: max_idx = idx
            except: continue
    return max_idx + 1

def _make_filename(name: str, idx: int) -> str:
    return f"{name}_{idx:04d}{IMG_EXT}"

def save_pil_face(face: Image.Image, person_dir: Path, name: str) -> Optional[Path]:
    person_dir.mkdir(parents=True, exist_ok=True)
    next_idx = _next_index_for_name(person_dir, name)
    fname = _make_filename(name, next_idx)
    out_path = person_dir / fname
    try:
        if face.size != CANON_SIZE:
            face = face.resize(CANON_SIZE)
        face.save(out_path, format="JPEG")
        return out_path
    except Exception as e:
        print(f"Lỗi lưu file: {e}")
        return None

def get_pose_info(saved: int, photos_per_pose: int = 2) -> tuple[int, int, str]:
    pose_idx = saved // photos_per_pose
    sub_idx = (saved % photos_per_pose) + 1

    if pose_idx < len(POSES):
        return pose_idx, sub_idx, POSES[pose_idx]
    
    return -1, saved + 1, "Góc tự do"

def get_fonts() -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    font_paths = [
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    
    bold_path = None
    reg_path = None
    for path in font_paths:
        p = Path(path)
        if p.exists():
            if "bd" in p.name.lower() or "segoeuib" in p.name.lower():
                if not bold_path: bold_path = str(p)
            else:
                if not reg_path: reg_path = str(p)
                
    bold_path = bold_path or reg_path or "C:\\Windows\\Fonts\\arial.ttf"
    reg_path = reg_path or bold_path or "C:\\Windows\\Fonts\\arial.ttf"
    
    def load_f(path: str, size: int):
        try:
            return ImageFont.truetype(path, size)
        except IOError:
            return ImageFont.load_default()
            
    return {
        "title": load_f(bold_path, 16),
        "subtitle": load_f(bold_path, 14),
        "body_bold": load_f(bold_path, 12),
        "body": load_f(reg_path, 12),
        "small": load_f(reg_path, 10)
    }

def draw_text_safe(draw: ImageDraw.Draw, xy: tuple[float, float], text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, fill: tuple[int, int, int, int], anchor: Optional[str] = None, is_default: bool = False, **kwargs):
    if is_default:
        nfkd = unicodedata.normalize('NFKD', text)
        text = "".join([c for c in nfkd if not unicodedata.combining(c)])
        text = text.replace('đ', 'd').replace('Đ', 'D')
    draw.text(xy, text, fill=fill, font=font, anchor=anchor, **kwargs)

def draw_rounded_rect(draw: ImageDraw.Draw, xy: list[float], radius: int, fill: Optional[tuple[int, int, int, int]] = None, outline: Optional[tuple[int, int, int, int]] = None, width: int = 1):
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except AttributeError:
        draw.rectangle(xy, fill=fill, outline=outline, width=width)

def draw_ui_overlay(
    frame: np.ndarray,
    face_box: Optional[tuple[float, float, float, float]],
    face_aligned: bool,
    alignment_hint: str,
    saved: int,
    max_images: int,
    name: str,
    last_face_thumb: Optional[Image.Image],
    success_counter: int,
    fonts: dict,
    is_default_font: bool,
    photos_per_pose: int = 2
) -> np.ndarray:
    H, W = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # LUÔN LUÔN CĂN GIỮA MÀN HÌNH CHÍNH XÁC
    cx, cy = W // 2, H // 2
    rx = int(min(W, H) * 0.22)
    ry = int(rx * 1.35)
    
    pose_idx, sub_idx, current_pose_text = get_pose_info(saved, photos_per_pose)
    
    # 1. Floating Top Banner
    draw.rectangle([0, 0, W, 45], fill=(0, 0, 0, 160))
    pose_req_str = f"HƯỚNG DẪN: {current_pose_text.upper()}"
    if pose_idx != -1:
        pose_req_str += f" ({sub_idx}/{photos_per_pose})"
    else:
        pose_req_str += f" ({saved + 1}/{max_images})"
    draw_text_safe(draw, (cx, 22), pose_req_str, fonts["title"], (255, 255, 255, 255), anchor="mm", is_default=is_default_font)
    
    # 2. Bottom Banner
    draw.rectangle([0, H - 60, W, H], fill=(0, 0, 0, 160))
    if face_aligned:
        hint_color = (52, 199, 89, 255)  # Apple Green
    elif "Không tìm" in alignment_hint:
        hint_color = (255, 59, 48, 255)  # Apple Red
    else:
        hint_color = (255, 204, 0, 255)  # Yellow
    draw_text_safe(draw, (cx, H - 40), alignment_hint, fonts["subtitle"], hint_color, anchor="mm", is_default=is_default_font)
    draw_text_safe(draw, (cx, H - 18), "Ấn SPACE để chụp  |  R: Reset  |  Q: Thoát", fonts["small"], (180, 180, 180, 255), anchor="mm", is_default=is_default_font)
    
    # 3. Face Guide Oval (Minimalist)
    oval_bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    if face_aligned:
        oval_color = (52, 199, 89, 240)  # Solid Green
        width = 4
    elif face_box is not None:
        oval_color = (255, 255, 255, 200) # Translucent White
        width = 2
    else:
        oval_color = (255, 59, 48, 180)   # Red Warning
        width = 2
        
    draw.ellipse(oval_bbox, outline=oval_color, width=width)
    
    # Crosshair ticks (clean)
    tick = 12
    draw.line([(cx, cy - ry), (cx, cy - ry + tick)], fill=oval_color, width=width)
    draw.line([(cx, cy + ry - tick), (cx, cy + ry)], fill=oval_color, width=width)
    draw.line([(cx - rx, cy), (cx - rx + tick, cy)], fill=oval_color, width=width)
    draw.line([(cx + rx - tick, cy), (cx + rx, cy)], fill=oval_color, width=width)
    
    # 4. Floating Minimalist Checklist (Top Left)
    if W >= 500:
        box_w = 175
        box_h = 30 + len(POSES) * 22
        draw_rounded_rect(draw, [15, 60, 15 + box_w, 60 + box_h], radius=8, fill=(0, 0, 0, 140))
        draw_text_safe(draw, (25, 70), "DANH SÁCH GÓC:", fonts["body_bold"], (255, 255, 255, 255), is_default=is_default_font)
        
        for i in range(len(POSES)):
            y_pos = 92 + i * 22
            curr_pose_idx = pose_idx
            pose_completed = i < curr_pose_idx or curr_pose_idx == -1
            pose_active = i == curr_pose_idx
            
            if pose_completed:
                draw_text_safe(draw, (25, y_pos), "✔", fonts["body"], (52, 199, 89, 255), is_default=is_default_font)
                text_color = (150, 150, 150, 255)
            elif pose_active:
                draw.ellipse([27, y_pos+4, 33, y_pos+10], fill=(10, 132, 255, 255)) # Apple Blue
                text_color = (255, 255, 255, 255)
            else:
                draw.ellipse([27, y_pos+4, 33, y_pos+10], outline=(150, 150, 150, 255), width=1)
                text_color = (150, 150, 150, 255)
                
            draw_text_safe(draw, (45, y_pos), POSES[i], fonts["body"], text_color, is_default=is_default_font)
            
    # 5. Thumbnail Preview (Top Right)
    if last_face_thumb is not None:
        draw_rounded_rect(draw, [W - 95, 60, W - 15, 160], radius=6, fill=(0, 0, 0, 140))
        draw_text_safe(draw, (W - 55, 72), "ĐÃ CHỤP", fonts["small"], (255, 255, 255, 255), anchor="mm", is_default=is_default_font)
        try:
            thumb = last_face_thumb.resize((70, 70))
            pil_img.paste(thumb, (W - 90, 82))
            draw.rectangle([W - 91, 81, W - 19, 153], outline=(52, 199, 89, 255), width=2)
        except Exception:
            pass

    # Success Badge
    if success_counter > 0:
        alpha = int(min(255, success_counter * 20))
        draw_rounded_rect(draw, [cx - 80, cy - 20, cx + 80, cy + 20], radius=8, fill=(52, 199, 89, alpha))
        draw_text_safe(draw, (cx, cy), "THÀNH CÔNG!", fonts["body_bold"], (255, 255, 255, alpha), anchor="mm", is_default=is_default_font)
        
    pil_img = Image.alpha_composite(pil_img, overlay)
        
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def capture_from_camera(name: str, max_images: int = 12, camera_index: int = 0):
    if FaceDetector is None or cv2 is None:
        print("Lỗi: Thiếu thư viện hoặc FaceDetector không khả dụng.")
        return

    person_dir = _ensure_person_dir(name)
    detector = FaceDetector()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Không thể mở camera index {camera_index}")
        return

    fonts = get_fonts()
    is_default_font = not hasattr(fonts["title"], "path")

    saved = 0
    success_counter = 0
    last_face_thumb = None

    print("\n" + "="*60)
    print(" HƯỚNG DẪN THU THẬP KHUÔN MẶT FACEBANK ".center(60, "="))
    print(f"Mỗi góc độ cơ bản sẽ được chụp {PHOTOS_PER_POSE} ảnh.")
    print("Vui lòng làm theo hướng dẫn hiển thị trên giao diện camera.")
    print("Ấn SPACE (Dấu cách) để chụp. Ấn 'r' để reset bộ đếm. Ấn 'q' hoặc 'ESC' để thoát.")
    print("="*60 + "\n")

    while saved < max_images:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)  # Mirror mode
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        face = None
        face_box = None
        face_aligned = False
        alignment_hint = "Không tìm thấy khuôn mặt"

        try:
            boxes, _ = detector.detect_faces(pil)
            if len(boxes) > 0:
                face_box = boxes[0][:4]
                
                # CHÍNH GIỮA MÀN HÌNH
                cx = w // 2
                cy = h // 2
                rx = int(min(w, h) * 0.22)
                ry = int(rx * 1.35)
                
                fx1, fy1, fx2, fy2 = face_box
                fcx = (fx1 + fx2) / 2
                fcy = (fy1 + fy2) / 2
                fw = fx2 - fx1
                fh = fy2 - fy1
                
                dist_from_center = ((fcx - cx) ** 2 + (fcy - cy) ** 2) ** 0.5
                ideal_w = rx * 2
                ideal_h = ry * 2
                
                if fw < ideal_w * 0.55 or fh < ideal_h * 0.55:
                    alignment_hint = "Hãy di chuyển lại GẦN camera hơn"
                elif fw > ideal_w * 1.3 or fh > ideal_h * 1.3:
                    alignment_hint = "Hãy di chuyển XA camera hơn"
                elif dist_from_center > (rx * 0.4):
                    alignment_hint = "Vui lòng đưa mặt vào GIỮA khung elip"
                else:
                    face_aligned = True
                    alignment_hint = "Sẵn sàng! Nhấn SPACE để chụp"
                    
                nx1, ny1, nx2, ny2 = detector._expand_square((fx1, fy1, fx2, fy2), w, h, scale=1.30)
                face_np = frame[ny1:ny2, nx1:nx2]
                if face_np.size > 0:
                    face = Image.fromarray(cv2.cvtColor(face_np, cv2.COLOR_BGR2RGB)).resize(CANON_SIZE)
            else:
                face_box = None
                face = None
        except Exception:
            face_box = None
            face = None

        disp = draw_ui_overlay(
            frame=frame,
            face_box=face_box,
            face_aligned=face_aligned,
            alignment_hint=alignment_hint,
            saved=saved,
            max_images=max_images,
            name=name,
            last_face_thumb=last_face_thumb,
            success_counter=success_counter,
            fonts=fonts,
            is_default_font=is_default_font,
            photos_per_pose=PHOTOS_PER_POSE
        )

        if success_counter > 0:
            success_counter -= 1

        cv2.imshow("Capture Facebank", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            saved = 0
            last_face_thumb = None
            print("🔄 Đã reset tiến trình chụp về 0.")
        elif key == 32:
            if face is None:
                print("❌ Không phát hiện thấy khuôn mặt. Vui lòng căn chỉnh lại.")
                continue
                
            out = save_pil_face(face, person_dir, name)
            if out is not None:
                saved += 1
                last_face_thumb = face
                success_counter = 15
                print(f"✅ Đã chụp và lưu thành công: {out.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nHoàn tất! Đã lưu tổng cộng {saved} ảnh cho nhân viên '{name}' tại {person_dir.resolve()}")

def crop_external_image(name: str, input_path: str):
    if FaceDetector is None or cv2 is None:
        print("Lỗi: Thiếu thư viện hoặc FaceDetector không khả dụng.")
        return

    path = Path(input_path)
    if not path.exists():
        print(f"Đường dẫn đầu vào không tồn tại: {input_path}")
        return

    detector = FaceDetector()
    person_dir = _ensure_person_dir(name)

    image_files = []
    if path.is_file():
        image_files.append(path)
    elif path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in path.iterdir():
            if p.is_file() and p.suffix.lower() in valid_exts:
                image_files.append(p)

    if not image_files:
        print(f"Không tìm thấy tệp ảnh hợp lệ tại: {input_path}")
        return

    print(f"Bắt đầu xử lý {len(image_files)} tệp ảnh từ: {input_path}")
    saved = 0
    for img_file in image_files:
        try:
            img = Image.open(img_file).convert("RGB")
            face = detector.align(img)
            if face is not None:
                out = save_pil_face(face, person_dir, name)
                if out:
                    print(f"✅ Đã trích xuất thành công: {out.name} (nguồn: {img_file.name})")
                    saved += 1
            else:
                print(f"❌ Không tìm thấy khuôn mặt trong ảnh: {img_file.name}")
        except Exception as e:
            print(f"Lỗi khi xử lý {img_file.name}: {e}")

    print(f"\nHoàn tất! Đã trích xuất và lưu {saved} khuôn mặt cho '{name}' tại {person_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help='Tên người cần chụp')
    parser.add_argument('--input', type=str, help='Đường dẫn tới tệp ảnh hoặc thư mục ảnh có sẵn để trích xuất')
    parser.add_argument('--max', type=int, default=12, help='Số lượng ảnh tối đa cần lưu (mặc định 12 để đủ 6 góc)')
    parser.add_argument('--camera', type=int, default=0, help='Index của thiết bị camera')
    args = parser.parse_args()

    if args.input:
        crop_external_image(args.name, args.input)
    else:
        min_required = len(POSES) * PHOTOS_PER_POSE
        if args.max < min_required:
            print(f"Lưu ý: Để đủ các góc độ (2 ảnh/góc), số lượng ảnh chụp được nâng lên tối thiểu là {min_required}.")
            args.max = min_required

        capture_from_camera(args.name, max_images=args.max, camera_index=args.camera)

if __name__ == '__main__':
    main()