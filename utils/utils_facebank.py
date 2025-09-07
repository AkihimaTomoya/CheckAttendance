# utils.py
# Chứa các hàm tiện ích hỗ trợ cho việc inference nhận dạng khuôn mặt.

from PIL import Image
import numpy as np
import torch
from torchvision import transforms as trans
import cv2


def load_facebank(conf):
    """
    Tải facebank đã được lưu.
    Đảm bảo 'names' được trả về dưới dạng dictionary.
    """
    facebank_path = conf.facebank_path / 'facebank.pth'
    if not facebank_path.is_file():
        print("Facebank file not found. Returning empty facebank.")
        return torch.empty(0), {}

    try:
        embeddings, names = torch.load(facebank_path)

        # Đảm bảo names luôn là dictionary để tương thích
        if isinstance(names, (list, np.ndarray)):
            print("Converting names from list/array to dictionary.")
            names_dict = {i: name for i, name in enumerate(names)}
            return embeddings, names_dict

        # Nếu đã là dictionary thì trả về luôn
        if isinstance(names, dict):
            return embeddings, names

        print(f"Warning: Unsupported type for names in facebank: {type(names)}")
        return torch.empty(0), {}

    except Exception as e:
        print(f"Error loading facebank: {e}. Returning empty facebank.")
        return torch.empty(0), {}


def prepare_facebank(conf, model, detector, tta=True):
    """
    Quét thư mục facebank, nhận diện, và tạo ra file facebank.pth.
    Đảm bảo 'names' được tạo và lưu dưới dạng dictionary.
    """
    facebank_path = conf.facebank_path
    if not facebank_path.is_dir():
        print("Facebank directory not found. Creating one.")
        facebank_path.mkdir(parents=True, exist_ok=True)
        return torch.empty(0), {}

    embeddings = []
    # SỬA LỖI: names phải là một dictionary
    names_dict = {}

    # Dùng một list tạm thời để xây dựng dict
    identity_list = []

    # Lấy danh sách các thư mục con (tên người)
    person_folders = sorted([p for p in facebank_path.iterdir() if p.is_dir()])

    # Ánh xạ tên vào một index duy nhất
    for idx, person_folder in enumerate(person_folders):
        name = person_folder.name
        identity_list.append(name)
        names_dict[idx] = name

        for img_path in person_folder.iterdir():
            if not img_path.is_file():
                continue

            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"Could not open image {img_path}: {e}")
                continue

            # Sử dụng detector để tìm khuôn mặt
            # Giả định detector.align trả về một khuôn mặt đã được căn chỉnh
            face_img = detector.align(img)

            if face_img is None:
                print(f"No face detected in {img_path}")
                continue

            with torch.no_grad():
                # Áp dụng transform và đưa vào model
                emb = model(conf.test_transform(face_img).to(conf.device).unsqueeze(0))

                if tta:
                    # Test-Time Augmentation (lật ảnh)
                    mirror = trans.functional.hflip(face_img)
                    emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    # Chuẩn hóa L2 cho embedding
                    embeddings.append(torch.nn.functional.normalize(emb + emb_mirror, p=2, dim=1))
                else:
                    embeddings.append(torch.nn.functional.normalize(emb, p=2, dim=1))

    if len(embeddings) == 0:
        print("No faces found to create facebank.")
        return torch.empty(0), {}

    # Gộp tất cả embedding thành một tensor duy nhất
    final_embeddings = torch.cat(embeddings).squeeze()

    # Lưu lại facebank
    save_path = facebank_path / 'facebank.pth'
    torch.save((final_embeddings, names_dict), save_path)

    print(f"Facebank prepared and saved. Found {len(names_dict)} identities.")
    return final_embeddings, names_dict


def draw_box_name(bbox, name, frame):
    """Vẽ bounding box và tên lên frame."""
    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    frame = cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame