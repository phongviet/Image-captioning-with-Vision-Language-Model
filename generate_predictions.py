import os
import json
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from models import ClipCapModel
from pycocotools.coco import COCO
from transformers import CLIPProcessor, CLIPModel


def extract_clip_features(image_path, clip_model, processor):
    image = Image.open(image_path).convert('RGB')

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        # Sử dụng get_image_features giống như trong dataset.py
        image_features = clip_model.get_image_features(**inputs)

    return image_features.cpu().numpy()[0]


def load_model(model_path, device):
    model = ClipCapModel(clip_dim=512)

    # Tải checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Kiểm tra xem checkpoint có phải dạng đầy đủ không
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Loading model from checkpoint with model_state_dict")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Loading model from direct state_dict")
        model.load_state_dict(checkpoint)

    return model


def generate_predictions_from_images(model_path, coco_annotations_path, coco_images_folder,
                                    output_file='model_predictions.json', batch_size=32):
    # Kiểm tra device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Tải COCO annotations để lấy image_ids
    print(f"Loading COCO annotations from {coco_annotations_path}")
    coco = COCO(coco_annotations_path)
    image_ids = coco.getImgIds()
    print(f"Found {len(image_ids)} images in validation set")

    # Tải CLIP model - sử dụng chính xác cấu hình như trong dataset.py
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_model.eval()

    # Tải caption model với hàm mới
    print(f"Loading caption model from {model_path}")
    model = load_model(model_path, device)
    model.to(device)
    model.eval()

    # Tạo danh sách dự đoán
    results = []
    print("Generating captions...")

    # Lặp qua từng ảnh trong dataset
    for idx, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        # Lấy thông tin ảnh từ COCO
        img_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(coco_images_folder, img_info['file_name'])

        # Kiểm tra file ảnh có tồn tại không
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}, skipping")
            continue

        try:
            # Trích xuất CLIP features
            image_feature = extract_clip_features(image_path, clip_model, processor)

            # Chuyển feature sang tensor
            image_feature = torch.tensor(image_feature).unsqueeze(0).to(device)

            # Tạo caption
            with torch.no_grad():
                caption = model.generate(image_feature, max_length=20)[0]

            # Thêm vào kết quả
            results.append({
                "image_id": image_id,
                "caption": caption,
                "id": idx
            })

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_ids)} images")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Lưu kết quả vào file JSON
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done! Generated captions for {len(results)} images")
    return results


def generate_predictions(model_path, clip_features_path, coco_annotations_path, output_file='model_predictions.json'):
    """
    Tạo file model_predictions.json chứa caption dự đoán cho tập dữ liệu val COCO.

    Args:
        model_path (str): Đường dẫn đến file model đã lưu (.pth)
        clip_features_path (str): Đường dẫn đến file CLIP features
        coco_annotations_path (str): Đường dẫn đến file COCO annotations
        output_file (str): Tên file output (mặc định: model_predictions.json)
    """
    # Kiểm tra device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Tải COCO annotations để lấy image_ids
    print(f"Loading COCO annotations from {coco_annotations_path}")
    coco = COCO(coco_annotations_path)
    image_ids = coco.getImgIds()
    print(f"Found {len(image_ids)} images in validation set")

    # Tải mô hình với hàm mới
    print(f"Loading model from {model_path}")
    model = load_model(model_path, device)
    model.to(device)
    model.eval()

    # Tải CLIP features
    print(f"Loading CLIP features from {clip_features_path}")
    with open(clip_features_path, 'rb') as f:
        clip_features = pickle.load(f)

    # Tạo dictionary từ image_id sang CLIP features
    features_dict = {}
    # Nếu clip_features là dictionary, sử dụng trực tiếp
    if isinstance(clip_features, dict):
        features_dict = clip_features
    # Nếu không, giả sử cấu trúc khác và cần xử lý phù hợp

    # Tạo danh sách dự đoán
    results = []
    print("Generating captions...")

    for idx, image_id in enumerate(image_ids):
        # Lấy CLIP feature cho image_id
        # Điều chỉnh cách truy cập features phù hợp với cấu trúc dữ liệu của bạn
        if str(image_id) in features_dict:
            image_feature = features_dict[str(image_id)]
        elif image_id in features_dict:
            image_feature = features_dict[image_id]
        else:
            print(f"Warning: No feature found for image_id {image_id}, skipping")
            continue

        # Chuyển feature sang tensor
        image_feature = torch.tensor(image_feature).unsqueeze(0).to(device)

        # Tạo caption
        with torch.no_grad():
            caption = model.generate(image_feature, max_length=20)[0]

        # Thêm vào kết quả
        results.append({
            "image_id": image_id,
            "caption": caption,
            "id": idx  # Hoặc có thể dùng image_id
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_ids)} images")

    # Lưu kết quả vào file JSON
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done! Generated captions for {len(results)} images")
    return results


# Hàm main mẫu để chạy trực tiếp (không dùng argparse)
def main():
    # Các tham số mặc định - bạn có thể thay đổi trực tiếp tại đây
    model_path = "best_model.pth"
    coco_annotations_path = "data/COCO/annotations/captions_val2017.json"
    coco_images_folder = "data/COCO/val2017"
    output_file = "model_predictions.json"

    # Gọi hàm tạo predictions - chọn một trong hai hàm dưới đây

    # Cách 1: Trích xuất CLIP features trực tiếp từ ảnh
    generate_predictions_from_images(
        model_path=model_path,
        coco_annotations_path=coco_annotations_path,
        coco_images_folder=coco_images_folder,
        output_file=output_file
    )

    # Cách 2: Dùng CLIP features đã có sẵn (bỏ comment nếu muốn dùng)
    # clip_features_path = "data/clip_features.pkl"
    # generate_predictions(
    #     model_path=model_path,
    #     clip_features_path=clip_features_path,
    #     coco_annotations_path=coco_annotations_path,
    #     output_file=output_file
    # )


if __name__ == "__main__":
    main()
