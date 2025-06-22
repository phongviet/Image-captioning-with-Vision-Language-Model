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
        image_features = clip_model.get_image_features(**inputs)

    return image_features.cpu().numpy()[0]


def load_model(model_path, device):
    model = ClipCapModel(clip_dim=512)

    # Tải checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def generate_predictions_from_images(model_path, coco_annotations_path, coco_images_folder,
                                    output_file='model_predictions.json', batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading COCO annotations from {coco_annotations_path}")
    coco = COCO(coco_annotations_path)
    image_ids = coco.getImgIds()
    print(f"Found {len(image_ids)} images in validation set")

    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_model.eval()

    print(f"Loading caption model from {model_path}")
    model = load_model(model_path, device)
    model.to(device)
    model.eval()

    results = []
    print("Generating captions...")

    for idx, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        # Lấy thông tin ảnh từ COCO
        img_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(coco_images_folder, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}, skipping")
            continue

        try:
            image_feature = extract_clip_features(image_path, clip_model, processor)

            image_feature = torch.tensor(image_feature).unsqueeze(0).to(device)

            with torch.no_grad():
                caption = model.generate(image_feature, max_length=20)[0]

            results.append({
                "image_id": image_id,
                "caption": caption,
                "id": idx
            })

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_ids)} images")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done! Generated captions for {len(results)} images")
    return results


def generate_predictions(model_path, clip_features_path, coco_annotations_path, output_file='model_predictions.json'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading COCO annotations from {coco_annotations_path}")
    coco = COCO(coco_annotations_path)
    image_ids = coco.getImgIds()
    print(f"Found {len(image_ids)} images in validation set")

    print(f"Loading model from {model_path}")
    model = load_model(model_path, device)
    model.to(device)
    model.eval()

    print(f"Loading CLIP features from {clip_features_path}")
    with open(clip_features_path, 'rb') as f:
        clip_features = pickle.load(f)

    features_dict = {}
    if isinstance(clip_features, dict):
        features_dict = clip_features

    results = []
    print("Generating captions...")

    for idx, image_id in enumerate(image_ids):
        if str(image_id) in features_dict:
            image_feature = features_dict[str(image_id)]
        elif image_id in features_dict:
            image_feature = features_dict[image_id]
        else:
            print(f"Warning: No feature found for image_id {image_id}, skipping")
            continue

        image_feature = torch.tensor(image_feature).unsqueeze(0).to(device)

        with torch.no_grad():
            caption = model.generate(image_feature, max_length=20)[0]

        results.append({
            "image_id": image_id,
            "caption": caption,
            "id": idx
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_ids)} images")

    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done! Generated captions for {len(results)} images")
    return results


def main():
    model_path = "best_model.pth"
    coco_annotations_path = "data/COCO/annotations/captions_val2017.json"
    coco_images_folder = "data/COCO/val2017"
    output_file = "model_predictions.json"

    generate_predictions_from_images(
        model_path=model_path,
        coco_annotations_path=coco_annotations_path,
        coco_images_folder=coco_images_folder,
        output_file=output_file
    )

    # clip_features_path = "data/clip_features.pkl"
    # generate_predictions(
    #     model_path=model_path,
    #     clip_features_path=clip_features_path,
    #     coco_annotations_path=coco_annotations_path,
    #     output_file=output_file
    # )


if __name__ == "__main__":
    main()
