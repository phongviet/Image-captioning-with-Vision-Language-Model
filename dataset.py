import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, GPT2Tokenizer
from sklearn.model_selection import train_test_split


class Flickr8kDataset(Dataset):
    def __init__(self, image_features, captions, tokenizer, prefix_length=10,
                 max_seq_len=50):
        self.image_features = image_features
        self.captions = captions
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.max_seq_len = max_seq_len

        self.tokenized_captions = []
        for caption in captions:
            tokens = self.tokenizer.encode(caption + self.tokenizer.eos_token)
            self.tokenized_captions.append(tokens)

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        prefix = self.image_features[idx]
        tokens = self.tokenized_captions[idx]

        padding = self.max_seq_len - len(tokens)
        if padding > 0:
            tokens = tokens + [self.tokenizer.pad_token_id] * padding
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]

        return {
            'image_features': torch.tensor(prefix, dtype=torch.float32),
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor([1 if t != self.tokenizer.pad_token_id else 0
                                            for t in tokens], dtype=torch.long)
        }


class Flickr8kPreprocessor:
    def __init__(self, data_dir, clip_model_name="openai/clip-vit-base-patch32"):
        self.data_dir = data_dir
        self.clip_model_name = clip_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()

        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_captions(self, captions_file="captions.txt"):
        captions_path = os.path.join(self.data_dir, captions_file)
        captions_data = pd.read_csv(captions_path)
        return captions_data.groupby('image')['caption'].apply(list).reset_index()

    def extract_clip_features(self, images_dir="Images", save_path="clip_features.pkl"):
        images_path = os.path.join(self.data_dir, images_dir)
        image_files = [f for f in os.listdir(images_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()

        features = []
        image_names = []

        with torch.no_grad():
            for image_file in image_files:
                try:
                    image_path = os.path.join(images_path, image_file)
                    image = Image.open(image_path).convert('RGB')

                    inputs = self.clip_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    image_features = self.clip_model.get_image_features(**inputs)
                    features.append(image_features.cpu().numpy())
                    image_names.append(image_file)

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    continue

        features = np.vstack(features)
        features_data = {'features': features, 'image_names': image_names}

        with open(os.path.join(self.data_dir, save_path), 'wb') as f:
            pickle.dump(features_data, f)

        return features_data

    def prepare_train_data(self, features_data, captions_data, test_size=0.2, val_size=0.1):
        name_to_idx = {name: i for i, name in enumerate(features_data['image_names'])}

        matched_features = []
        matched_captions = []

        for _, row in captions_data.iterrows():
            image_name = row['image']
            if image_name in name_to_idx:
                idx = name_to_idx[image_name]
                for caption in row['caption']:
                    matched_features.append(features_data['features'][idx])
                    matched_captions.append(caption)

        matched_features = np.array(matched_features)

        X_temp, X_test, y_temp, y_test = train_test_split(
            matched_features, matched_captions, test_size=test_size, random_state=42)

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
