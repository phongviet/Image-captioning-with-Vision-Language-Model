# Image Captioning with Vision-Language Model (VLM)

An image captioning system that combines CLIP (Contrastive Language-Image Pre-training) for visual feature extraction with GPT-2 for natural language generation to automatically generate descriptive captions for images.

Implementation based on the paper: [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2104.08773).
## 🚀 Features

- **Vision-Language Integration**: Uses CLIP to extract rich visual features and GPT-2 for caption generation
- **Flexible Architecture**: Modular design with separate components for feature extraction, model definition, and training
- **Multiple Datasets Support**: Works with Flickr8k dataset and COCO validation set
- **Comprehensive Evaluation**: Includes COCO evaluation metrics (BLEU, METEOR, ROUGE-L, CIDEr, SPICE)
- **Pre-trained Model**: Includes best model checkpoint for immediate inference

## 📁 Project Structure

```
├── data/
│   ├── captions.txt              # Flickr8k captions
│   ├── clip_features.pkl         # Pre-extracted CLIP features
│   ├── Images/                   # Flickr8k images
│   └── COCO/                     # COCO dataset
│       ├── annotations/
│       └── val2017/
├── models.py                     # Model architecture definitions
├── dataset.py                    # Dataset classes and preprocessing
├── train.ipynb                   # Training notebook
├── generate_predictions.py       # Generate predictions on COCO dataset
├── coco_eval.py                  # COCO evaluation script
├── best_model.pth               # Best trained model checkpoint
├── model_predictions.json       # Generated predictions
└── main.py                      # Main entry point
```

## 🏗️ Architecture

### ClipCapModel
- **Visual Encoder**: CLIP ViT-B/32 for extracting 512-dimensional image features
- **Projection Layer**: Multi-layer perceptron (MLP) to map CLIP features to GPT-2 embedding space
- **Language Model**: GPT-2 for autoregressive caption generation
- **Prefix Integration**: Uses learnable prefix embeddings to condition text generation on visual features

### Key Components
1. **MLP Projection**: Maps 512-D CLIP features to 768-D GPT-2 embedding space
2. **Prefix Length**: 10 tokens used as visual context for text generation
3. **Temperature Sampling**: Configurable generation with temperature control and beam search

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Image captioning with VLM"

# Install required packages
pip install torch torchvision transformers
pip install pillow pandas scikit-learn
pip install pycocotools
pip install pycocoevalcap
```

## 📊 Dataset Setup

### Flickr8k Dataset
1. Download the Flickr8k dataset
2. Place images in `data/Images/`
3. Place captions file as `data/captions.txt`

### COCO Dataset (for evaluation)
1. Download COCO val2017 images
2. Download annotations
3. Place in `data/COCO/` structure as shown above

## 🎯 Usage

### Training
```python
# Open and run the training notebook
jupyter notebook train.ipynb
```

## 📈 Model Performance

The model is evaluated using standard image captioning metrics:
- **BLEU-1, BLEU-4**: N-gram overlap scores
- **METEOR**: Semantic similarity metric
- **CIDEr**: Consensus-based evaluation

### COCO Validation Set Results
| Bleu-1 | Bleu-4 | METEOR  | CIDEr  |
|--------|--------|---------|--------|
| 51.1   | 11.9   | 17.8    | 36.7   |

## 🔧 Key Features

### Data Preprocessing
- Automatic CLIP feature extraction and caching
- Train/validation/test data splitting
- Caption tokenization and padding
- Attention mask generation

### Model Architecture
- Learnable visual-to-text projection
- Prefix-based conditioning for GPT-2
- Temperature-controlled generation
- Beam search decoding
- Early stopping mechanisms

### Generation Options
- Configurable temperature and top-p sampling
- Beam search with configurable beam size
- N-gram repetition prevention
- Forced sentence ending

## 🚀 Getting Started

1. **Install dependencies** as listed above
2. **Prepare datasets** in the required structure
3. **Run training** using the Jupyter notebook
4. **Generate predictions** using the trained model
5. **Evaluate performance** using COCO metrics


