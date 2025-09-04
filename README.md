# Image Captioning with Vision-Language Model (VLM)

An image captioning system that combines CLIP (Contrastive Language-Image Pre-training) for visual feature extraction with GPT-2 for natural language generation to automatically generate descriptive captions for images.

Implementation based on the paper: [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2104.08773).

## Architecture

### ClipCapModel
- **Visual Encoder**: CLIP ViT-B/32 for extracting 512-dimensional image features
- **Projection Layer**: Multi-layer perceptron (MLP) to map CLIP features to GPT-2 embedding space
- **Language Model**: GPT-2 for autoregressive caption generation
- **Prefix Integration**: Uses learnable prefix embeddings to condition text generation on visual features

### Key Components
1. **MLP Projection**: Maps 512-D CLIP features to 768-D GPT-2 embedding space
2. **Prefix Length**: 10 tokens used as visual context for text generation
3. **Temperature Sampling**: Configurable generation with temperature control and beam search

## Installation

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

## Dataset Setup

### Flickr8k Dataset
1. Download the Flickr8k dataset 
2. Place images in `data/Images/`
3. Place captions file as `data/captions.txt`

### COCO Dataset (for evaluation)
1. Download COCO val2017 images
2. Download annotations
3. Place in `data/COCO/` structure as shown above

## Usage

### Training
```python
# Open and run the training notebook
jupyter notebook train.ipynb
```

## Model Performance

The model is evaluated using standard image captioning metrics:
- **BLEU-1, BLEU-4**: N-gram overlap scores
- **METEOR**: Semantic similarity metric
- **CIDEr**: Consensus-based evaluation

### COCO Validation Set Results
| Bleu-1 | Bleu-4 | METEOR  | CIDEr  |
|--------|--------|---------|--------|
| 51.1   | 11.9   | 17.8    | 36.7   |



