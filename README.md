# CSCK506 End of Module Assignment - Pneumonia Binary Classifier

A collection of deep CNN architectures for classifying patient chest X-ray images for pneumonia detection. Architectures built and evaluated:

1. A baseline CNN
2. A 16 layer VGG (ImageNet ILSVRC challenger)
3. A 34 layer residual network (ImageNet ILSVRC challenger)
4. EfficientNet

## Project Structure

```
├── / (root)               # Datapreprocessing, classifier and evaluation notebooks
├── checkpoints/           # Saved model weights
├── src/                   # Source package (data analysis and preprocessing, plotting utilities and model definitions)
├── outputs/               # Generated figures for each model
├── processed_data/        # Preprocessed data directly read within notebooks
└── requirements.txt       # Project dependencies
```
