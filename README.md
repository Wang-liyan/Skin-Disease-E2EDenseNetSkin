# 🩺 E2E-DenseNetSkin: End-to-End Deep Learning Model for Skin Disease Prediction [Paper] [Cite]

**Authors**: Yiran Lin, Liyan Wang, Xiyan Zhang  
**Competition**: Winner of the 10th National Biomedical Engineering Innovation Design Competition (ID:5366)  

![Model Visualization](docs/model_visualization.png)

## About

### E2E-DenseNetSkin Architecture
Our proposed model is an end-to-end deep learning framework for multi-class skin disease diagnosis, featuring:
- **DenseNet121 backbone** with medical-adapted modifications
- **Dynamic hybrid sampling** (K-Means undersampling + targeted oversampling)
- **Virtual pathology engine** for lesion-specific data augmentation
- **Grad-CAM integration** for clinically interpretable decisions

Achieves **85.2% accuracy** on ISIC2019 dataset, outperforming dermatologists in controlled trials.

## Datasets
We use the **ISIC2019** dataset ([Official Link](https://challenge.isic-archive.com/data/#2019)):

| Class | Original Samples | Processed Samples |
|-------|------------------|-------------------|
| NV    | 12,875           | 3,500             |
| MEL   | 4,522            | 3,500             |
| BCC   | 3,323            | 1,500             |
| BKL   | 2,624            | 1,500             |
| SCC   | 628              | 1,500             |
| AK    | 867              | 1,500             |
| VASC  | 253              | 1,500             |
| DF    | 239              | 1,500             |

**Download Preprocessed Data**:  
[Google Drive](https://drive.google.com/drive/your-folder) | [Baidu Netdisk (Extraction Code: xxxx)](https://pan.baidu.com/s/your-link)

## Configurations
### 1. Data Configuration
Edit `configs/data_config.yaml`:
```yaml
# Sampling parameters
sampling:
  minority_target: 1500    # Target sample count for minority classes
  kmeans_clusters: 8       # Number of clusters for K-Means undersampling

# Augmentation parameters
augmentation:
  rotation_range: 30       # Max rotation angle (degrees)
  h_flip: True             # Enable horizontal flipping
  v_flip: True             # Enable vertical flipping
  color_jitter: [0.2, 0.2, 0.2, 0.1] # [brightness, contrast, saturation, hue]
# Model parameters
model:
  name: "E2E_DenseNetSkin"
  num_classes: 8
  dropout_rate: 0.5

# Training parameters
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  class_weights: [0.8, 1.2, 1.0, 1.0, 1.5, 1.3, 1.8, 1.8] # Per-class weights

# Paths
paths:
  checkpoint_dir: "checkpoints/"
  log_dir: "logs/"
## 📊 Results & Clinical Validation

| Metric        | E2E-DenseNetSkin | ResNet50 (Baseline) |
|---------------|------------------|---------------------|
| **Accuracy**  | **85.2 %**       | 83.2 %              |
| **Recall**    | **84.2 %**       | 80.1 %              |
| **AUC**       | **0.89**         | 0.85                |
| **F1-Score**  | **0.86**         | 0.82                |

> **Clinical Validation Highlights**  
> - 82.4 % overlap with dermatologists’ annotations  
> - 23 ms average inference on NVIDIA T4 GPU  
> - 38 % reduction in misdiagnosis for rare lesions

![Performance comparison](assets/results.png)

*Grad-CAM examples: [docs/gradcam_examples.png](https://docs/gradcam_examples.png)*
