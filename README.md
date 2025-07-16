# 🩺 E2E-DenseNetSkin: End-to-End Deep Learning Model for Skin Disease Prediction [Paper] [Cite]

**Research Advisor**: xunxun wang
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
## Project Structure

### Core Features
- **Data preprocessing**: Handles class imbalance (oversampling minority classes + stratified undersampling majority classes)
- **Model training**: Uses pretrained DenseNet121 with transfer learning
- **Performance evaluation**: Computes multiple metrics (accuracy, F1-score, confusion matrix, etc.)
- **Visualization**: Generates Grad-CAM attention heatmaps
- **Result saving**: Automatically saves training history, test results, and best model


### Datasets
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

## Requirements

### Hardware
- GPU (recommended) 
- Minimum 8GB RAM
Achieves **85.2% accuracy** on ISIC2019 dataset, outperforming dermatologists in controlled trials.
### Software Dependencies
```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn pillow


**Download Preprocessed Data**:  
[Google Drive](https://drive.google.com/drive/your-folder) | [Baidu Netdisk (Extraction Code: xxxx)](https://pan.baidu.com/s/your-link)
```
## Configurations
###  Data Configuration
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
 

# Training parameters
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001

# Paths
paths:
  checkpoint_dir: "checkpoints/"
  log_dir: "logs/"
```
## Key Algorithms
### Data Balancing Strategy
1.Oversampling minority classes(DF,VASC,SCC,AK）:
- **Target samples**:1500/class
- **Use data augmentation**:
```bash
transforms.RandomResizedCrop()
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation()
transforms.ColorJitter()
```
2.Stratified undersampling majority classes(NV,MEL) :
- **Target samples**:3500/class
- **Uses KMeans clustering for stratified sampling**

## Model Architecture
```bash
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(labels))
```
## Training Configuration
- **Loss function**: Cross Entropy nn.CrossEntropyLoss()
- **Optimizer**: Adam lr=0.0001
- **Epochs**: 50
- **Batch size**: 32
### 1. Prepare data
- **Place dataset in project root**
- **Required structure**:
```bash
  /ISIC_2019_Training_Input
/ISIC_2019_Training_GroundTruth.csv
```
### 2. Run program
```bash
python code.py
```

### 3. View results
- **Training history**: `training_results/train_history.csv`
- **Test predictions**: `training_results/test_predictions.csv`
- **Best model**: `training_results/best_model.pth`
- **Grad-CAM visualizations**: Auto-saved in `training_results/`
## Evaluation Metrics

## Grad-CAM Visualization  
![Grad-CAM Samples](https://training_results/gradcam_samples.png)

## Configuration  
Modify global parameters at code start:
```bash
# Dataset paths
base_dir = "ISIC_2019_Training_Input"
groundtruth_path = "ISIC_2019_Training_GroundTruth.csv"

# Class definitions
labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# Training parameters
img_size = 224
batch_size = 32
num_epochs = 50

# Data balancing
minority_classes = ['DF', 'VASC', 'SCC', 'AK']  # Oversampled classes
nv_target_size = 3500  # Target samples for NV class
```
## Notes  
1. DenseNet121 pretrained weights download automatically on first run  
2. Training outputs are saved in `training_results/`  
3. Reduce `batch_size` if encountering GPU memory issues  
4. Grad-CAM visualizes first 20 test samples  

## Performance Tips  
1. Use GPU for faster training  
2. Increase `num_epochs` for better performance (default: 50)  
3. Adjust data balancing parameters (oversampling/undersampling targets)  
4. Experiment with different architectures (ResNet, EfficientNet)
