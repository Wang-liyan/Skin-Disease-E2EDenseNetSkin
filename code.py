import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from itertools import cycle  # 这是解决错误的关键导入
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.cluster import KMeans  # 添加这行
import json
# 全局参数
base_dir = "ISIC_2019_Training_Input"
groundtruth_path = "ISIC_2019_Training_GroundTruth.csv"
labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
minority_classes = ['DF', 'VASC', 'SCC', 'AK']  # 添加这行定义少数类
nv_target_size = 3500  # 添加这行：NV类目标样本量
img_size = 224
batch_size = 32
num_epochs =50
save_dir = "training_results"  # 添加保存目录
os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
groundtruth = pd.read_csv(groundtruth_path)
# 1. 数据准备（按指定数量删除）
groundtruth = pd.read_csv(groundtruth_path)
groundtruth['label'] = groundtruth[labels].idxmax(axis=1)

print("groundtruth[labels]:",groundtruth[labels])
print("groundtruth[label]:",groundtruth["label"])
# 验证图片存在性
groundtruth['exists'] = groundtruth['image'].apply(
    lambda x: os.path.exists(os.path.join(base_dir, x + '.jpg')))
valid_data = groundtruth[groundtruth['exists']].copy()
def oversample_minority_classes(file_paths, labels_indices, minority_classes, labels, target_samples=1000):
    """
    对少数类进行过采样
    :param file_paths: 原始文件路径列表
    :param labels_indices: 原始标签索引列表
    :param minority_classes: 少数类名称列表
    :param labels: 所有类别名称列表
    :param target_samples: 每个少数类的目标样本数
    :return: 过采样后的文件路径和标签索引
    """
    # 将少数类名称转换为索引
    minority_indices = [labels.index(cls) for cls in minority_classes]

    # 打印原始少数类样本数量
    print("\n过采样前少数类样本数量:")
    original_counts = {}
    for cls in minority_classes:
        cls_idx = labels.index(cls)
        count = sum(1 for l in labels_indices if l == cls_idx)
        original_counts[cls] = count
        print(f"{cls}: {count}张")

    # 创建增强转换
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augmented_paths = []
    augmented_labels = []

    for cls in minority_classes:
        cls_idx = labels.index(cls)
        # 获取当前少数类的所有样本
        cls_paths = [p for p, l in zip(file_paths, labels_indices) if l == cls_idx]
        current_count = len(cls_paths)

        if current_count == 0:
            print(f"警告: {cls}类没有样本可供过采样")
            continue
        # 计算需要生成的样本数量
        needed = max(0, target_samples - current_count)

        # 对当前少数类进行过采样
        for i in range(needed):
            # 随机选择一个原始样本
            original_path = cls_paths[i % current_count]

            # 创建增强后的"虚拟路径"
            augmented_path = f"augmented_{i}_{original_path}"
            augmented_paths.append(augmented_path)
            augmented_labels.append(cls_idx)

        print(f"{cls}类: 原始{current_count}张 -> 新增{needed}张 -> 总计{current_count + needed}张")

    # 合并原始数据和增强数据
    all_paths = file_paths + augmented_paths
    all_labels = labels_indices + augmented_labels

    # 打印总体统计
    print("\n过采样后少数类样本数量:")
    for cls in minority_classes:
        cls_idx = labels.index(cls)
        count = sum(1 for l in all_labels if l == cls_idx)
        print(f"{cls}: {count}张 (增加了{count - original_counts[cls]}张)")

    print(f"\n总样本量: 原始{len(file_paths)}张 -> 过采样后{len(all_paths)}张")

    return all_paths, all_labels
def stratified_undersample_class(train_paths, train_labels, labels,target_class, target_size):
    """
    NV类分层欠采样
    :param train_paths: 训练集路径列表
    :param train_labels: 训练集标签列表
    :param labels: 所有类别列表
    :param target_size: NV目标样本量
    :return: 欠采样后的路径和标签
    """
    class_index = labels.index(target_class)

    class_paths = np.array([p for p, l in zip(train_paths, train_labels) if l == class_index])

    if len(class_paths) <= target_size:
        return train_paths, train_labels
    # 简易特征提取
    features = []
    for path in class_paths:
        img = Image.open(path).convert('L').resize((64, 64))
        features.append(np.array(img).flatten())
    features = np.array(features)

    # 聚类分层
    n_clusters = min(50, target_size // 100)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)

    # 分层采样
    sampled_indices = []
    for c in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == c)[0]
        if len(cluster_indices) > 0:
            n_samples = max(1, int(target_size * len(cluster_indices) / len(class_paths)))
            sampled_indices.extend(np.random.choice(cluster_indices, n_samples, replace=False))

    # 构建新数据集
    sampled_paths = class_paths[sampled_indices[:target_size]]
    non_class_paths = [p for p, l in zip(train_paths, train_labels) if l != class_index]
    non_class_labels = [l for l in train_labels if l != class_index]

    return non_class_paths + sampled_paths.tolist(), non_class_labels + [class_index]*len(sampled_paths)
# 2. 创建数据集
file_paths = [os.path.join(base_dir, row['image'] + '.jpg') for _, row in groundtruth.iterrows()]
labels_indices = [labels.index(row['label']) for _, row in groundtruth.iterrows()]


# 划分训练集和测试集
train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths, labels_indices,
    test_size=0.3,
    random_state=42,
    stratify=labels_indices
)
# 对训练集中的少数类进行过采样 (添加这部分)
train_paths, train_labels = oversample_minority_classes(
    train_paths, train_labels,
    minority_classes=minority_classes,
    labels=labels,
    target_samples=1500  # 目标样本数
)
# 2. 再对NV类分层欠采样 (添加这部分)
train_paths, train_labels = stratified_undersample_class(
    train_paths, train_labels,
    labels=labels,
    target_class='NV',
    target_size=nv_target_size
)
train_paths, train_labels = stratified_undersample_class(
    train_paths, train_labels,
    labels=labels,
    target_class='MEL',
    target_size=nv_target_size
)

print(f"\n训练集: {len(train_paths)}张 | 测试集: {len(test_paths)}张")
from sklearn.cluster import KMeans
from collections import Counter
# 3. 数据加载
class ISICDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # 判断是否是增强样本
        if file_path.startswith('augmented_'):
            # 提取原始路径
            original_path = '_'.join(file_path.split('_')[2:])
            image = Image.open(original_path).convert('RGB')
            # 对增强样本应用更强的数据增强
            image = self.augment_transform(image)
        else:
            image = Image.open(file_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

        return image, label
# Grad-CAM核心实现（独立模块）
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        """
        :param model: 加载好的DenseNet模型
        :param target_layer: 需要可视化的目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # 注册hook
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        :param input_tensor: 输入图像张量(batch=1)
        :param class_idx: 目标类别索引（默认使用模型预测类别）
        :return: Grad-CAM热力图(numpy数组)
        """
        # 前向传播
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        # 反向传播
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        # 计算权重
        pooled_grads = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(self.activations * pooled_grads, dim=1, keepdim=True)
        cam = F.relu(cam)  # 只保留正向激活

        # 后处理
        cam = F.interpolate(cam, input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()
#Grad-CAM可视化
def visualize_gradcam(model, test_loader, device, target_layer, num_samples=3):
    """
    Grad-CAM可视化演示
    :param model: 训练好的模型
    :param test_loader: 测试集DataLoader
    :param device: 计算设备
    :param target_layer: 目标卷积层
    :param num_samples: 可视化样本数
    """
    gradcam = GradCAM(model, target_layer)
    samples, labels = next(iter(test_loader))
    save_path=save_dir
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(min(num_samples, len(samples))):
        # 准备输入
        img_tensor = samples[i].unsqueeze(0).to(device)
        true_label = labels[i].item()

        # 生成CAM
        cam = gradcam.generate(img_tensor, true_label)

        # 原始图像（反归一化）
        img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # 可视化
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(img)
        plt.title(f"True: {labels[true_label]}")
        plt.axis('off')

        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(img)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title("Grad-CAM")
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)  # 保存图片  plt.save("lujing/g.png")
    plt.show()  # 显示图片

# 图像增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 创建Dataset和DataLoader
train_dataset = ISICDataset(train_paths, train_labels, transform=train_transform)
test_dataset = ISICDataset(test_paths, test_labels, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 4. 模型定义
# model = models.resnet50(pretrained=True)
# num_ftrs = model.fc.in_features  # ResNet的全连接层名为fc
# model.fc = nn.Linear(num_ftrs, len(labels))  # 替换分类层

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(labels))
# 使用带CBAM的DenseNet
# model = DenseNet121_CBAM(num_classes=len(labels))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
import os
from torch import nn

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 初始化最佳准确率和模型路径
best_accuracy = 0.0
best_model_path = os.path.join(save_dir, 'best_model.pth')
# 训练循环中添加以下代码
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 训练阶段
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 计算训练指标
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:  # 使用测试集作为验证集
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

    # 打印训练/验证指标
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss/len(test_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
    # 将当前epoch的指标保存到列表中
    train_history = []
    train_history.append({
        'Epoch': epoch + 1,
        'Train Loss': train_loss,
        'Train Accuracy': train_accuracy,
        'Val Loss': val_loss / len(test_loader),
        'Val Accuracy': val_accuracy
    })
    # 在训练循环结束后，将训练历史保存为CSV文件
    train_history_df = pd.DataFrame(train_history)
    train_history_df.to_csv(f"{save_dir}/train_history.csv", index=False)
    # 保存最佳模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'accuracy': val_accuracy,
            'labels': labels  # 保存标签映射关系
        }, best_model_path)
        print(f'New best model saved at epoch {epoch+1} with accuracy {val_accuracy:.2f}%')

model.eval()
all_labels = []
all_predictions = []
all_probabilities = []
all_image_paths = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predicted.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

# 转换为numpy数组
y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_scores = np.array(all_probabilities)
# # 1. 分类报告
# print("\n========== Classification Report ==========")
# print(classification_report(y_true, y_pred, digits=4))


# #  关键指标
# from sklearn.metrics import accuracy_score, balanced_accuracy_score

# print("\n========== Key Metrics ==========")
# print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
# print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")

# 保存测试结果
test_results = {
    'true_labels': y_true.tolist(),
    'pred_labels': y_pred.tolist(),
    'probabilities': y_scores.tolist(),
    'image_paths': test_paths[:len(y_true)]  # 确保长度一致
}
pd.DataFrame(test_results).to_csv(f"{save_dir}/test_predictions.csv", index=False)
# 获取目标层（DenseNet121的最后一个卷积层）
#target_layer = model.layer4[-1].conv3  # 或根据实际结构调整
target_layer = model.features.denseblock4.denselayer16.conv2  # 根据实际模型结构调整
# 执行可视化
print("\n========== Grad-CAM 可视化 ==========")
#visualize_gradcam(model, test_loader, device, target_layer, num_samples=8,sasve_path="/root/autodl-tmp/skin_disease/Grad-Cam.pnj"）
visualize_gradcam(model, test_loader, device, target_layer, num_samples=20)


