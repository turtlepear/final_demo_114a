import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 資料集路徑
data_folders = {
    'happy': "C:/Users/turtlepeer/project1/archive/images/happy",
    'sad': "C:/Users/turtlepeer/project1/archive/images/sad",
    'angry': "C:/Users/turtlepeer/project1/archive/images/angry",
    'relaxed': "C:/Users/turtlepeer/project1/archive/images/relaxed"
}
# 情緒對應表，模型輸出數字，轉換成對應文字
label_map = {'happy':0, 'sad':1, 'angry':2, 'relaxed':3}

# 自訂 Dataset
class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images    # 圖片
        self.labels = labels    # 對應圖片的label
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

# 載入資料夾內的圖片
def load_images(folder):
    imgs = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224))
            imgs.append(img)
    return imgs

# 載入所有類別的圖片和label
images, labels = [], []
for emotion, folder in data_folders.items():
    imgs = load_images(folder)
    lbls = [label_map[emotion]] * len(imgs)
    images.extend(imgs)
    labels.extend(lbls)

images = np.array(images, dtype=np.uint8)
labels = np.array(labels)

# 切分資料(80%, 20%)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Data Augmentation(資料增強)和正規劃
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = EmotionDataset(X_train, y_train, transform=train_transform)
test_dataset = EmotionDataset(X_test, y_test, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)   # 訓練集隨機打亂
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)    # 測試集不打亂

# 模型：ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False


num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 4)  # 分4類
)
model = model.to(device)

# Loss & 用adam將loss降到最低
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# 訓練
epochs = 100
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

# 計算訓練集準確率
    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss/total:.4f}, Acc: {train_acc:.2f}%")

    # 測試
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

# 計算驗證準確率
    acc = 100. * correct / total
    print(f"Validation Acc: {acc:.2f}%")

    # 儲存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "resnet_expression_model.pth")
        print("Model saved!")

print("Best Validation Accuracy:", best_acc)
