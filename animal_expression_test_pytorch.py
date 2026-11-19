import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# Emotion label map
label_map = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Relaxed"}

# 建立 ResNet50 結構
model = models.resnet50(weights=None)  # 不用預訓練權重
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, len(label_map))
)

# 載入訓練好的權重
model.load_state_dict(torch.load("TA_project/final_demo/resnet_expression_model.pth", map_location="cpu"))
model.eval()

# 圖片預處理 (要和訓練時一致)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or cannot be opened")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)          # apply torchvision transform
    img = img.unsqueeze(0)        # 增加 batch 維度 (1,3,224,224)
    return img

# 測試一張圖片
img_path = "TA_project/final_demo/307DE165-D143-40FE-BB39-351FC460284C.jpg"
processed_img = preprocess_image(img_path)

with torch.no_grad():
    outputs = model(processed_img)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    predicted_label = label_map[predicted_class]

print("Predicted Emotion:", predicted_label)
