import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

# ========== Emotion Model ==========
label_map = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Relaxed"}

# 設定ResNet50
emotion_model = models.resnet50(weights=None)
num_features = emotion_model.fc.in_features
emotion_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, len(label_map))
)

emotion_model.load_state_dict(torch.load(
    "TA_project/final_demo/resnet_expression_model.pth", map_location="cpu"))
emotion_model.eval()

# 圖片處理   → Resize → Tensor → Normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========== Pose Model ==========
pose_model = YOLO("C:/Users/turtlepeer/project1/TA_project/final_demo/pose_best.pt")

# ========== 資料夾設定 ==========
input_folder = "C:/Users/turtlepeer/project1/dog-pose/images/train"
output_folder = "C:/Users/turtlepeer/project1/TA_project/image"
os.makedirs(output_folder, exist_ok=True)

# ========== 批次處理圖片 ==========
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        results = pose_model(img)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # 偵測到的邊界框座標
            keypoints = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在圖片上畫出偵測框

                if len(keypoints) > i:  # 繪製關鍵點
                    for (x, y) in keypoints[i]:
                        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
                else:
                    print(f"⚠️ No keypoints for detection in {filename}")

                # Emotion detection
                face_roi = img[y1:int(y1+(y2-y1)*0.5), x1:x2]
                if face_roi.size > 0:
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    face_tensor = transform(face_rgb).unsqueeze(0)

                    with torch.no_grad():
                        outputs = emotion_model(face_tensor)
                        _, predicted = torch.max(outputs, 1)
                        predicted_class = predicted.item()
                        emotion = label_map[predicted_class]

                    cv2.putText(img, f"Emotion: {emotion}", # 在框的上方寫上情緒結果
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)
                else:
                    print(f"⚠️ ROI empty for {filename}, skip emotion detection")

        # 儲存處理後圖片
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img)
        print(f"Processed and saved: {filename}")

print("✅ 所有圖片處理完成！")
