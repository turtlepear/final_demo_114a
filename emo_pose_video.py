import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

# ResNet 50 Emotion Model
label_map = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Relaxed"}

emotion_model = models.resnet50(weights=None)
num_features = emotion_model.fc.in_features
emotion_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, len(label_map))
)
emotion_model.load_state_dict(torch.load("TA_project/final_demo/resnet_expression_model.pth", map_location="cpu", weights_only=True))
emotion_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Pose Model
pose_model = YOLO("C:/Users/turtlepeer/project1/TA_project/final_demo/pose_best.pt")

# 定義骨架連接關係 (依 dataset 調整)
skeleton = [
    (5, 9), (0, 2),
    (22, 12), (2, 5),
    (3, 5),
    (2, 22), (7, 7),
    (2, 6), (22, 14),
    (11, 12),
    (12, 13), (13, 13),
    (20, 20), (14, 16)
]

# 開啟影片
cap = cv2.VideoCapture("TA_project/final_demo/IMG_6070.mp4")

# 取得影片尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 設定輸出影片
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_pose = cv2.VideoWriter("output_pose.mp4", fourcc, fps, (frame_width, frame_height))

# 儲存骨架數據用於 3D 動畫
frames_3d = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model(frame)

    for r in results:
        if r.keypoints is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            keypoints = r.keypoints.xy.cpu().numpy()

            for (box, kp) in zip(boxes, keypoints):
                x1, y1, x2, y2 = map(int, box)

                # 畫 Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Dog", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Emotion 辨識
                face_roi = frame[y1:y2, x1:x2]
                emotion = "Unknown"
                if face_roi.size > 0:
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    face_tensor = transform(face_rgb).unsqueeze(0)

                    with torch.no_grad():
                        outputs = emotion_model(face_tensor)
                        _, predicted = torch.max(outputs, 1)
                        predicted_class = predicted.item()
                        emotion = label_map[predicted_class]

                # 畫關鍵點 + 顯示index
                kp3d = []
                for idx, (x, y) in enumerate(kp):
                    x, y = int(x), int(y)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (x+5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    kp3d.append([x, y, 0])  # 存 3D (Z=0)

                frames_3d.append(kp3d)

                # 畫骨架
                for (i, j) in skeleton:
                    if i < len(kp) and j < len(kp):
                        xi, yi = kp[i]
                        xj, yj = kp[j]
                        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (255, 0, 0), 2)

                # 顯示 Emotion
                cv2.putText(frame, f"Emotion: {emotion}", (x1, y1-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # 寫入輸出影片
    out_pose.write(frame)

    cv2.imshow("Dog Pose + Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_pose.release()
cv2.destroyAllWindows()