# 狗狗情緒與姿態辨識（final_demo_114a）

結合 **ResNet50 情緒分類** 與 **YOLO 姿態估計**，偵測圖片或影片中的狗，標出骨架關鍵點，並判斷牠的情緒（Happy / Sad / Angry / Relaxed）。

## 功能

- **情緒分類**：用 ImageNet 預訓練的 ResNet50 做遷移學習（凍結 backbone，只訓練分類層），分 4 種情緒。
- **姿態估計**：用 Ultralytics YOLO pose 模型偵測狗的位置與關鍵點。
- **整合推論**：先用 YOLO 框出狗，再把頭部區域裁切下來送進情緒模型，結果畫在圖片或影片上。

## 檔案說明

| 檔案 | 說明 |
| --- | --- |
| `animal_expression_predict_pytorch.py` | 訓練 ResNet50 情緒分類模型（80/20 切分、資料增強、100 epochs），驗證準確率最高時存成 `resnet_expression_model.pth` |
| `animal_expression_test_pytorch.py` | 載入訓練好的模型，對單張圖片預測情緒 |
| `emo_pose.py` | 批次處理資料夾中的圖片：姿態偵測 + 情緒辨識，輸出標註後的圖片 |
| `emo_pose_video.py` | 對影片逐格做姿態偵測、骨架繪製與情緒辨識，輸出 `output_pose.mp4` |

## 環境需求

```bash
pip install torch torchvision ultralytics opencv-python numpy scikit-learn tqdm
```

有 NVIDIA GPU 時會自動使用 CUDA。

## 資料與模型（未包含在 repo 中）

因為檔案太大，下列檔案沒有上傳，需要自行準備：

- **情緒資料集**：依情緒分成 `happy/`、`sad/`、`angry/`、`relaxed/` 四個資料夾的狗狗圖片。
- **姿態資料集**：狗狗姿態關鍵點資料集（dog-pose），用來訓練 YOLO pose 模型。
- **模型權重**：`resnet_expression_model.pth`（執行訓練腳本產生）、`pose_best.pt`（YOLO pose 訓練結果）。

## 使用方式

> 程式中的資料與權重路徑是寫死的絕對路徑，執行前請先改成自己電腦上的路徑。

1. 修改 `animal_expression_predict_pytorch.py` 裡的 `data_folders`，然後訓練情緒模型：
   ```bash
   python animal_expression_predict_pytorch.py
   ```
2. 測試單張圖片（修改 `img_path`）：
   ```bash
   python animal_expression_test_pytorch.py
   ```
3. 批次處理圖片（修改 `input_folder`、`output_folder`）：
   ```bash
   python emo_pose.py
   ```
4. 處理影片（修改影片路徑，執行時按 `q` 可提前結束）：
   ```bash
   python emo_pose_video.py
   ```
