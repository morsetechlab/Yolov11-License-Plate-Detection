# 📘 [เวอร์ชั่นภาษาไทยที่นี่](README.md)

# YOLOv11-License-Plate Detection

This model is fine-tuned from various YOLOv11 versions (n, s, m, l, x) using a dataset from Roboflow Universe:  
[License Plate Recognition Dataset (10,125 images)](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11)  
to save time and cost for developers who need a fast and accurate license plate detection model.

## 🔥 YOLOv11-License-Plate Performance

| Model     | Precision | Recall  | mAP@50  | mAP@50-95 | Box Loss  | Class Loss  |
|-----------|-----------|---------|---------|-----------|-----------|-------------|
| YOLOv11n  | 0.9835    | 0.9505  | 0.9786  | 0.723     | 1.0300    | 0.3765      |
| YOLOv11s  | 0.9831    | 0.9524  | 0.9794  | 0.7285    | 1.0274    | 0.3576      |
| YOLOv11m  | 0.9831    | 0.9553  | 0.9805  | 0.7301    | 1.0295    | 0.3519      |
| YOLOv11l  | 0.9836    | 0.9608  | 0.9826  | 0.7307    | 1.0338    | 0.3481      |
| YOLOv11x  | 0.9893    | 0.9508  | 0.9813  | 0.7260    | 1.0364    | 0.3661      |

> Notes:  
> - **Precision**: Detection accuracy  
> - **Recall**: Detection coverage  
> - **mAP@50**: Mean Average Precision at IoU ≥ 50%  
> - **mAP@50-95**: Mean Average Precision across multiple IoUs  
> - **Box Loss / Class Loss**: Loss functions during training

**Recommended Use Cases:**

- `YOLOv11n` – Ultra lightweight, ideal for **Jetson Nano, Raspberry Pi, or CPU**
- `YOLOv11s` – Lightweight and accurate, suitable for **edge devices**
- `YOLOv11m` – Balanced, great for **PC/Server without GPU**
- `YOLOv11l` – High accuracy, best for **Cloud or GPU Desktop**
- `YOLOv11x` – Highest precision, suitable for **Cloud GPU only**

> `.pt` is for Python and Ultralytics CLI (`yolo task=detect`)  
> `.onnx` is for other inference systems such as OpenCV DNN, TensorRT, or ONNXRuntime

## Model

### PyTorch Format (.pt)
- [`lpr-finetune-v1n.pt`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1n.pt)
- [`lpr-finetune-v1s.pt`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1s.pt)
- [`lpr-finetune-v1m.pt`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1m.pt)
- [`lpr-finetune-v1l.pt`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1l.pt)
- [`lpr-finetune-v1x.pt`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1x.pt)

### ONNX Format (.onnx)
- [`lpr-finetune-v1n.onnx`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1n.onnx)
- [`lpr-finetune-v1s.onnx`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1s.onnx)
- [`lpr-finetune-v1m.onnx`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1m.onnx)
- [`lpr-finetune-v1l.onnx`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1l.onnx)
- [`lpr-finetune-v1x.onnx`](https://github.com/morsetechlab/yolov11-license-plate-detection/releases/download/v1.0.0/lpr-finetune-v1x.onnx)

## Inference
![CLI Inference](results/cli_inference_result.jpg)

<p align="center">
  <img src="output.gif" width="100%" />
</p>

### Detect on Command Line (`inference-cli.py`)
```bash
python inference-cli.py \
--model yolov11n-license-plate.pt \
--source examples/plate.jpg \
--conf 0.25 \
--imgsz 1280
```
> Detects license plate from image via CLI and saves to `output/cli_inference_result.jpg`

### Detect with PyTorch (Ultralytics Python API)
```python
# (Same content as provided previously)
```

### Detect with ONNX (ONNX Runtime)
```python
# (Same content as provided previously)
```

## Results

### PR Curve (Precision-Recall)

![PR Curve](results/PR_curve.png)

### Training Losses and mAP over time

![Training Results](results/results.png)

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Validation Batches

![val_batch0](results/val_batch0_pred.jpg)  
![val_batch1](results/val_batch1_pred.jpg)  
![val_batch2](results/val_batch2_pred.jpg)

## requirements.txt
```txt
ultralytics
onnxruntime
opencv-python
matplotlib
pandas
```

## ALPR Use with OCR for Reading License Plates

This model only **detects** the license plate region. For **reading characters**, pair it with OCR tools such as:
- EasyOCR
- Tesseract OCR
- PaddleOCR

## 💡 Real-World Applications
- Smart Parking Systems
- Tollgate / Access Control
- Traffic Surveillance Cameras
- License Plate-based Vehicle Tracking

## License
- **Dataset**: CC BY 4.0 from Roboflow Universe
- **Base Model (YOLOv11)**: AGPLv3 by Ultralytics
- **Fine-tuned Models**: AGPLv3 by MorseTechLab