# YOLOv11-License-Plate Detection

This repository contains fine-tuned models based on YOLOv11 (n, s, m, l, x) using a dataset from Roboflow Universe:  
[License Plate Recognition Dataset (10,125 images)](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11)

The purpose is to save time and resources for developers who need a ready-to-use license plate detection model — especially since the `license plate` class is not typically included in general detection datasets.

## 🔥 Model Performance

| Model     | Precision | Recall  | mAP@50  | mAP@50-95 | Box Loss  | Class Loss  |
|-----------|-----------|---------|---------|-----------|-----------|-------------|
| YOLOv11n  | 0.9835    | 0.9505  | 0.9786  | 0.723     | 1.0300    | 0.3765      |
| YOLOv11s  | 0.9831    | 0.9524  | 0.9794  | 0.7285    | 1.0274    | 0.3576      |
| YOLOv11m  | 0.9831    | 0.9553  | 0.9805  | 0.7301    | 1.0295    | 0.3519      |
| YOLOv11l  | 0.9836    | 0.9608  | 0.9826  | 0.7307    | 1.0338    | 0.3481      |
| YOLOv11x  | 0.9893    | 0.9508  | 0.9813  | 0.7260    | 1.0364    | 0.3661      |

> **Notes**:
> - **Precision**: Accuracy of detection  
> - **Recall**: Coverage of detection  
> - **mAP@50**: Mean average precision with IoU ≥ 50%  
> - **mAP@50-95**: Mean average precision across multiple IoU thresholds  
> - **Box Loss / Class Loss**: Loss values during training

### Recommended Usage

- `YOLOv11n` – Lightest model, ideal for **Jetson Nano, Raspberry Pi, CPU**
- `YOLOv11s` – Lightweight and accurate for **edge devices**
- `YOLOv11m` – Balanced for **non-GPU PC/Server**
- `YOLOv11l` – Most accurate, ideal for **Cloud GPU / GPU Desktop**
- `YOLOv11x` – Highest precision, **GPU cloud only**

> `.pt` is for Python/Ultralytics CLI  
> `.onnx` is for inference engines like ONNX Runtime, OpenCV DNN, TensorRT

## Model Downloads

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

## License
- **Dataset**: CC BY 4.0 from [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
- **Base Model (YOLOv11)**: AGPLv3 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Fine-tuned Model**: AGPLv3 by [MorseTechLab](https://www.morsetechlab.com)
