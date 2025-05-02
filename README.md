
# YOLOv11-License-Plate Detection (โมเดลตรวจจับป้ายทะเบียน)

โมเดลนี้ถูก Fine-Tune มาจาก YOLOv11 รุ่นต่าง ๆ (n, s, m, l, x) โดยใช้ Dataset จาก Roboflow Universe:  
[License Plate Recognition Dataset (10,125 ภาพ)](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11)  
เพื่อช่วยลดเวลาและต้นทุนในการฝึกโมเดลสำหรับผู้ที่ต้องการใช้งานระบบตรวจจับแผ่นป้ายทะเบียนแบบรวดเร็วและแม่นยำ

## 🔥 ประสิทธิภาพของ YOLOv11-License-Plate

| โมเดล     | Precision | Recall  | mAP@50  | mAP@50-95 | Box Loss  | Class Loss  |
|-----------|-----------|---------|---------|-----------|-----------|-------------|
| YOLOv11n  | 0.9835    | 0.9505  | 0.9786  | 0.723     | 1.0300    | 0.3765      |
| YOLOv11s  | 0.9831    | 0.9524  | 0.9794  | 0.7285    | 1.0274    | 0.3576      |
| YOLOv11m  | 0.9831    | 0.9553  | 0.9805  | 0.7301    | 1.0295    | 0.3519      |
| YOLOv11l  | 0.9836    | 0.9608  | 0.9826  | 0.7307    | 1.0338    | 0.3481      |
| YOLOv11x  | 0.9893    | 0.9508  | 0.9813  | 0.7260    | 1.0364    | 0.3661      |

> หมายเหตุ:  
> - **Precision**: ความแม่นยำในการตรวจจับ  
> - **Recall**: ความครอบคลุมของการตรวจจับ  
> - **mAP@50**: ค่าความแม่นยำเฉลี่ยเมื่อ IoU ≥ 50%  
> - **mAP@50-95**: ค่าความแม่นยำเฉลี่ยทุกระดับ IoU  
> - **Box Loss / Class Loss**: ความสูญเสียระหว่างการเรียนรู้ (loss functions)

**คำแนะนำการใช้งาน:**

- `YOLOv11n` – เบาสุด เหมาะกับ **Jetson Nano, Raspberry Pi, CPU**
- `YOLOv11s` – เบา + แม่นดี ใช้ได้กับ **อุปกรณ์ Edge**
- `YOLOv11m` – สมดุล เหมาะกับ **PC/Server ที่ไม่ใช้ GPU**
- `YOLOv11l` – แม่นสุด เหมาะกับ **Cloud หรือ GPU Desktop**
- `YOLOv11x` – แม่น + Precision สูง เหมาะกับ **Cloud GPU เท่านั้น**

> `.pt` สำหรับใช้กับ Python และ CLI ของ Ultralytics (เช่น `yolo task=detect`) 
> `.onnx` สำหรับใช้งานในระบบ inference อื่น เช่น OpenCV DNN, TensorRT, ONNXRuntime

## Model

### PyTorch Format (.pt)
- [`yolov11n-license-plate.pt`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11n-license-plate.pt)
- [`yolov11s-license-plate.pt`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11s-license-plate.pt)
- [`yolov11m-license-plate.pt`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11m-license-plate.pt)
- [`yolov11l-license-plate.pt`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11l-license-plate.pt)
- [`yolov11x-license-plate.pt`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11x-license-plate.pt)


### ONNX Format (.onnx)
- [`yolov11n-license-plate.onnx`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11n-license-plate.onnx)
- [`yolov11s-license-plate.onnx`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11s-license-plate.onnx)
- [`yolov11m-license-plate.onnx`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11m-license-plate.onnx)
- [`yolov11l-license-plate.onnx`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11l-license-plate.onnx)
- [`yolov11x-license-plate.onnx`](https://github.com/morsetechlab/yolov11-license-plate/releases/download/v1.0/yolov11x-license-plate.onnx)

## ทดสอบการใช้งาน (Inference)
<!-- Image path=results/PR_curve.png -->

### Detect on image using CLI:
```bash
yolo task=detect \
mode=predict \
model=yolov11n-license-plate.pt \
conf=0.25 \
imgsz=1280 \
line_thickness=1 \
max_det=1000 \
source=examples/plate.jpg
```

### Results:

**PR curve** (Precision-Recall)
![PR Curve](results/PR_curve.png)

**Losses และค่า mAP** ตลอดการฝึก
![Training Results](results/results.png)

**Confusion matrix**
![Confusion Matrix](results/confusion_matrix.png)

**Validation Batch**
![val_batch0](results/val_batch0_pred.jpg)
![val_batch1](results/val_batch1_pred.jpg)
![val_batch2](results/val_batch2_pred.jpg)

## 📚 requirements.txt
```txt
ultralytics
opencv-python
matplotlib
pandas
```

## 🔤 การรวม OCR (อ่านป้ายทะเบียน)

โมเดลนี้ทำหน้าที่ “ตรวจจับ” ตำแหน่งป้ายทะเบียนเท่านั้น หากต้องการ “อ่านตัวอักษร” บนป้าย ให้ใช้ OCR ร่วม เช่น:
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## 💡 ตัวอย่างการใช้งาน
- ระบบจอดรถอัจฉริยะ (Smart Parking)
- ระบบประตูอัตโนมัติ (Tollgate / Access Control)
- กล้องตรวจสอบการจราจร
- ระบบติดตามรถตามหมายเลขทะเบียน

## 📜 License
- **ชุดข้อมูล**: CC BY 4.0 จาก [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
- **โมเดลต้นทาง (YOLOv11)**: AGPLv3 โดย [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Fine-tuned**: AGPLv3 โดย [MorseTechLab](https://www.morsetechlab.com)
