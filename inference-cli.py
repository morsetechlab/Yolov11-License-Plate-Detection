
import argparse
import os
import cv2
from ultralytics import YOLO

def run_inference(model_path, image_path, conf=0.25, imgsz=1280):
    # ตรวจสอบว่าไฟล์ภาพมีอยู่หรือไม่
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File {image_path} is not found")
    
    # โหลดภาพด้วย OpenCV
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # โหลดโมเดล YOLO
    model = YOLO(model_path)

    # ทำ Inference
    results = model.predict(
        source=image_rgb,
        conf=conf,
        imgsz=imgsz,
        save=False
    )

    # วาดกรอบบนภาพต้นฉบับ
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            label = r.names[int(box.cls[0])]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 10)
            cv2.putText(image_bgr, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # บันทึกผลลัพธ์
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "cli_inference_result.jpg")
    cv2.imwrite(output_path, image_bgr)
    print(f"output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 License Plate Detection CLI")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image input size")

    args = parser.parse_args()
    run_inference(args.model, args.source, args.conf, args.imgsz)
