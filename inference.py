
import os
import cv2
import numpy as np
from ultralytics import YOLO

# กำหนด path ของไฟล์ภาพ
image_dir = "images"
image_file = "0001.jpg"
image_path = os.path.join(image_dir, image_file)

# โหลดโมเดล YOLOv11 ที่ fine-tune แล้ว
model = YOLO("yolov11x-license-plate.pt")

# โหลดภาพด้วย OpenCV
original_image = cv2.imread(image_path)

# ตรวจสอบว่าภาพโหลดสำเร็จหรือไม่
if original_image is None:
    raise FileNotFoundError(f"File {image_path} is not found")

# แปลงจาก BGR เป็น RGB
image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# ทำ Inference
results = model.predict(
    source=image_rgb,
    conf=0.25,
    imgsz=1280,
    save=False,
    show=False
)

# วาดผลลัพธ์บนภาพต้นฉบับ
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        label = r.names[int(box.cls[0])]

        cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(original_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)

# แสดงผลลัพธ์
cv2.imshow("YOLOv11 License Plate Detection", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# บันทึกภาพผลลัพธ์
output_path = os.path.join("output", "inference_result.jpg")
os.makedirs("output", exist_ok=True)
cv2.imwrite(output_path, original_image)
print(f"output: {output_path}")