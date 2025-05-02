
import os
import cv2
from ultralytics import YOLO

# กำหนด path วิดีโอ
video_path = os.path.join("videos", "video.mp4")

# ตรวจสอบว่าไฟล์วิดีโอมีอยู่หรือไม่
if not os.path.exists(video_path):
    raise FileNotFoundError(f"File {video_path} is not found")

# โหลดโมเดล
model = YOLO("yolov11x-license-plate.pt")

# เปิดวิดีโอ
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# เตรียมไฟล์ output
os.makedirs("output", exist_ok=True)
output_path = os.path.join("output", "video_inference_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ประมวลผลทีละเฟรม
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็น RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ทำ Inference
    results = model.predict(
        source=frame_rgb,
        conf=0.6,
        imgsz=1280,
        save=False,
        show=False
    )

    # วาดผลลัพธ์ลงบนเฟรม
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            label = r.names[int(box.cls[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 3)
            
    cv2.imshow("video_inference_output", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    # เขียนเฟรมลงไฟล์
    out.write(frame)

# ปิดวิดีโอ
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"output: {output_path}")
