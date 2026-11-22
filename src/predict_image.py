from ultralytics import YOLO
import cv2
import time
import os
from datetime import datetime

# Load trained model
model = YOLO(
    r"models/best.pt"
)

save_dir = "inspection_results"
i = 0
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not opened")
else:
    print("Camera opened successfully")

# Create window
cv2.namedWindow("YOLO Defect Detection", cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    "YOLO Defect Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_NORMAL
)
defect_counts = { }
last_save_time = 0
save_interval = 5
while True:

    # Capture frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip webcam frame
    frame = cv2.flip(frame, 1)

    start_time = time.time()

    # YOLO prediction
    results = model.predict(
        source=frame,
        imgsz=256,
        conf=0.25,
        verbose=False
    )

    end_time = time.time()

    # Get result
    result = results[0]

    # Process detections
    a_f = result.plot()

    # Calculate FPS
    processing_time = end_time - start_time
    fps = 1 / processing_time

    # Inspection logic
    if len(result.boxes) > 0:
        inspection_status = "FAIL"
        colour = (0, 0, 255)  # Red
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{i+1}_defect_{timestamp}.jpg"
            filepath = os.path.join(save_dir,filename)
            cv2.imwrite(filepath,frame)
            i += 1
            last_save_time = current_time
            print("Defective image saved:", filepath)
    else:
        inspection_status = "PASS"
        colour = (0, 255, 0)  # Green

    # Display detected class names
    detected_defects = []

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        detected_defects.append(class_name)
        defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        print("Class Name:", class_name)
    
    y_position = 200
    for class_name,count in defect_counts.items():
        text = f"{class_name}: {count}"

        cv2.putText(
            a_f,
            text,
            (25,y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2
        )

        y_position += 40
    # Display inspection status
    cv2.putText(
        a_f,
        f"Inspection: {inspection_status}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        colour,
        3
    )

    # Display FPS
    cv2.putText(
        a_f,
        f"FPS: {fps:.2f}",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    # Display detected defects
    if len(detected_defects) > 0:

        defect_text = ", ".join(detected_defects)

        cv2.putText(
            a_f,
            f"Defects: {defect_text}",
            (25, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2
        )

    # Display webcam frame
    cv2.imshow("YOLO Defect Detection", a_f)

    # Press q or ESC to exit
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break

# Release camera
cap.release()

# Close windows
cv2.destroyAllWindows()