from ultralytics import YOLO
import cv2
import time
import os
from datetime import datetime

# Import Telegram function
from telegram_alert import send_telegram_image

# Load trained model
model = YOLO("models/best.pt")

# Image saving directory
save_dir = "inspection_results"
os.makedirs(save_dir, exist_ok=True)

# Image counter
i = 0

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not opened")
    exit()

print("Camera opened successfully")

# Create window
cv2.namedWindow("YOLO Defect Detection", cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    "YOLO Defect Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_NORMAL
)

# Telegram notification cooldown
last_save_time = 0
save_interval = 5  # seconds

while True:

    # Capture frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip webcam frame
    frame = cv2.flip(frame, 1)

    # Start processing time
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

    # Draw bounding boxes
    a_f = result.plot()

    # Calculate FPS
    processing_time = end_time - start_time

    fps = 1 / processing_time if processing_time > 0 else 0

    # Reset defect counts for current frame
    defect_counts = {}

    detected_defects = []

    # Process detected objects
    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        detected_defects.append(class_name)

        defect_counts[class_name] = (
            defect_counts.get(class_name, 0) + 1
        )

    # Inspection logic
    if len(result.boxes) > 0:

        inspection_status = "FAIL"

        colour = (0, 0, 255)

        current_time = time.time()

        # Save and send image every 5 seconds
        if current_time - last_save_time >= save_interval:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = f"{i+1}_defect_{timestamp}.jpg"

            filepath = os.path.join(
                save_dir,
                filename
            )

            # Save annotated frame
            cv2.imwrite(filepath, a_f)

            i += 1

            last_save_time = current_time

            print("Defective image saved:", filepath)

            # Prepare defect summary
            defect_summary = ", ".join(
                f"{name}: {count}"
                for name, count in defect_counts.items()
            )

            # Telegram caption
            caption = (
                "🚨 INDUSTRIAL DEFECT DETECTED 🚨\n\n"
                f"Inspection Status: FAIL\n"
                f"Detected Defects: {defect_summary}\n"
                f"Timestamp: {timestamp}\n"
                f"Image: {filename}"
            )

            # Send image to Telegram
            send_telegram_image(
                filepath,
                caption
            )

    else:

        inspection_status = "PASS"

        colour = (0, 255, 0)

    # Display defect counts
    y_position = 200

    for class_name, count in defect_counts.items():

        text = f"{class_name}: {count}"

        cv2.putText(
            a_f,
            text,
            (25, y_position),
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
        (255, 180, 0),
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
    cv2.imshow(
        "YOLO Defect Detection",
        a_f
    )

    # Press q or ESC to exit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

# Release camera
cap.release()

# Close windows
cv2.destroyAllWindows()

print("Inspection system stopped.")