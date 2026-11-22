from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(
    r"runs\industrial_defect\yolo11n_neu-2\weights\best.pt"
)

# Load single image
image_path = r"dataset\yolo\images\validation\crazing_241.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# Run prediction
results = model.predict(
    source=image,
    imgsz=256,
    conf=0.50,
    verbose=False
)

result = results[0]

# Annotate image
annotated_image = result.plot()

# Inspection status
if len(result.boxes) > 0:
    print("FAIL - Defect detected")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(f"Defect: {class_name}")
        print(f"Confidence: {confidence:.2f}")

else:
    print("PASS - No defect detected")

# Display image
cv2.imshow("Single Image Defect Detection", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()