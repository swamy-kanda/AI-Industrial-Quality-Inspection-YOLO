import os
import cv2
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Dataset location
DATASET_PATH = r"C:\Users\swamy\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET"

IMAGE_DIR = os.path.join(DATASET_PATH, "train", "images")
ANNOTATION_DIR = os.path.join(DATASET_PATH, "train", "annotations")

# Get image files
image_files = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(os.path.join(root, file))

# Select random images
random_images = random.sample(image_files, 6)

# Create visualization
plt.figure(figsize=(15, 10))

for index, image_path in enumerate(random_images):

    # Load image
    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image filename
    filename = os.path.basename(image_path)

    # Find corresponding XML file
    xml_filename = os.path.splitext(filename)[0] + ".xml"

    xml_path = os.path.join(ANNOTATION_DIR, xml_filename)

    # Parse XML annotation
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Draw bounding boxes
    for obj in root.findall("object"):

        class_name = obj.find("name").text

        bbox = obj.find("bndbox")

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # Draw rectangle
        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        # Draw class label
        cv2.putText(
            image,
            class_name,
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    # Display image
    plt.subplot(2, 3, index + 1)
    plt.imshow(image)
    plt.title(filename)
    plt.axis("off")

plt.tight_layout()
plt.show()