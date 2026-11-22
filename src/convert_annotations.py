import os
import xml.etree.ElementTree as ET
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\swamy\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET"

OUTPUT_PATH = r"dataset"

# Defect class mapping
CLASS_MAPPING = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
}

# ============================================================
# CONVERSION FUNCTION
# ============================================================

def convert_bbox(size, bbox):

    image_width, image_height = size

    xmin, ymin, xmax, ymax = bbox

    # Calculate bounding box center
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    # Calculate bounding box width and height
    box_width = xmax - xmin
    box_height = ymax - ymin

    # Normalize coordinates
    center_x /= image_width
    center_y /= image_height

    box_width /= image_width
    box_height /= image_height

    return center_x, center_y, box_width, box_height


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(split):

    print(f"\nProcessing {split.upper()} dataset...")

    annotation_dir = os.path.join(
        DATASET_PATH,
        split,
        "annotations"
    )

    image_dir = os.path.join(
        DATASET_PATH,
        split,
        "images"
    )

    output_dir = os.path.join(
        OUTPUT_PATH,
        "labels",
        split
    )

    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    missing_images = 0
    invalid_annotations = 0

    # Process every XML annotation
    for xml_file in os.listdir(annotation_dir):

        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotation_dir, xml_file)

        # XML filename -> image filename
        image_filename = os.path.splitext(xml_file)[0] + ".jpg"

        # Search image inside class directories
        image_path = None

        for root, dirs, files in os.walk(image_dir):

            if image_filename in files:

                image_path = os.path.join(
                    root,
                    image_filename
                )

                break

        # Check image existence
        if image_path is None:

            print("Missing image:", image_filename)

            missing_images += 1

            continue

        # Read image
        image = cv2.imread(image_path)

        if image is None:

            print("Unable to read image:", image_filename)

            missing_images += 1

            continue

        image_height, image_width = image.shape[:2]

        # Parse XML
        tree = ET.parse(xml_path)

        root = tree.getroot()

        yolo_labels = []

        # Process every defect object
        for obj in root.findall("object"):

            class_name = obj.find("name").text

            # Check class
            if class_name not in CLASS_MAPPING:

                print("Unknown class:", class_name)

                invalid_annotations += 1

                continue

            class_id = CLASS_MAPPING[class_name]

            # Extract bounding box
            bbox = obj.find("bndbox")

            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Validate bounding box
            if xmax <= xmin or ymax <= ymin:

                print("Invalid bbox:", xml_file)

                invalid_annotations += 1

                continue

            # Convert to YOLO format
            center_x, center_y, box_width, box_height = convert_bbox(
                (image_width, image_height),
                (xmin, ymin, xmax, ymax)
            )

            # Create YOLO label
            label = (
                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            yolo_labels.append(label)

        # Save labels
        output_filename = os.path.splitext(xml_file)[0] + ".txt"

        output_file = os.path.join(
            output_dir,
            output_filename
        )

        with open(output_file, "w") as file:

            file.write("\n".join(yolo_labels))

        converted += 1

    print(f"Converted annotations: {converted}")
    print(f"Missing images: {missing_images}")
    print(f"Invalid annotations: {invalid_annotations}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset("train")

    process_dataset("validation")

    print("\nAnnotation conversion completed!")