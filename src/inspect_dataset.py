import os
import xml.etree.ElementTree as ET
from collections import Counter

# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = r"C:\Users\swamy\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET"

# ============================================================
# DATASET DIRECTORIES
# ============================================================

TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images")
TRAIN_ANNOTATIONS = os.path.join(DATASET_PATH, "train", "annotations")

VAL_IMAGES = os.path.join(DATASET_PATH, "validation", "images")
VAL_ANNOTATIONS = os.path.join(DATASET_PATH, "validation", "annotations")


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(directory):

    total = 0

    for root, dirs, files in os.walk(directory):

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                total += 1

    return total


# ============================================================
# COUNT ANNOTATIONS
# ============================================================

def count_annotations(directory):

    return len([
        file
        for file in os.listdir(directory)
        if file.endswith(".xml")
    ])


# ============================================================
# EXTRACT DEFECT CLASSES
# ============================================================

def extract_classes(annotation_directory):

    class_counter = Counter()

    for file in os.listdir(annotation_directory):

        if file.endswith(".xml"):

            xml_path = os.path.join(annotation_directory, file)

            tree = ET.parse(xml_path)

            root = tree.getroot()

            for obj in root.findall("object"):

                class_name = obj.find("name").text

                class_counter[class_name] += 1

    return class_counter


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("INDUSTRIAL DEFECT DATASET INSPECTION")
print("=" * 60)


# Training dataset

train_images = count_images(TRAIN_IMAGES)

train_annotations = count_annotations(TRAIN_ANNOTATIONS)

train_classes = extract_classes(TRAIN_ANNOTATIONS)


print("\nTRAIN DATASET")

print("Images:", train_images)

print("Annotations:", train_annotations)

print("\nDefect Classes:")

for class_name, count in train_classes.items():

    print(class_name, ":", count)


# Validation dataset

val_images = count_images(VAL_IMAGES)

val_annotations = count_annotations(VAL_ANNOTATIONS)

val_classes = extract_classes(VAL_ANNOTATIONS)


print("\nVALIDATION DATASET")

print("Images:", val_images)

print("Annotations:", val_annotations)

print("\nDefect Classes:")

for class_name, count in val_classes.items():

    print(class_name, ":", count)


print("\n" + "=" * 60)