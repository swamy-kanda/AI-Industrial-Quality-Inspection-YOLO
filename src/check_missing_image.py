import os

DATASET_PATH = r"C:\Users\swamy\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET"

IMAGE_DIR = os.path.join(
    DATASET_PATH,
    "validation",
    "images"
)

ANNOTATION_DIR = os.path.join(
    DATASET_PATH,
    "validation",
    "annotations"
)

target = "crazing_240"

print("Searching for image...")

for root, dirs, files in os.walk(IMAGE_DIR):

    for file in files:

        if target in file.lower():

            print("Found image:", os.path.join(root, file))

print("\nSearching for annotation...")

for file in os.listdir(ANNOTATION_DIR):

    if target in file.lower():

        print("Found XML:", file)