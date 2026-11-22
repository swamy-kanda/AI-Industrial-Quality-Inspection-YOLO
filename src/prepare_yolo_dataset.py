import os
import shutil

# Original dataset
SOURCE_PATH = r"C:\Users\swamy\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET"

# Converted YOLO labels
LABEL_PATH = "dataset/labels"

# Final dataset
OUTPUT_PATH = "dataset/yolo"

def prepare_split(split):

    print(f"\nPreparing {split.upper()} dataset...")

    image_source = os.path.join(
        SOURCE_PATH,
        split,
        "images"
    )

    label_source = os.path.join(
        LABEL_PATH,
        split
    )

    image_output = os.path.join(
        OUTPUT_PATH,
        "images",
        split
    )

    label_output = os.path.join(
        OUTPUT_PATH,
        "labels",
        split
    )

    os.makedirs(image_output, exist_ok=True)
    os.makedirs(label_output, exist_ok=True)

    copied = 0
    skipped = 0

    # Find images recursively
    for root, dirs, files in os.walk(image_source):

        for image_file in files:

            if not image_file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_name = os.path.splitext(image_file)[0]

            image_path = os.path.join(root, image_file)

            label_file = image_name + ".txt"

            label_path = os.path.join(
                label_source,
                label_file
            )

            # Ensure matching label exists
            if not os.path.exists(label_path):

                print("Skipping unmatched image:", image_file)

                skipped += 1

                continue

            # Copy image
            shutil.copy2(
                image_path,
                os.path.join(image_output, image_file)
            )

            # Copy corresponding label
            shutil.copy2(
                label_path,
                os.path.join(label_output, label_file)
            )

            copied += 1

    print("Valid image-label pairs:", copied)
    print("Skipped images:", skipped)


if __name__ == "__main__":

    prepare_split("train")

    prepare_split("validation")

    print("\nYOLO dataset preparation completed!")