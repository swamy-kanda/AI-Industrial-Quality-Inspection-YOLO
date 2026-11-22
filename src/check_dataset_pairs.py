import os

DATASET_PATH = r"C:\Users\swamy\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET"

for split in ["train", "validation"]:

    image_dir = os.path.join(DATASET_PATH, split, "images")
    annotation_dir = os.path.join(DATASET_PATH, split, "annotations")

    image_files = set()

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.add(os.path.splitext(file)[0])

    annotation_files = set()

    for file in os.listdir(annotation_dir):
        if file.endswith(".xml"):
            annotation_files.add(os.path.splitext(file)[0])

    images_without_annotations = image_files - annotation_files
    annotations_without_images = annotation_files - image_files

    print(f"\n{split.upper()} DATASET")
    print("-" * 40)

    print("Images without annotations:", len(images_without_annotations))

    for filename in sorted(images_without_annotations):
        print(filename)

    print("\nAnnotations without images:", len(annotations_without_images))

    for filename in sorted(annotations_without_images):
        print(filename)