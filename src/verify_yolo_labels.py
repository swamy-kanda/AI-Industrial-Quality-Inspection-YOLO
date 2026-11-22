import os

LABEL_PATH = r"dataset\labels"

CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches"
}

for split in ["train", "validation"]:

    label_dir = os.path.join(LABEL_PATH, split)

    label_files = [
        file for file in os.listdir(label_dir)
        if file.endswith(".txt")
    ]

    print(f"\n{split.upper()} DATASET")
    print("Label files:", len(label_files))

    for file in label_files[:3]:

        file_path = os.path.join(label_dir, file)

        print("\nFile:", file)

        with open(file_path, "r") as f:

            for line in f:

                values = line.strip().split()

                class_id = int(values[0])

                print(
                    "Class:",
                    CLASSES[class_id],
                    "| YOLO coordinates:",
                    values[1:]
                )