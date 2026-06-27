import os
import cv2
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/split"

IMG_SIZE = (224, 224)

# Common class names
all_images = {
    "glioma": [],
    "meningioma": [],
    "pituitary": [],
    "no_tumor": []
}

# Folder name mapping
mapping = {
    "glioma": "glioma",
    "Glioma": "glioma",

    "meningioma": "meningioma",
    "Meningioma": "meningioma",

    "pituitary": "pituitary",
    "Pituitary": "pituitary",

    "notumor": "no_tumor",
    "No Tumor": "no_tumor",
    "no_tumor": "no_tumor"
}

# Scan all folders inside data/raw
for root, dirs, files in os.walk(RAW_DIR):

    folder_name = os.path.basename(root)

    if folder_name in mapping:

        class_name = mapping[folder_name]

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                img_path = os.path.join(
                    root,
                    file
                )

                all_images[class_name].append(
                    img_path
                )

# Create output folders
for split in ["train", "val", "test"]:
    for cls in all_images.keys():

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                cls
            ),
            exist_ok=True
        )

# Split and save
for cls, images in all_images.items():

    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=42
    )

    data = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split_name, img_list in data.items():

        for idx, path in enumerate(img_list):

            try:

                img = cv2.imread(path)

                if img is None:
                    continue

                img = cv2.resize(
                    img,
                    IMG_SIZE
                )

                save_path = os.path.join(
                    OUTPUT_DIR,
                    split_name,
                    cls,
                    f"{idx}.jpg"
                )

                cv2.imwrite(
                    save_path,
                    img
                )

            except Exception as e:
                print(e)

print("Preprocessing completed!")