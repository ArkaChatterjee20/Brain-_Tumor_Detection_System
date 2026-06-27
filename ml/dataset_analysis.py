import os
import cv2

DATA_DIR = "data/split"

splits = ["train", "val", "test"]
classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

total_images = 0
corrupted_images = []

print("\n" + "=" * 50)
print("BRAIN TUMOR DATASET VERIFICATION")
print("=" * 50)

for split in splits:
    print(f"\n{split.upper()} DATASET")
    print("-" * 30)

    split_count = 0

    for cls in classes:
        class_path = os.path.join(DATA_DIR, split, cls)

        if not os.path.exists(class_path):
            print(f"{cls:<15}: Folder Not Found")
            continue

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        count = len(images)
        split_count += count

        print(f"{cls:<15}: {count}")

        # Check corrupted images
        for image in images:
            image_path = os.path.join(class_path, image)

            try:
                img = cv2.imread(image_path)

                if img is None:
                    corrupted_images.append(image_path)

            except Exception:
                corrupted_images.append(image_path)

    total_images += split_count

    print("-" * 30)
    print(f"Total {split} images: {split_count}")

print("\n" + "=" * 50)
print(f"TOTAL DATASET SIZE: {total_images}")
print("=" * 50)

# Corrupted image report
if corrupted_images:
    print("\nCORRUPTED IMAGES FOUND:")
    for img in corrupted_images:
        print(img)
else:
    print("\nNo corrupted images found.")