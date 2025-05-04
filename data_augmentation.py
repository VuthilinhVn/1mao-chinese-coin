import os
import cv2
import numpy as np
from tqdm import tqdm
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Directories
image_dir = "E:/coin_count/Dataset/image"
label_dir = "E:/coin_count/Dataset/labels_fixed"
output_image_dir = "E:/coin_count/Dataset/images_aug"
output_label_dir = "E:/coin_count/Dataset/labels_aug"

# Create output directories
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Aggressive augmenter pipeline
augmenter = iaa.Sequential([
    iaa.Affine(rotate=(-45, 45), scale=(0.5, 1.5), shear=(-16, 16)),  # Extreme rotation and scaling
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),  # Add noise
    iaa.Multiply((0.5, 1.5)),  # Random brightness adjustment
    iaa.LinearContrast((0.5, 2.0)),  # Increase contrast
    iaa.MultiplyHue((0.5, 1.5)),  # Modify hue for color variation
    iaa.GaussianBlur(sigma=(0, 2.0)),  # Apply heavy Gaussian blur
])

def yolo_to_bbox(yolo_line, img_w, img_h):
    cls, x, y, w, h = map(float, yolo_line.strip().split())
    x1 = (x - w/2) * img_w
    y1 = (y - h/2) * img_h
    x2 = (x + w/2) * img_w
    y2 = (y + h/2) * img_h
    return int(cls), BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

def bbox_to_yolo(cls_id, bbox, img_w, img_h):
    x_center = (bbox.x1 + bbox.x2) / 2 / img_w
    y_center = (bbox.y1 + bbox.y2) / 2 / img_h
    width = (bbox.x2 - bbox.x1) / img_w
    height = (bbox.y2 - bbox.y1) / img_h
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# Main loop
print("🔁 Augmenting dataset with bounding boxes...")

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in tqdm(image_files):
    name_wo_ext = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, f"{name_wo_ext}.txt")

    if not os.path.exists(label_path):
        print(f"⚠️ Skipping {filename} — No label found.")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not read image {filename}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    with open(label_path, 'r') as f:
        label_lines = f.readlines()

    # Parse YOLO labels to imgaug bboxes
    bboxes = []
    classes = []
    for line in label_lines:
        cls_id, bbox = yolo_to_bbox(line, width, height)
        bboxes.append(bbox)
        classes.append(cls_id)

    bb_on_image = BoundingBoxesOnImage(bboxes, shape=image.shape)

    for i in range(25):
        aug_image, aug_bbs = augmenter(image=image, bounding_boxes=bb_on_image)
        aug_bbs = aug_bbs.remove_out_of_image().clip_out_of_image()

        aug_filename = f"{name_wo_ext}_aug{i}.jpg"
        aug_labelname = f"{name_wo_ext}_aug{i}.txt"

        aug_image_path = os.path.join(output_image_dir, aug_filename)
        aug_label_path = os.path.join(output_label_dir, aug_labelname)

        # Save augmented image
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aug_image_path, aug_image_bgr)

        # Save augmented labels
        with open(aug_label_path, 'w') as out_f:
            for cls_id, bbox in zip(classes, aug_bbs.bounding_boxes):
                yolo_line = bbox_to_yolo(cls_id, bbox, width, height)
                out_f.write(yolo_line + '\n')

print("✅ All images and labels augmented successfully.")