import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "./raw_data"
OUTPUT_DIR = "./dataset"

# Create destination directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def mask_to_yolo_bbox(mask_path, img_w, img_h):
    """Extracts YOLO bounding box from binary mask image."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    # Threshold mask to ensure binary format
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:  # Filter out tiny noise
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        
        bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    return bboxes

# Collect matching image and mask file pairs
images_dir = os.path.join(RAW_DIR, "images")
masks_dir = os.path.join(RAW_DIR, "masks")

# If files are in a flat directory, adapt path discovery
valid_pairs = []
if os.path.exists(images_dir) and os.path.exists(masks_dir):
    for img_name in os.listdir(images_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_name)[0]
            mask_name = f"{base_name}.png"  # Adjust mask extension if needed
            mask_path = os.path.join(masks_dir, mask_name)
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))

# Fallback: Find image files directly if directory structure differs
if not valid_pairs:
    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(RAW_DIR) for f in filenames]
    img_files = [f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg')) and 'mask' not in f.lower()]
    for img_path in img_files:
        mask_path = img_path.replace('images', 'masks')
        if os.path.exists(mask_path):
            valid_pairs.append((img_path, mask_path))

print(f"Found {len(valid_pairs)} total valid image-mask pairs.")

# Train/Val Split (80% train, 20% val)
train_pairs, val_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)

def process_and_save(pairs, split):
    for img_path, mask_path in pairs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        
        bboxes = mask_to_yolo_bbox(mask_path, w, h)
        if not bboxes:  # Skip unannotated images
            continue
            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Copy image
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, f"{base_name}.jpg")
        cv2.imwrite(dst_img_path, img)
        
        # Write label TXT file
        dst_lbl_path = os.path.join(OUTPUT_DIR, 'labels', split, f"{base_name}.txt")
        with open(dst_lbl_path, 'w') as f:
            f.write("\n".join(bboxes))

process_and_save(train_pairs, 'train')
process_and_save(val_pairs, 'val')
print("Dataset conversion and splitting completed successfully.")