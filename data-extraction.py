import os
import json
import random
from collections import defaultdict
from PIL import Image, ImageOps

# ----------------- Paths -----------------
DATA_ROOT = "./data/cub_200/CUB_200_2011"
IMG_DIR = os.path.join(DATA_ROOT, "images")
PARTS_DIR = os.path.join(DATA_ROOT, "parts")

OUT_DIR = "./data/beak_crops"
os.makedirs(OUT_DIR, exist_ok=True)

beak_part_id = 2
CROP_SCALE = 0.15   # size of crop relative to bounding box
MAX_PER_CLASS = 250  # new maximum after augmentation

part_locs_path = os.path.join(PARTS_DIR, "part_locs.txt")
bbox_file = os.path.join(DATA_ROOT, "bounding_boxes.txt")
images_file = os.path.join(DATA_ROOT, "images.txt")
beaks_file = "./data/beaks.txt"
species_id_to_beak_file = "./data/species_id_to_beak.txt"

# ----------------- Load Beak Coordinates -----------------
beak_coords = {}  # img_id -> (x, y)

with open(part_locs_path, "r") as f:
    for line in f:
        img_id, part_id, x, y, vis = line.strip().split()
        img_id = int(img_id)
        part_id = int(part_id)
        vis = int(vis)

        if part_id == beak_part_id and vis == 1:
            beak_coords[img_id] = (float(x), float(y))

print(f"✔ Loaded beak coordinates for {len(beak_coords)} images")

# ----------------- Load Bounding Boxes -----------------
bboxes = {}  # img_id -> (x, y, w, h)

with open(bbox_file, "r") as f:
    for line in f:
        img_id, x, y, w, h = line.strip().split()
        bboxes[int(img_id)] = (float(x), float(y), float(w), float(h))

print("✔ Loaded bounding boxes")

# ----------------- Load Image Paths -----------------
img_paths = {}

with open(images_file, "r") as f:
    for line in f:
        img_id, rel_path = line.strip().split()
        img_paths[int(img_id)] = os.path.join(IMG_DIR, rel_path)

print("✔ Loaded image file paths")

# ----------------- Load Beak Class Mappings -----------------
beak_name_to_id = {}  # e.g. "hooked" -> 0
with open(beaks_file, "r") as f:
    for line in f:
        beak_id, beak_name = line.strip().split()
        beak_name_to_id[beak_name] = beak_id

species_id_to_beak_id = {}  # species -> beak class
with open(species_id_to_beak_file, "r") as f:
    for line in f:
        species_id, beak_name = line.strip().split()
        species_id_to_beak_id[species_id] = beak_name_to_id[beak_name]

# ----------------- Organize Images per Beak Class -----------------
class_to_images = defaultdict(list)

for img_id, (bx, by, bw, bh) in bboxes.items():

    if img_id not in beak_coords:
        continue

    img_path = img_paths[img_id]
    if not os.path.exists(img_path):
        continue

    species_id = os.path.basename(os.path.dirname(img_path))[0:3]
    beak_id = species_id_to_beak_id[species_id]

    class_to_images[beak_id].append((img_id, img_path, species_id))

# ----------------- Select, Augment, and Save Crops -----------------
total_saved = 0

for beak_id, images in class_to_images.items():

    # Ensure output folder exists for this class
    class_dir = os.path.join(OUT_DIR, beak_id)
    os.makedirs(class_dir, exist_ok=True)

    # Shuffle → ensures species are mixed
    random.shuffle(images)

    # Initial selection (keep MAX_PER_CLASS from original images)
    selected = images[:MAX_PER_CLASS]

    saved_images = []

    for img_id, img_path, species_id in selected:
        beak_x, beak_y = beak_coords[img_id]
        bx, by, bw, bh = bboxes[img_id]

        crop_w = bw * CROP_SCALE
        crop_h = bh * CROP_SCALE

        with Image.open(img_path) as im:
            left   = max(0, beak_x - crop_w / 2)
            right  = min(im.width, beak_x + crop_w / 2)
            top    = max(0, beak_y - crop_h / 2)
            bottom = min(im.height, beak_y + crop_h / 2)

            crop = im.crop((left, top, right, bottom))
            crop = crop.convert("L").resize((64, 64), Image.BICUBIC)

            out_path = os.path.join(class_dir, f"{img_id}_beak.png")
            crop.save(out_path, icc_profile=None)
            saved_images.append(crop)
            total_saved += 1

    # Augmentation if fewer than 200 images
    # num_to_augment = min(MAX_PER_CLASS - len(saved_images), len(saved_images))
    # aug_idx = 0
    #
    # while len(saved_images) < MAX_PER_CLASS:
    #     img = saved_images[aug_idx % len(saved_images)]
    #     aug_img = img.copy()
    #
    #     # Random flip
    #     if random.random() < 0.5:
    #         aug_img = ImageOps.mirror(aug_img)
    #     if random.random() < 0.5:
    #         aug_img = ImageOps.flip(aug_img)
    #
    #     # Random rotation
    #     angle = random.uniform(-25, 25)
    #     aug_img = aug_img.rotate(angle, resample=Image.BICUBIC)
    #
    #     out_path = os.path.join(class_dir, f"{MAX_PER_CLASS + len(saved_images)}_beak_aug.png")
    #     aug_img.save(out_path, icc_profile=None)
    #     saved_images.append(aug_img)
    #     total_saved += 1
    #     aug_idx += 1

print(f"✔ Done. Saved {total_saved} beak crops to {OUT_DIR}")

# ----------------- Print Counts Per Class -----------------
print("\nImages per class:")
for beak_id in sorted(class_to_images.keys()):
    n = len(os.listdir(os.path.join(OUT_DIR, beak_id)))
    print(f"Class {beak_id}: {n} images")


beak_id_to_name = {v: k for k, v in beak_name_to_id.items()}

print("\nImages per class:")
for beak_id in sorted(class_to_images.keys()):
    class_dir = os.path.join(OUT_DIR, beak_id)
    n = len(os.listdir(class_dir))
    class_name = beak_id_to_name.get(beak_id, "Unknown")
    print(f"Class {beak_id} ({class_name}): {n} images")