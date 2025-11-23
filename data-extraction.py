import os
import json
from PIL import Image
from collections import Counter

DATA_ROOT = "./data/cub_200/CUB_200_2011"
IMG_DIR = os.path.join(DATA_ROOT, "images")
PARTS_DIR = os.path.join(DATA_ROOT, "parts")

OUT_DIR = "./data/beak_crops"
os.makedirs(OUT_DIR, exist_ok=True)

beak_part_id = 2

part_locs_path = os.path.join(PARTS_DIR, "part_locs.txt")

# part_locs.txt format:
# image_id part_id x y visibility
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

bbox_file = os.path.join(DATA_ROOT, "bounding_boxes.txt")

# Format:
# img_id x y width height
bboxes = {}

with open(bbox_file, "r") as f:
    for line in f:
        img_id, x, y, w, h = line.strip().split()
        img_id = int(img_id)
        bboxes[img_id] = (
            float(x),
            float(y),
            float(w),
            float(h)
        )

print("✔ Loaded bounding boxes")

images_file = os.path.join(DATA_ROOT, "images.txt")

# Format:
# img_id path/to/image.jpg
img_paths = {}

with open(images_file, "r") as f:
    for line in f:
        img_id, rel_path = line.strip().split()
        img_paths[int(img_id)] = os.path.join(IMG_DIR, rel_path)

print("✔ Loaded image file paths")

CROP_SCALE = 0.15   # fraction of bounding box width/height

saved = 0

beaks_file = "./data/beaks.txt"

beak_name_to_id = {}
with open(beaks_file, "r") as f:
    for line in f:
        beak_id, beak_name = line.strip().split()
        beak_name_to_id[beak_name] = beak_id

species_id_to_beak_file = "./data/species_id_to_beak.txt"

species_id_to_beak_id = {}
with open(species_id_to_beak_file, "r") as f:
    for line in f:
        species_id, beak_name = line.strip().split()
        species_id_to_beak_id[species_id] = beak_name_to_id[beak_name]

labels_file = "./data/labels.txt"
labels = {}
with open(labels_file, "w") as f:
    for img_id, (bx, by, bw, bh) in bboxes.items():

        if img_id not in beak_coords:
            continue  # beak not visible

        img_path = img_paths[img_id]
        if not os.path.exists(img_path):
            continue

        species_id = os.path.basename(os.path.dirname(img_path))[0:3]
        beak_id = species_id_to_beak_id[species_id]

        f.write(f"{img_path} {beak_id}\n")

        beak_x, beak_y = beak_coords[img_id]

        # Crop size proportional to bird bounding box
        crop_w = bw * CROP_SCALE
        crop_h = bh * CROP_SCALE

        left   = beak_x - crop_w / 2
        right  = beak_x + crop_w / 2
        top    = beak_y - crop_h / 2
        bottom = beak_y + crop_h / 2

        # Clamp to image boundaries
        img = Image.open(img_path)

        left   = max(0, left)
        top    = max(0, top)
        right  = min(img.width,  right)
        bottom = min(img.height, bottom)

        # Perform crop
        crop = img.crop((left, top, right, bottom))

        # Convert to grayscale
        crop = crop.convert("L")

        # Resize to 32×32
        crop = crop.resize((32, 32), Image.BICUBIC)

        # Save
        crop.save(os.path.join(OUT_DIR, f"{img_id}_beak.png"))
        saved += 1

print(f"✔ Done. Saved {saved} beak crops to {OUT_DIR}")


beak_id_to_name = {v:k for k,v in beak_name_to_id.items()}

counts = Counter()

with open("./data/labels.txt", "r") as f:
    for line in f:
        _, beak_id = line.strip().split()
        counts[beak_id] += 1

print("Images per class:")
for beak_id, n in sorted(counts.items()):
    print(f"{beak_id_to_name[beak_id]} (ID {beak_id}): {n} images")

