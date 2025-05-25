import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

# — CONFIG —
WORDS_TXT  = Path('words.txt')         # input annotations
IMAGES_DIR = Path('images')            # contains train/, val/
LABELS_DIR = Path('labels')            # target labels root
SPLIT      = {'train': 0.8, 'val': 0.2}
CLASS_ID   = 0                          # single class index

# 1. Read and group entries by image ID
entries = defaultdict(list)
with open(WORDS_TXT, encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=6)
        if len(parts) < 6:
            continue
        img_id = parts[0]
        # parts: [img_id, status, x, y, w, h, text]
        x1, y1, w, h = map(int, parts[2:6])
        x2 = x1 + w
        y2 = y1 + h
        entries[img_id].append((x1, y1, x2, y2))

# 2. Split into train/val
all_ids = list(entries.keys())
cutoff  = int(len(all_ids) * SPLIT['train'])
train_ids = set(all_ids[:cutoff])

# 3. Write YOLO-format label files
for img_id, boxes in entries.items():
    subset = 'train' if img_id in train_ids else 'val'
    # find image file (any extension)
    img_files = list((IMAGES_DIR / subset).glob(f"{img_id}.*"))
    if not img_files:
        print(f"⚠️ Image not found for ID '{img_id}' in {subset}")
        continue
    img_path = img_files[0]
    W, H = Image.open(img_path).size

    # ensure output dir exists
    out_dir = LABELS_DIR / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{img_id}.txt"

    with open(txt_path, 'w', encoding='utf-8') as out:
        for x1, y1, x2, y2 in boxes:
            # normalize to YOLO format
            xc     = ((x1 + x2) / 2) / W
            yc     = ((y1 + y2) / 2) / H
            w_norm = (x2 - x1) / W
            h_norm = (y2 - y1) / H
            out.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("✅ Labels converted successfully.")
