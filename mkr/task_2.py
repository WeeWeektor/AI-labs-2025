import cv2
import yaml
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A

# Налаштування
SRC_ROOT = Path("/kaggle/input/microsoft-logos")
DST_ROOT = Path("/kaggle/working/microsoft_aug_v2")

IMG_FORMATS = [".jpg", ".jpeg", ".png"]
TRAIN_RATIO = 0.8
AUG_PER_IMAGE = 7
SEED = 42
MIN_YOLO_AREA = 0.0005

random.seed(SEED)
np.random.seed(SEED)

# Шляхи
SRC_IMG_DIR = SRC_ROOT / "images"
SRC_LBL_DIR = SRC_ROOT / "labels"

if not SRC_IMG_DIR.exists():
    raise FileNotFoundError(f"Каталог з зображеннями не знайдено: {SRC_IMG_DIR}")

if not SRC_LBL_DIR.exists():
    raise FileNotFoundError(f"Каталог з анотаціями не знайдено: {SRC_LBL_DIR}")

for subset in ("train", "val"):
    (DST_ROOT / subset / "images").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / subset / "labels").mkdir(parents=True, exist_ok=True)

# Аугментації
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(20, 30, 20, p=0.5),
        A.ShiftScaleRotate(0.05, 0.15, 25, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.35),
        A.MotionBlur(blur_limit=7, p=0.25),
        A.CLAHE(clip_limit=3.0, p=0.25),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        min_visibility=0.3,
        min_area=1,
        label_fields=["labels"]
    )
)

# Функції роботи з YOLO
def load_annotation(txt_path: Path):
    """Зчитування YOLO-боксів."""
    if not txt_path.exists():
        return [], []

    bboxes, labels = [], []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            bboxes.append([x, y, w, h])
            labels.append(int(cls))
    return bboxes, labels


def save_annotation(path: Path, bbs, classes):
    """Запис у форматі YOLO."""
    with open(path, "w") as f:
        for c, (x, y, w, h) in zip(classes, bbs):
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def is_valid_box(bb):
    """Перевірка коректності YOLO боксу."""
    x, y, w, h = bb

    if not (0 < w <= 1 and 0 < h <= 1):
        return False
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return False

    if x - w / 2 < 0 or x + w / 2 > 1:
        return False
    if y - h / 2 < 0 or y + h / 2 > 1:
        return False

    return True


def box_area(bb):
    return bb[2] * bb[3]


# Основна логіка обробки
all_images = [p for p in SRC_IMG_DIR.iterdir() if p.suffix.lower() in IMG_FORMATS]

train_set, val_set = train_test_split(
    all_images, train_size=TRAIN_RATIO, random_state=SEED
)

print(f"Train: {len(train_set)}, Val: {len(val_set)}")


def process_split(file_list, split_name, do_aug):
    saved = 0

    img_out = DST_ROOT / split_name / "images"
    lbl_out = DST_ROOT / split_name / "labels"

    for img_path in file_list:
        base = img_path.stem
        lbl_path = SRC_LBL_DIR / f"{base}.txt"

        original = cv2.imread(str(img_path))
        if original is None:
            print(f"Помилка читання зображення: {img_path}")
            continue

        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        bbs, cls = load_annotation(lbl_path)
        if not bbs:
            continue

        # збереження оригіналу
        cv2.imwrite(str(img_out / f"{base}.jpg"), original)
        save_annotation(lbl_out / f"{base}.txt", bbs, cls)
        saved += 1

        # аугментації лише для train
        if do_aug:
            for i in range(AUG_PER_IMAGE):
                try:
                    aug = transform(image=rgb, bboxes=bbs, labels=cls)

                    new_bbs, new_cls = [], []
                    for bb, lb in zip(aug["bboxes"], aug["labels"]):
                        if is_valid_box(bb) and box_area(bb) >= MIN_YOLO_AREA:
                            new_bbs.append(bb)
                            new_cls.append(lb)

                    if not new_bbs:
                        continue

                    aug_name = f"{base}_aug{i}.jpg"
                    aug_img_bgr = cv2.cvtColor(aug["image"], cv2.COLOR_RGB2BGR)

                    cv2.imwrite(str(img_out / aug_name), aug_img_bgr)
                    save_annotation(lbl_out / f"{base}_aug{i}.txt", new_bbs, new_cls)
                    saved += 1

                except Exception as e:
                    print(f"Аугментація помилка ({base}_aug{i}): {e}")

    return saved


train_count = process_split(train_set, "train", True)
val_count = process_split(val_set, "val", False)

print(f"Train saved: {train_count}")
print(f"Val saved: {val_count}")
print(f"Total saved: {train_count + val_count}")

# Генерація data.yaml
unique_classes = set()

for subset in ("train", "val"):
    for lbl in (DST_ROOT / subset / "labels").glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                cls = int(float(line.split()[0]))
                unique_classes.add(cls)

names = [f"class_{i}" for i in sorted(unique_classes)]

yaml_dict = {
    "path": str(DST_ROOT.resolve()),
    "train": "train/images",
    "val": "val/images",
    "nc": len(names),
    "names": names,
}

with open(DST_ROOT / "data.yaml", "w") as f:
    yaml.dump(yaml_dict, f)

print("Файл data.yaml створено.")
