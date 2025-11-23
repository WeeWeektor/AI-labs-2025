!pip install ultralytics --quiet

import gc
from ultralytics import YOLO
import matplotlib.pyplot as plt

gc.collect()

# Завантаження моделі
yolo_model = YOLO("yolo11n.pt")

# Навчання моделі
yolo_model.train(
    data="/kaggle/working/microsoft_aug_v2/data.yaml",  # шлях до dataset
    epochs=20,                                    # кількість епох
    imgsz=288,                                    # розмір зображень
    batch=2,                                      # розмір батчу
    workers=1,                                    # кількість потоків
    pretrained=True,                              # використати попередньо навчений вес
    optimizer="AdamW"                             # оптимізатор
)

# Валідація моделі
val_results = yolo_model.val()

# Побудова Precision-Recall кривої
pr_data = val_results.curves_results[0]
recall = pr_data[0].ravel()
precision = pr_data[1].ravel()

plt.figure(figsize=(6,6))
plt.plot(recall, precision, color='blue', lw=2, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("precision_recall_curve.png")
plt.show()

# Побудова F1-score кривої
f1_data = val_results.curves_results[1]
conf = f1_data[0].ravel()
f1 = f1_data[1].ravel()

plt.figure(figsize=(6,4))
plt.plot(conf, f1, color='green', lw=2, label="F1-score")
plt.xlabel("Confidence")
plt.ylabel("F1")
plt.title("F1-score vs Confidence")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("f1_curve.png")
plt.show()

# Побудова mAP50 кривої
map50_data = val_results.curves_results[2]
conf = map50_data[0].ravel()
map50 = map50_data[1].ravel()

plt.figure(figsize=(6,4))
plt.plot(conf, map50, color='red', lw=2, label="mAP50")
plt.xlabel("Confidence")
plt.ylabel("mAP50")
plt.title("mAP50 vs Confidence")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("map50_curve.png")
plt.show()
