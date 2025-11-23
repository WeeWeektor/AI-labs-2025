from ultralytics import YOLO
import cv2
from pathlib import Path

MODEL_PATH = "/kaggle/working/runs/detect/train/weights/best.pt"
INPUT_VIDEO = "/kaggle/input/microsoft-logo-video/YTDown.com_YouTube_MICROSOFT-NEW-LOGO-NEW-LOOK-VIDEO_Media_R0s-qKJzppc_001_720p.mp4"
OUTPUT_VIDEO = "/kaggle/working/video output.mp4"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
LINE_WIDTH = 2
FONT_SCALE = 0.5
SHOW_LABELS = True
SHOW_CONF = True


def process_video_simple():
    print("=" * 50)
    print("Автоматична обробка")
    print("=" * 50)

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=INPUT_VIDEO,
        save=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        show_labels=SHOW_LABELS,
        show_conf=SHOW_CONF,
        line_width=LINE_WIDTH,
        project="video_results",
        name="detection",
        verbose=True
    )

    print("Відео збережено у: video_results/detection/")
    return results


if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        print(f"Модель не знайдено: {MODEL_PATH}")
        exit(1)

    if not Path(INPUT_VIDEO).exists():
        print(f"Відео не знайдено: {INPUT_VIDEO}")
        exit(1)

    process_video_simple()
