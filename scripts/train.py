from ultralytics import YOLO

# Configuration constants
MODEL_PATH = 'yolov8n.pt'
DATA_CFG   = 'data.yaml'
EPOCHS     = 50
IMGSZ      = 640
BATCH      = 16
PROJECT    = 'runs/handwriting'
NAME       = 'iam_finetuned'
CACHE      = True

if __name__ == '__main__':
    model = YOLO(MODEL_PATH)
    model.train(
        data=DATA_CFG,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        cache=CACHE
    )