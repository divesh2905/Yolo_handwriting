from ultralytics import YOLO
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='runs/handwriting/iam_finetuned/weights/best.pt',
                    help='Path to YOLOv8 weights file')
parser.add_argument('--source',  default='images/test',
                    help='Image or directory to run inference on')
parser.add_argument('--conf',    type=float, default=0.25,
                    help='Confidence threshold')
parser.add_argument('--project', default='runs',
                    help='Base project directory for outputs')
parser.add_argument('--name',    default='predict',
                    help='Subdirectory under project for inference outputs')
args = parser.parse_args()

# Load model
model = YOLO(args.weights)

# Run prediction and save annotated images
results = model.predict(
    source=args.source,
    conf=args.conf,
    save=True,
    project=args.project,
    name=args.name
)

output_dir = Path(args.project) / args.name
print(f'Inference complete. Outputs saved to {output_dir}')