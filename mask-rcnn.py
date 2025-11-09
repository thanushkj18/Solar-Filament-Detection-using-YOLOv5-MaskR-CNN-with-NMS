!pip install -q tensorflow==2.11.0 keras==2.11.0 scikit-image imgaug #install dependencies
!git clone https://github.com/matterport/Mask_RCNN.git

# Install detectron2 from GitHub (latest version)
!pip install git+https://github.com/facebookresearch/detectron2.git

import os 
import random
import numpy as np
import torch
from detectron2.engine import DefaultTrainer, launch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances


# Register your custom datasets
register_coco_instances("train_dataset", {}, "/content/drive/MyDrive/SolarFilamentDetection/train_data_mask/train_data/annotations/train_annotations.json", "/content/drive/MyDrive/SolarFilamentDetection/train_data_mask/train_data/images/train")
register_coco_instances("val_dataset", {}, "/content/drive/MyDrive/SolarFilamentDetection/train_data_mask/train_data/annotations/val_annotations.json", "/content/drive/MyDrive/SolarFilamentDetection/train_data_mask/train_data/images/val")

from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

# Initialize config
cfg = get_cfg()

# Load base config from model zoo (COCO pre-trained)
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Register your datasets (already done earlier)
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.TEST = ("val_dataset",)

# DataLoader settings
cfg.DATALOADER.NUM_WORKERS = 2

# Pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Solver / training settings
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000  # Adjust for your dataset size
cfg.SOLVER.STEPS = []       # No learning rate decay
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only 1 class: 'Filament'

# Output directory
cfg.OUTPUT_DIR = "/content/drive/MyDrive/SolarFilamentDetection/output"

# ✅ Force CPU if no GPU is available
if not torch.cuda.is_available():
    print("⚠️ No GPU detected — running on CPU")
    cfg.MODEL.DEVICE = "cpu"


from detectron2.engine import DefaultTrainer   #training
import os

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import matplotlib.pyplot as plt
import torch
import os

# Initialize the config
cfg = get_cfg()

# Load the base config and pretrained model
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/SolarFilamentDetection/output/model_final.pth"  # Make sure this path is correct

# Set device
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set prediction confidence threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lowered to increase chance of detection

# Initialize the predictor
predictor = DefaultPredictor(cfg)

# Load the input image
image_path = "/content/drive/MyDrive/SolarFilamentDetection/detected_result.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
outputs = predictor(image)
instances = outputs["instances"].to("cpu")

# Print how many instances are detected
print(f"🔍 Number of predicted instances: {len(instances)}")

# Prepare output path
output_path = "/content/drive/MyDrive/SolarFilamentDetection/output/output_image.jpg"

# Handle output visualization and saving
output_image = image.copy()
if len(instances.pred_masks) > 0:
    for mask in instances.pred_masks:
        mask = mask.numpy()
        output_image[mask] = [0, 255, 0]  # Green mask
    print("✅ Masks detected and applied.")
else:
    print("⚠️ No masks found. Saving original image.")

# Save output image
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
print(f"📁 Output saved at: {output_path}")
