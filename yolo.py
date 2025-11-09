!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt comet_ml  # install

import torch
import utils
display = utils.notebook_init()  # checks

# Train YOLOv5s on COCO128 for 3 epochs
# !rm /content/Train_data/labels/*.cache
!python train.py --img 640 --batch 2 --epochs 40 --data /content/drive/MyDrive/SolarFilamentDetection/yolov5/data/custom.yaml --weights yolov5s.pt --cache

!python detect.py --weights runs/train/exp3/weights/last.pt --img 640 --conf 0.25 --source /content/drive/MyDrive/SolarFilamentDetection/76.jpg