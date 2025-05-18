# CVD_Object_Detection

## Project Overview

This repository contains a comparative study of four state-of-the-art object detection models—RT-DETR, YOLOv8s, YOLOv8s-Swin, and YOLOv12m—evaluated on a custom Canadian Vehicle Dataset (CVD2) under challenging conditions. We analyze training performance, convergence, accuracy (mAP @0.50–0.95), loss dynamics, and qualitative results.


## Features

End-to-end training scripts for each architecture

Automated evaluation of detection metrics (precision, recall, F1, mAP)

Visualizations: PR curves, confusion matrices, qualitative inference samples

## Requirements

Python 3.8+

PyTorch 1.12+

Ultralytics 

torchvision

OpenCV, NumPy, Matplotlib

YAML


## Configuration

Update paths in data/cvd2_dataset.yaml to your local dataset directories.

Edit the global parameters (batch size, epochs, learning rates).

## Training

To train each model:

#### RT-DETR
python src/train/train_rtdetr.py --config data/cvd2_dataset.yaml --epochs 200

#### YOLOv8s
python src/train/train_yolov8s.py --config data/cvd2_dataset.yaml --epochs 50

#### YOLOv8s-Swin
python src/train/train_yolov8s_swin.py --config data/cvd2_dataset.yaml --epochs 200

#### YOLOv12m
python src/train/train_yolo12m.py --config data/cvd2_dataset.yaml --epochs 50



## Contributing

Fork the repository

Create a feature branch (git checkout -b feature/YourFeature)

Commit your changes (git commit -m 'Add new feature')

Push to branch (git push origin feature/YourFeature)

Open a pull request


## Authors 

Rohit Singh Nitwal

Special thanks to the Ultralytics team.
