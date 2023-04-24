import cv2 as cv
import numpy as np
#import serial
import time
from datetime import datetime
#from serial.tools.list_ports import comports
#from beeprint import pp
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "video_416_p.avi"
results = model.predict(video_path, imgsz=416, conf=0.5)
