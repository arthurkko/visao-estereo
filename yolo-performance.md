# GPU and Image Size performance

## NVIDIA GeForce 840M 2004.5MB
YOLOR 🚀 2023-4-17 torch 1.10.1+cu102 CUDA:0 (NVIDIA GeForce 840M, 2004.5MB)
 
### yolov7-tiny 
* 1280 - 124ms/8fps
* 640 - 60ms/16fps
* 640_p - 60ms/16fps
* 416_p - 30ms/33fps
Model Summary: 200 layers, 6219709 parameters, 229245 gradients

### yolov8n
* 640_p - 50ms/20fps
* 416_p - 24ms/41fps


## NVIDIA Tegra X1 3962.90625MB
* 640 - 175ms/5fps
* 416 - 94ms/10fps
* 416_p - 86ms/11fps
