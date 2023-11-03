# Object Detection with Stereo Vision System for Autonomous Cars (Jetson Nano)

## Introduction
This repository is part of my final paper, where I developed a stereo vision system for the autonomous car project at Instituto Mauá de Tecnologia (IMT). Here, you will find the code developed.

## Hardware

Our goal is to run this system on the edge, specifically on a Jetson Nano.

## Set Up

For your convenience, I've provided a step-by-step guide to setting up the Jetson Nano, as it can be a bit challenging:

We'll run Yolov8, and it requires python3.8.

### Install Jetpack
Jetpack SDK (Jetson OS from Nvidia) comes with pre-installed python2.7 and python3.6. Therefore, the plan is to create an python virtual environment to install python3.8.

1. Follow the steps outlined in this [link](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write) to install Jetpack on the Jetson Nano. 
2. Choose your OS and then boot the Jetson.

### Increase swap
For better performance, it's recommended to increase the swap size. Follow the instructions in this [tutorial](https://youtu.be/uvU8AXY1170?t=650)

### Enable nvcc
1. Open your terminal and run:
```bash
nvcc --version
```

If it result in `bash: nvcc: command not found`, it'll be necessary to add `nvcc` to the `PATH`.

2. Run the following command in your terminal:
```bash
vi /home/$USER/.bashrc
```

Then, go to the end of the file and add:
```bash
export PATH="/usr/local/cuda-10.2/bin:$PATH"

export LD_LIBRARY_PATH="/usr/local/cuda-10.2/LIB64:$LD_LIBRARY_PATH"
```

3. Close and open the terminal again. Running `nvcc --version` should display the Cuda version.

### Set up environment

1. Update your system: 
```bash
sudo apt-get update
```

2. Install python3.8 and required packages:
```bash
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip \
libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev
```

3. Create virtual envireonment (you can rename it if desired):
```bash
python3.8 -m venv venv
```

4. Activate the virtual environment:
```bash
source venv/bin/activate
```

5. Upgrade `pip`
```bash
pip install --upgrade pip
```

### Dependencies
1. Install opencv
```bash
pip install opencv-python
```

2. Install numpy
```bash
pip install numpy
```

3. Install ultralitics

Ultralitics is the name of the company who developed the 8th version of YOLO (You Only Look Once).
```bash
sudo apt-get install libpython3.8-dev
```
```bash
pip install ultralytics
```
4. Install pytorch

We cannot install pytorch defalut packege on PiPy because it hasn't CUDA compatibility.

Thus, we'll do as follows:

```bash
pip install -U pip wheel gdown
```

```bash
# pytorch 1.11.0
gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
# torchvision 0.12.0
gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV
pip install torch-*.whl torchvision-*.whl
```