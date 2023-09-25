# Intro
This repo is part of my Final paper. 

The job was to develop a stereo vision system to be part of the autonomous car project in Instituto Mauá de Tecnologia (IMT).

Here you'll find the codes, python virtual environment, and the chessboard used to calibrate the cameras.

# Hardware

The goal was to run all this in the edge, in this case, a Jetson Nano.

# Set Up

For future consults, I'll give an step-by-step on setting up the Jetson Nano, as it's always turned bad when I had to do it.

We'll run Yolov8, and it requires python3.8.

Jetpack SDK (Jetson OS from Nvidia) comes with pre-installed python2.7 and python3.6. Therefore, the plan is to create an python virtual environment to install python3.8.

## Install Jetpack
To install the Jetpack into the Jetson Nano, follow the steps in this [link](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write).

Choose your OS and then boot the Jetson.

## Increase swap
Follow the steps in this [tutorial](https://youtu.be/uvU8AXY1170?t=650)

## Enable nvcc
Write on terminal
```bash
nvcc --version
```

If it result in `bash: nvcc: command not found`, it'll be necessary to add `nvcc` to the `PATH`.

Write in terminal
```bash
vi /home/$USER/.bashrc
```

Go to the end of the final and add
```bash
export PATH="/usr/local/cuda-10.2/bin:$PATH"

export LD_LIBRARY_PATH="/usr/local/cuda-10.2/LIB64:$LD_LIBRARY_PATH"
```

Close and open the terminal again. Write `nvcc --version` and the Cuda version should be displayed.

## Set up environment

Update sudo 
```bash
sudo apt-get update
```

Install python3.8
```bash
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip \
libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev
```

Create env
```bash
python3.8 -m venv venv # Rename your virtual environment if you want
```

To activate the env
```bash
source venv/bin/activate
```

Upgrade pip
```bash
pip install --upgrade pip
```

## Dependencies
### Install opencv
```bash
pip install opencv-python
```

### Install numpy
```bash
pip install numpy
```

### Install yolov8
```bash
sudo apt-get install libpython3.8-dev
```
```bash
pip install ultralytics
```
### Install pytorch

We cannot install pytorch defalut packege on PiPy because it hasn't CUDA compatibility.

Thus, follow the steps:

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