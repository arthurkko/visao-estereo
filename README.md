# Intro
This repo is part of my Final paper. 

The job was to develop a stereo vision system to be part of the autonomous car project in Instituto Mauá de Tecnologia (IMT).

Here you'll find the codes, python virtual environment, and the chessboard used to calibrate the cameras.

# Hardware

The goal was to run all this in the edge, in this case, a Jetson Nano.

## Set Up

For future consults, I'll give an step-by-step on setting up the Jetson Nano, as it's always turned bad when I had to do it.

We'll run Yolov8, and it requires python3.8.

Jetpack SDK (Jetson OS from Nvidia) comes with pre-installed python2.7 and python3.6. Therefore, the plan is to create an python virtual environment to install python3.8.

### Increase swap
Follow the steps in this [tutorial](https://youtu.be/uvU8AXY1170?t=650)

### Enable nvcc
Digite no terminal
```bash
nvcc --version
```

Caso resulte em `bash: nvcc: command not found`, é necessario adicionar o nvcc ao `PATH`.

Para isso digite no terminal
```bash
vi /home/$USER/.bashrc
```

Vá para o final do arquivo e insira as seguintes linhas
```bash
export PATH="/usr/local/cuda-10.2/bin:$PATH"
```
```bash
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/LIB64:$LD_LIBRARY_PATH"
```

Feche o terminal e abra novamente. Insira `nvcc --version` e a versão do cuda deve aparecer.

### Set up environment

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

### Dependencies
Install opencv
```bash
pip install opencv-python
```

Install numpy
```bash
pip install numpy
```

Install yolov8
```bash
sudo apt-get install libpython3.8-dev
```
```bash
pip install ultralytics
```
Install pytorch
Follow the steps

```bash
pip install -U pip wheel gdown```

```bash
# pytorch 1.11.0
gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
# torchvision 0.12.0
gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV
pip install torch-*.whl torchvision-*.whl
```